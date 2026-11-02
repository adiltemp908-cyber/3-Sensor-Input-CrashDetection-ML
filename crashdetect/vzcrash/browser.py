"""Local recording browser. Run with python -m crashdetect.vzcrash.browser."""
import argparse
import errno
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import socket
from urllib.parse import parse_qs, urlsplit
from urllib.request import build_opener, ProxyHandler
import webbrowser

import numpy as np

from .csv_input import COLUMNS, read_clip
from .data import digest

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
LABELS = {0: "Crash", 1: "Near miss", 2: "Normal driving"}
MAX_UPLOAD = 5 * 1024 * 1024


class LocalServer(HTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def predict(model_dir, clip):
    # Bound CPU work explicitly; also avoids Windows physical-core discovery warnings.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    from threadpoolctl import threadpool_limits
    from .experiment import predict_clip
    with threadpool_limits(limits=4):
        return predict_clip(model_dir, clip)


class RecordingLibrary:
    def __init__(self, prepared, model_dir, predictor=predict):
        self.prepared, self.model_dir = Path(prepared), Path(model_dir)
        self.predictor = predictor
        catalog_path = self.prepared / "catalog.json"
        self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        config = json.loads((self.model_dir / "selection.json").read_text(encoding="utf-8"))
        if digest(catalog_path) != config["catalog_sha256"]:
            raise ValueError("The prepared catalog does not match the trained model.")
        self.trained_ids = set(np.load(self.model_dir / "training_event_ids.npy", allow_pickle=False).tolist())
        self.meta, self.arrays, self.indices, self.ranks = {}, {}, {}, {}
        self.by_id, self.starts, self.rejected = {}, {}, {}
        count = 0
        for split in SPLITS:
            with np.load(self.prepared / f"{split}_meta.npz", allow_pickle=False) as stored:
                self.meta[split] = {key: stored[key] for key in (
                    "event_id", "offsets", "valid", "eligible", "provider_label", "vehiclesize", "gyro_is_hd")}
            meta = self.meta[split]
            self.arrays[split] = np.load(self.prepared / f"{split}_x.npy", mmap_mode="r", allow_pickle=False)
            self.indices[split] = np.flatnonzero(meta["valid"])
            rank = np.full(len(meta["event_id"]), -1, dtype=np.int64)
            rank[self.indices[split]] = np.arange(len(self.indices[split]))
            self.ranks[split] = rank
            self.starts[split] = count
            count += len(self.indices[split])
            for i, event_id in enumerate(meta["event_id"]):
                if int(event_id) in self.by_id:
                    raise ValueError("Duplicate event IDs in prepared data.")
                self.by_id[int(event_id)] = (split, i)
            for rejected in self.catalog.get("rejected", {}).get(split, []):
                self.rejected[int(rejected["event_id"])] = rejected["reason"]
        self.total = count

    def summary(self):
        return {"usable_total": self.total, "original_total": len(self.by_id),
                "training_used": len(self.trained_ids), "splits": {
                    s: {"usable": len(self.indices[s]), "original": len(self.meta[s]["event_id"])} for s in SPLITS}}

    def locate(self, event_id=None, number=None, split="all"):
        if (event_id is None) == (number is None):
            raise ValueError("Enter either an event ID or a recording number.")
        if event_id is not None:
            if event_id not in self.by_id:
                raise ValueError(f"Event ID {event_id} was not found. Event IDs are not sequential recording numbers.")
            return self.by_id[event_id]
        if split not in ("all",) + SPLITS:
            raise ValueError("Unknown partition.")
        limit = self.total if split == "all" else len(self.indices[split])
        if number < 1 or number > limit:
            raise ValueError(f"Recording number must be between 1 and {limit:,}.")
        if split == "all":
            for name in SPLITS:
                if number <= self.starts[name] + len(self.indices[name]):
                    return name, int(self.indices[name][number - self.starts[name] - 1])
        return split, int(self.indices[split][number - 1])

    def record(self, **query):
        split, i = self.locate(**query)
        meta = self.meta[split]
        event_id = int(meta["event_id"][i])
        eligible, valid = bool(meta["eligible"][i]), bool(meta["valid"][i])
        reason = self.rejected.get(event_id)
        if not eligible:
            status = "Excluded: identical-input duplicate group"
        elif not valid:
            status = "Unavailable: sensor alignment failed"
        else:
            status = {"train": "Used for training" if event_id in self.trained_ids else "Not used for training",
                      "validation": "Used for model / threshold selection", "test": "Used for held-out evaluation"}[split]
        rank = int(self.ranks[split][i])
        response = {"event_id": event_id, "split": split, "status": status,
                    "used_for_training": split == "train" and event_id in self.trained_ids,
                    "eligible": eligible, "usable": valid, "reason": reason,
                    "dataset_label": LABELS[int(meta["provider_label"][i])],
                    "provider_label": int(meta["provider_label"][i]),
                    "vehiclesize": str(meta["vehiclesize"][i]),
                    "gyro_resolution": "High" if meta["gyro_is_hd"][i] else "Standard",
                    "recording_number": self.starts[split] + rank + 1 if rank >= 0 else None,
                    "partition_number": rank + 1 if rank >= 0 else None,
                    "prediction": None, "rows": [], "sample_count": 0, "duration_s": None}
        # Sensor failures were stored as zeros by preparation. Never display these placeholders as measurements.
        if reason or (not valid and eligible):
            return response
        begin, end = int(meta["offsets"][i]), int(meta["offsets"][i+1])
        clip = np.asarray(self.arrays[split][begin:end])
        response.update(rows=clip.tolist(), sample_count=len(clip), duration_s=len(clip)/100)
        if valid:
            response["prediction"] = self.predictor(self.model_dir, clip)
        return response

    def upload(self, raw):
        clip = read_clip(io.StringIO(raw.decode("utf-8-sig")))
        return {"event_id": None, "split": "custom", "status": "Uploaded file; no dataset membership assigned",
                "dataset_label": "Unknown", "provider_label": None, "vehiclesize": "Unknown",
                "gyro_resolution": "Unknown", "used_for_training": False,
                "rows": clip.tolist(), "sample_count": len(clip), "duration_s": len(clip)/100,
                "recording_number": None, "partition_number": None, "reason": None,
                "prediction": self.predictor(self.model_dir, clip), "usable": True}


def make_handler(library):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, value):
            self.send(status, json.dumps(value, allow_nan=False).encode())

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self.json(403, {"error": "Use the local address printed by the server."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{h}" for h in allowed}:
                self.json(403, {"error": "Cross-origin requests are not accepted."})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            url = urlsplit(self.path)
            assets = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if url.path in assets:
                filename = assets[url.path]
                types = {"html": "text/html", "js": "text/javascript", "css": "text/css"}
                self.send(200, (Path(__file__).parent / "web" / filename).read_bytes(),
                          types[filename.split(".")[-1]] + "; charset=utf-8")
                return
            try:
                if url.path == "/api/summary":
                    self.json(200, library.summary())
                elif url.path == "/api/record":
                    args = parse_qs(url.query)
                    query = {k: int(args[k][0]) for k in ("event_id", "number") if k in args}
                    query["split"] = args.get("split", ["all"])[0]
                    self.json(200, library.record(**query))
                else:
                    self.json(404, {"error": "Page not found."})
            except (ValueError, UnicodeError) as exc:
                self.json(400, {"error": str(exc)})
            except Exception as exc:
                self.log_error("Recording request failed: %s", exc)
                self.json(500, {"error": "Could not read or predict this recording. Check the server terminal."})

        def do_POST(self):
            if not self.local_request():
                return
            if urlsplit(self.path).path != "/api/predict":
                self.json(404, {"error": "Page not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    raise ValueError("Choose a nonempty CSV file smaller than 5 MB.")
                self.json(200, library.upload(self.rfile.read(length)))
            except (ValueError, UnicodeError) as exc:
                self.json(400, {"error": str(exc)})
            except Exception:
                self.json(500, {"error": "Prediction failed. Check the model files and server terminal."})

        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args) > 1 else "") == "200":
                return
            super().log_message(fmt, *args)

    return Handler


class CsvLibrary:
    """Use a saved model on uploads without downloading the training corpus."""

    def __init__(self, model_dir, predictor=predict):
        self.model_dir = Path(model_dir)
        self.predictor = predictor
        config = json.loads((self.model_dir / "selection.json").read_text(encoding="utf-8"))
        self.model_identity = digest(self.model_dir / "selection.json")
        from .experiment import verify_model
        verify_model(self.model_dir, config["selected"], config)

    def summary(self):
        return {"usable_total": 0, "original_total": 0, "training_used": 0,
                "csv_only": True, "model_identity": self.model_identity,
                "splits": {s: {"usable": 0, "original": 0} for s in SPLITS}}

    def record(self, **query):
        raise ValueError("Dataset browsing requires --prepared and a model trained on that prepared dataset.")

    upload = RecordingLibrary.upload


def is_recording_browser(port, summary):
    """Recognize this interface, including already-running versions without a health endpoint."""
    opener = build_opener(ProxyHandler({}))
    url = f"http://127.0.0.1:{port}"
    try:
        with opener.open(url + "/api/summary", timeout=1) as response:
            if json.loads(response.read(16384)) != summary:
                return False
        with opener.open(url, timeout=1) as response:
            html = response.read(16384).decode("utf-8")
        return "<title>Crash detection · Recording browser</title>" in html
    except (OSError, ValueError, UnicodeError):
        return False


def start_or_reuse(library, port):
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535.")
    # Deterministic fallback ports also let repeated launches find an earlier fallback instance.
    candidates = [0] if port == 0 else list(range(port, min(port + 10, 65536))) + [0]
    for candidate in candidates:
        try:
            server = LocalServer(("127.0.0.1", candidate), make_handler(library))
            return server, f"http://127.0.0.1:{server.server_port}"
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, 10048) and getattr(exc, "winerror", None) != 10048:
                raise
            if candidate and is_recording_browser(candidate, library.summary()):
                return None, f"http://127.0.0.1:{candidate}"
    raise OSError("Could not allocate a local port.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "models/hist-gradient-boosting")
    parser.add_argument("--prepared", type=Path, help="Optional prepared dataset; omit for CSV uploads only")
    args = parser.parse_args()
    try:
        library = (RecordingLibrary(args.prepared, args.model) if args.prepared
                   else CsvLibrary(args.model))
        server, url = start_or_reuse(library, args.port)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot start recording browser: {exc}\n")
    if server is None:
        print(f"Recording browser is already running: {url}\nReusing the existing interface.", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        return
    print(f"Recording browser: {url}\nKeep this terminal open. Press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
