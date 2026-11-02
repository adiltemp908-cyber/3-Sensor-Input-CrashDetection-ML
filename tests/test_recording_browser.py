"""Verify lookup identities, exclusions and CSV validation with small artificial fixtures."""
import io
import json

import numpy as np
import pytest

from crashdetect.vzcrash.browser import RecordingLibrary
from crashdetect.vzcrash.csv_input import read_clip
from crashdetect.vzcrash.data import digest


@pytest.fixture
def library(tmp_path):
    prepared, model = tmp_path / "prepared", tmp_path / "model"
    prepared.mkdir(); model.mkdir()
    catalog = {"rejected": {"test": [{"event_id": 99, "reason": "Unequal sensor lengths"}]}}
    (prepared / "catalog.json").write_text(json.dumps(catalog))
    (model / "selection.json").write_text(json.dumps({"catalog_sha256": digest(prepared / "catalog.json")}))
    np.save(model / "training_event_ids.npy", [10])
    for split, ids, valid, eligible in [
        ("train", [10, 12], [True, False], [True, False]),
        ("validation", [44], [True], [True]),
        ("test", [80, 99], [True, False], [True, True]),
    ]:
        n = len(ids)
        x = np.zeros((n * 40, 7), dtype=np.float32)
        for i, event_id in enumerate(ids):
            x[i*40:(i+1)*40, 0] = event_id
        np.save(prepared / f"{split}_x.npy", x)
        np.savez(prepared / f"{split}_meta.npz", event_id=ids, valid=valid, eligible=eligible,
                 offsets=np.arange(n+1)*40, provider_label=np.full(n, 2),
                 vehiclesize=np.full(n, "Light"), gyro_is_hd=np.full(n, True))
    def predictor(path, x):
        return {"score": .1, "threshold": .5, "crash_clip": False}
    return RecordingLibrary(prepared, model, predictor)


def test_numbering_crosses_partitions_and_is_not_event_id(library):
    assert library.summary()["usable_total"] == 3
    assert library.record(number=1)["event_id"] == 10
    assert library.record(number=2)["split"] == "validation"
    assert library.record(number=3)["event_id"] == 80
    record = library.record(number=1, split="test")
    assert record["recording_number"] == 3
    assert record["rows"][0][0] == 80
    assert library.record(event_id=10)["used_for_training"] is True
    assert record["used_for_training"] is False
    with pytest.raises(ValueError, match="not found"):
        library.record(event_id=1)
    for bad in [0, -1, 4]:
        with pytest.raises(ValueError):
            library.record(number=bad)


def test_duplicate_is_not_training_and_unavailable_values_are_hidden(library):
    record = library.record(event_id=12)
    assert record["split"] == "train"
    assert not record["used_for_training"]
    assert record["prediction"] is None
    assert record["recording_number"] is None
    assert record["rows"][0][0] == 12
    unavailable = library.record(event_id=99)
    assert unavailable["rows"] == []
    assert unavailable["reason"] == "Unequal sensor lengths"
    assert unavailable["prediction"] is None


def csv_text():
    return "time_s,ax,ay,az,gx,gy,gz,speed\n" + "\n".join(
        f"{i/100:.2f},0,0,9.81,0,0,0,10" for i in range(40))


def test_csv_only_browser_needs_no_dataset(tmp_path):
    from crashdetect.vzcrash.browser import CsvLibrary
    model = tmp_path / "model"
    model.mkdir()
    (model / "selection.json").write_text(json.dumps({"selected": "physical"}))
    library = CsvLibrary(model, predictor=lambda path, clip: {"score": .1, "crash_clip": False})
    assert library.summary()["csv_only"] is True
    original_identity = library.summary()["model_identity"]
    (model / "selection.json").write_text(json.dumps({"selected": "physical", "threshold": 2.0}))
    assert CsvLibrary(model).summary()["model_identity"] != original_identity
    result = library.upload(csv_text().encode())
    assert result["sample_count"] == 40
    assert result["dataset_label"] == "Unknown"
    assert result["prediction"]["crash_clip"] is False
    with pytest.raises(ValueError, match="--prepared"):
        library.record(event_id=1)


def test_upload_has_no_invented_dataset_membership(library):
    result = library.upload(("\ufeff" + csv_text()).encode("utf-8"))
    assert result["split"] == "custom"
    assert result["dataset_label"] == "Unknown"
    assert result["event_id"] is None
    assert result["sample_count"] == 40
    assert result["prediction"]["score"] == .1


@pytest.mark.parametrize("bad", [
    csv_text().replace("0.01,", "0.03,", 1),
    csv_text().replace("9.81", "nan", 1),
    csv_text().replace(",10", ",-1", 1),
    csv_text().replace("speed", "wrong"),
])
def test_bad_csv_rejected(bad):
    with pytest.raises(ValueError):
        read_clip(io.StringIO(bad))


def test_http_lookup_validation_origin_and_exclusive_port(library):
    from http.client import HTTPConnection
    import threading
    from crashdetect.vzcrash.browser import LocalServer, make_handler
    server = LocalServer(("127.0.0.1", 0), make_handler(library))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        with pytest.raises(OSError):
            LocalServer(("127.0.0.1", server.server_port), make_handler(library))
        connection.request("GET", "/api/record?number=2")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["split"] == "validation"
        connection.request("GET", "/api/record?number=0")
        response = connection.getresponse()
        assert response.status == 400
        assert "between" in json.loads(response.read())["error"]
        connection.request("POST", "/api/predict", body=csv_text(), headers={"Origin": "https://unrelated.example"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("GET", "/api/summary", headers={"Host": "unrelated.example"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_repeated_launch_reuses_existing_browser(library):
    import threading
    from crashdetect.vzcrash.browser import start_or_reuse
    server, url = start_or_reuse(library, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        reused, existing_url = start_or_reuse(library, server.server_port)
        assert reused is None
        assert existing_url == url
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_occupied_non_app_port_falls_back_without_stopping_it(library):
    from http.server import BaseHTTPRequestHandler
    from http.client import HTTPConnection
    import threading
    from crashdetect.vzcrash.browser import LocalServer, start_or_reuse

    class OtherApp(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    other = LocalServer(("127.0.0.1", 0), OtherApp)
    thread = threading.Thread(target=other.serve_forever, daemon=True)
    thread.start()
    new_server = None
    connection = HTTPConnection("127.0.0.1", other.server_port, timeout=5)
    try:
        new_server, url = start_or_reuse(library, other.server_port)
        assert new_server is not None
        assert new_server.server_port != other.server_port
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        connection.close()
        if new_server is not None:
            new_server.server_close()
        other.shutdown()
        thread.join(timeout=5)
        other.server_close()
