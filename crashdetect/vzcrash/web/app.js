"use strict";
const $ = id => document.getElementById(id);
let summary, current, page = 0, busy = false;
const fmt = n => Number(n).toLocaleString("en-US");
const names = {train:"Training", validation:"Validation", test:"Test", custom:"Custom upload"};
function message(text, error=false) { $("message").textContent=text; $("message").classList.toggle("error",error); }
function setBusy(value) { busy=value; for(const id of ["load-button","upload-button","previous-record","next-record"]) $(id).disabled=value; }
async function request(url, options) { const response=await fetch(url,options); const data=await response.json(); if(!response.ok) throw new Error(data.error||"Request failed."); return data; }
function switchTab(upload) { $("browse-pane").hidden=upload; $("upload-pane").hidden=!upload; $("browse-tab").setAttribute("aria-selected",String(!upload)); $("upload-tab").setAttribute("aria-selected",String(upload)); message(""); }
$("browse-tab").onclick=()=>switchTab(false); $("upload-tab").onclick=()=>switchTab(true);
function updateMode() {
  const numbered=$("lookup-mode").value==="number";
  $("partition-control").hidden=!numbered;
  $("number-label").textContent=numbered?"Recording number (starts at 1)":"Event ID";
  const split=$("partition").value;
  const max=summary?(split==="all"?summary.usable_total:summary.splits[split].usable):0;
  $("lookup-hint").textContent=numbered?`Enter 1–${fmt(max)}. Numbers cover usable recordings in provider order; “All” lists training first, then validation, then test. Excluded records are accessible by event ID.`:"An event ID is the dataset’s original identifier. Try 19088 (test), 9322 (training), or 18909 (validation).";
  if(numbered) $("lookup-value").max=String(max); else $("lookup-value").removeAttribute("max");
}
$("lookup-mode").onchange=()=>{ $("lookup-value").value=$("lookup-mode").value==="number"?"1":"19088";updateMode(); };
$("partition").onchange=()=>{$("lookup-value").value="1";updateMode();};
async function load(query) {
  if(busy) return;
  setBusy(true); $("record").hidden=true; message("Reading the recording and running the saved detector…");
  try { render(await request(`/api/record?${new URLSearchParams(query)}`)); message(""); }
  catch(error) { message(error.message,true); }
  finally { setBusy(false); updateNav(); }
}
$("lookup-form").onsubmit=event=>{event.preventDefault();const mode=$("lookup-mode").value;load({[mode]:$("lookup-value").value,split:$("partition").value});};
$("upload-form").onsubmit=async event=>{
  event.preventDefault();if(busy)return;const file=$("csv-file").files[0];if(!file)return;
  if(file.size>5*1024*1024){message("Choose a CSV file smaller than 5 MB.",true);return;}
  setBusy(true);$("record").hidden=true;message("Checking the CSV and analyzing the recording…");
  try { const data=await request("/api/predict",{method:"POST",headers:{"Content-Type":"text/csv"},body:file});data.filename=file.name;render(data);message(""); }
  catch(error){message(error.message,true);}finally{setBusy(false);updateNav();}
};
function updateNav(){const n=current?.recording_number;$("previous-record").disabled=busy||!n||n<=1;$("next-record").disabled=busy||!n||n>=summary?.usable_total;}
function navigate(delta){if(!current?.recording_number)return;switchTab(false);$("lookup-mode").value="number";$("partition").value="all";$("lookup-value").value=current.recording_number+delta;updateMode();load({number:current.recording_number+delta,split:"all"});}
$("previous-record").onclick=()=>navigate(-1);$("next-record").onclick=()=>navigate(1);
function render(data){
  current=data;page=0;$("record").hidden=false;
  $("record-source").textContent=data.split==="custom"?"YOUR RECORDING":"VZCRASH RECORDING";
  $("record-title").textContent=data.event_id===null?data.filename:`Event ${data.event_id}`;
  $("split-badge").textContent=names[data.split];$("split-badge").className=`badge ${data.split}`;
  $("label-badge").textContent=`Dataset label: ${data.dataset_label}`;
  $("record-status").textContent=data.status;
  $("ordinal").textContent=data.recording_number?`${fmt(data.recording_number)} of ${fmt(summary.usable_total)}`:"Not numbered";
  $("length").textContent=data.sample_count?`${data.duration_s.toFixed(2)} s · ${fmt(data.sample_count)} samples`:"Sensor values unavailable";
  $("vehicle").textContent=data.vehiclesize;$("gyro").textContent=data.gyro_resolution;
  const prediction=data.prediction;
  $("prediction-title").textContent=prediction?(prediction.crash_clip?"Crash detected":"No crash detected"):"Prediction unavailable";
  $("prediction-title").className=prediction?(prediction.crash_clip?"positive":"negative"):"";
  $("score").textContent=prediction?prediction.score.toFixed(6):"—";
  $("threshold").textContent=prediction?prediction.threshold.toFixed(6):"—";
  $("prediction-context").textContent=prediction?(data.split==="custom"?"The uploaded file has no verified dataset label.":`The dataset marks this event as ${data.dataset_label.toLowerCase()}.`):data.reason||"This recording was excluded from the experiment.";
  $("record-note").textContent=data.reason?`Values are not displayed: ${data.reason}. Stored placeholders are not sensor measurements.`:!data.usable?"Excluded duplicate: measurements can be inspected, but no evaluation prediction is presented.":data.used_for_training?"The model learned from this recording. Its prediction is not independent test evidence.":data.split==="validation"?"This recording helped select the model and threshold; it is not an independent test example.":data.split==="test"?"This recording belongs to the retained held-out test set. Inspecting one event is not a new performance evaluation.":"CSV units and sensor orientation must be correct before upload. Numeric checks cannot verify the physical sensor setup.";
  $("measurements").hidden=!data.rows.length;$("download").disabled=!data.rows.length;
  if(data.rows.length){drawCharts();renderRows();}updateNav();
}
function renderRows(){
  const rows=current.rows,start=page*50,end=Math.min(start+50,rows.length);const fragment=document.createDocumentFragment();
  for(let i=start;i<end;i++){const tr=document.createElement("tr");for(const value of [i+1,(i/100).toFixed(2),...rows[i].map(v=>v.toFixed(6))]){const td=document.createElement("td");td.textContent=value;tr.appendChild(td);}fragment.appendChild(tr);}
  $("sample-rows").replaceChildren(fragment);$("row-range").textContent=`Rows ${fmt(start+1)}–${fmt(end)} of ${fmt(rows.length)}`;
  $("previous-page").disabled=page===0;$("first-page").disabled=page===0;$("next-page").disabled=end===rows.length;
  document.querySelector(".table-scroll").scrollTop=0;
}
$("first-page").onclick=()=>{page=0;renderRows();};$("previous-page").onclick=()=>{page--;renderRows();};$("next-page").onclick=()=>{page++;renderRows();};
$("download").onclick=()=>{
  const csv="time_s,ax,ay,az,gx,gy,gz,speed\n"+current.rows.map((row,i)=>[(i/100).toFixed(2),...row].join(",")).join("\n")+"\n";
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));const a=document.createElement("a");a.href=url;a.download=current.event_id===null?"uploaded_recording.csv":`event_${current.event_id}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function chart(id,columns){
  const canvas=$(id),box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;if(!box.width)return;
  canvas.width=box.width*dpr;canvas.height=box.height*dpr;const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
  const width=box.width,height=box.height,left=43,right=10,top=13,bottom=26;const rows=current.rows;
  let min=Infinity,max=-Infinity;for(const row of rows)for(const col of columns){min=Math.min(min,row[col]);max=Math.max(max,row[col]);}
  const pad=Math.max((max-min)*.12,.01);min-=pad;max+=pad;
  const X=i=>left+i/(rows.length-1)*(width-left-right),Y=v=>top+(max-v)/(max-min)*(height-top-bottom);
  ctx.font="10px Segoe UI, Arial";ctx.fillStyle="#607078";
  for(let i=0;i<=2;i++){const val=min+(max-min)*i/2,y=Y(val);ctx.strokeStyle="#e5ebed";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();ctx.textAlign="right";ctx.fillText(Math.abs(val)>=1000?val.toExponential(1):val.toFixed(2),left-6,y+3);}
  const colors=["#086d68","#326eb3","#a9601e"];
  columns.forEach((col,j)=>{ctx.strokeStyle=colors[j];ctx.lineWidth=1.3;ctx.beginPath();rows.forEach((row,i)=>{if(i===0)ctx.moveTo(X(i),Y(row[col]));else ctx.lineTo(X(i),Y(row[col]));});ctx.stroke();});
  ctx.fillStyle="#607078";ctx.textAlign="left";ctx.fillText("0 s",left,height-7);ctx.textAlign="right";ctx.fillText(`${((rows.length-1)/100).toFixed(2)} s`,width-right,height-7);
}
function drawCharts(){if(!current?.rows.length||!$("sensor-graphs").open)return;chart("accel-chart",[0,1,2]);chart("gyro-chart",[3,4,5]);chart("speed-chart",[6]);}
$("sensor-graphs").addEventListener("toggle",()=>{
  $("sensor-graphs").querySelector("summary").textContent=$("sensor-graphs").open?"Hide sensor graphs":"Show sensor graphs";
  drawCharts();
});
window.addEventListener("resize",drawCharts);
(async()=>{try{
  summary=await request("/api/summary");
  if(summary.csv_only){
    document.querySelector(".stats").hidden=true;
    $("browse-tab").disabled=true;
    $("browse-tab").title="Start the browser with --prepared to browse a downloaded dataset.";
    switchTab(true);
    message("Ready. Choose a CSV from the examples folder or upload your own recording.");
  }else{
    $("total").textContent=fmt(summary.usable_total);
    for(const name of ["train","validation","test"])$(`${name}-count`).textContent=fmt(summary.splits[name].usable);
    updateMode();await load({event_id:19088});
  }
}catch(error){message(error.message,true);}})();
