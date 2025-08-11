const statusEl = document.getElementById('status');
const startServerBtn = document.getElementById('start-server');
const startClientBtn = document.getElementById('start-client');

const socket = io();

function append(line){
  statusEl.textContent += `\n${line}`;
}

startServerBtn.onclick = async ()=>{
  const res = await fetch('/start/server', {method:'POST'});
  const j = await res.json();
  append(`Server started pid=${j.pid}`);
}

startClientBtn.onclick = async ()=>{
  const res = await fetch('/start/client', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({client_id: `webui-${Date.now()}`})});
  const j = await res.json();
  append(`Client started pid=${j.pid} id=${j.client_id}`);
}

async function refresh(){
  const r = await fetch('/status');
  const j = await r.json();
  statusEl.textContent = JSON.stringify(j, null, 2);
}
setInterval(refresh, 1500);
refresh();

socket.on('metrics', (payload)=>{
  append(`metrics: ${JSON.stringify(payload)}`);
});
