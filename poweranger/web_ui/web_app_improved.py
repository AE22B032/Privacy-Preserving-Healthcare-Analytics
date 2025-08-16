from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
import os
import io
import zipfile
import socket
import subprocess
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
# Use threading to keep runtime simple and compatible across environments
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

@app.route("/")
def dashboard():
    return render_template("dashboard_improved.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/status")
def status():
    host = os.getenv("FLOWER_HOST", "127.0.0.1")
    port = int(os.getenv("FLOWER_PORT", "8081"))
    reachable = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            reachable = True
        except Exception:
            reachable = False
    return jsonify({"flower": {"host": host, "port": port, "reachable": reachable}})

@app.post("/start/server")
def start_server_proc():
    py = sys.executable
    # Start the Flower server using the maintained entrypoint
    cmd = [py, "-m", "poweranger.server.server_app"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PROJECT_ROOT)
    return jsonify({"started": True, "pid": proc.pid})

@app.post("/start/client")
def start_client_proc():
    data = request.get_json(silent=True) or {}
    client_id = str(data.get("client_id", "webui"))
    data_dir = data.get("data_dir")  # optional per-client dataset dir
    server_host = os.getenv("FLOWER_HOST", "127.0.0.1")
    server_port = int(os.getenv("FLOWER_PORT", "8081"))
    py = sys.executable
    # Launch standalone Flower client module
    cmd = [
        py,
        "-m",
        "poweranger.client.standalone_client",
        "--client-id",
        client_id,
        "--server-address",
        f"{server_host}:{server_port}",
    ]
    if data_dir:
        cmd.extend(["--data-dir", str(data_dir)])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PROJECT_ROOT)
    return jsonify({"started": True, "pid": proc.pid, "client_id": client_id})

@app.post("/api/metrics")
def api_metrics():
    payload = request.get_json(force=True)
    socketio.emit("metrics", payload)
    return ("", 204)

@app.get("/download/model")
def download_model():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(results_dir):
            for root, _, files in os.walk(results_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arc = os.path.relpath(full, PROJECT_ROOT)
                    zf.write(full, arc)
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="model_results.zip")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    socketio.run(app, host="0.0.0.0", port=port)
