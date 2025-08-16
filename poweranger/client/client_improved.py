"""
Client entrypoint that works both as a module (python -m client.client_improved)
and as a script (python client/client_improved.py).
"""

import argparse
import os
import requests
import flwr as fl

# Ensure project root is on sys.path when run as a script from the repo root
# so imports like `client.*` and `server.*` resolve.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from client.data_loader import load_data
from client.local_training import train_local

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, *, data_dir: str | None = None, image_size=(128, 128), epochs: int = 1, batch_size: int = 32):
        self.client_id = client_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.model, self.x_train, self.y_train, self.x_test, self.y_test = load_data(
            client_id, data_dir=data_dir, image_size=image_size, batch_size=batch_size
        )
        self.web_ui = os.getenv("WEB_UI_URL", "http://127.0.0.1:8080")

    def _emit(self, payload: dict):
        try:
            requests.post(f"{self.web_ui}/api/metrics", json=payload, timeout=1.5)
        except Exception:
            pass

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Allow server-side config to override local epochs/batch_size if provided
        epochs = int(config.get("local_epochs", self.epochs)) if isinstance(config, dict) else self.epochs
        batch_size = int(config.get("batch_size", self.batch_size)) if isinstance(config, dict) else self.batch_size
        train_local(self.model, self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        self._emit({"event": "fit", "client_id": self.client_id, "samples": len(self.x_train)})
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self._emit({"event": "evaluate", "client_id": self.client_id, "loss": float(loss), "accuracy": float(acc)})
        return float(loss), len(self.x_test), {"accuracy": float(acc)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--server", help="Flower server address host:port; overrides FLOWER_SERVER")
    parser.add_argument("--data-dir", help="Optional path to local image dataset root")
    parser.add_argument("--image-size", default="128,128", help="Image size as H,W (default 128,128)")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("LOCAL_EPOCHS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    args = parser.parse_args()

    try:
        h, w = [int(x) for x in args.image_size.split(",")]
        image_size = (h, w)
    except Exception:
        image_size = (128, 128)

    server_addr = args.server or os.getenv("FLOWER_SERVER", "localhost:8081")
    client = HospitalClient(
        args.client_id,
        data_dir=args.data_dir,
        image_size=image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    fl.client.start_numpy_client(server_address=server_addr, client=client)
