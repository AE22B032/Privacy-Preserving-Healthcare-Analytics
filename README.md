# Federated Health

Privacy-preserving federated learning for medical images using Flower and TensorFlow. Includes a simple web UI, low-memory defaults, and automatic saving of aggregated metrics and global model snapshots per round.

## What's inside

- Flower ServerApp/ClientApp (TensorFlow Keras CNN)
- Low-memory simulation defaults (Ray backend safe)
- Aggregated and per-client metrics saved to CSV each round
- Global model parameters saved per round (.npz)
- Standalone client runner and simple Flask web UI
- Configuration via `pyproject.toml` and environment variables

## Repository layout

```
pyproject.toml
requirements.txt
quick_start.sh
results/
	metrics/
	models/
poweranger/
	server/
		server_app.py      # Flower ServerApp + SaveableFedAvg strategy (saves metrics/models)
		model.py           # Keras CNN and helpers
		utils.py
	client/
		client_app.py      # Flower ClientApp (NumPyClient + TF)
		data_loader.py     # Loads tf.data from directory or synthetic fallback
		standalone_client.py
	web_ui/
		web_app_improved.py  # Simple dashboard to start server/clients
		templates/
		static/
```

## Quickstart

Prereqs:
- Python 3.11+ (TensorFlow >= 2.17 supports Python 3.12)
- Linux/macOS recommended
- Optional: GPU with proper TF build

Create a virtual environment and install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Option A: Run a local simulation

The simulation uses your `pyproject.toml` ServerApp/ClientApp components.

```bash
flwr run . --run-config "num-server-rounds=3 fraction-fit=0.3 batch-size=16 image-size=96"
```

- Defaults minimize memory: samples 1 client per phase, small image size/batch.
- Adjust via flags above or `pyproject.toml`.

### Option B: Run server and one client manually

Start the server (binds to 0.0.0.0:8081 by default):

```bash
python -m poweranger.server.server_app
```

Start a client in another terminal:

```bash
python -m poweranger.client.standalone_client \
	--client-id C1 \
	--server-address 127.0.0.1:8081
```

Optional: point a client at a local dataset directory (see Data section):

```bash
DATA_DIR=/path/to/images python -m poweranger.client.standalone_client \
	--client-id C1 --server-address 127.0.0.1:8081
```

### Option C: Web UI

Launch the dashboard:

```bash
python -m poweranger.web_ui.web_app_improved
```

Then open http://127.0.0.1:5000 to start a server and spawn clients.

## Data

The client can load images from a directory structure compatible with `tf.keras.preprocessing.image_dataset_from_directory`:

```
DATA_DIR/
	class_a/
		img001.png
		...
	class_b/
		imgXYZ.png
		...
```

- Set `DATA_DIR` env var per client to use real images.
- If `DATA_DIR` is not set or invalid, the client falls back to a tiny synthetic dataset to keep memory usage low.
- Configure `image-size` and `batch-size` via run config or `pyproject.toml`.

## Configuration

Key configuration lives in `pyproject.toml`:

- Flower components:
	- `serverapp = "poweranger.server.server_app:app"`
	- `clientapp = "poweranger.client.client_app:app"`
- Defaults (overridable at runtime):
	- `num-server-rounds`, `fraction-fit`, `fraction-evaluate`
	- `min-fit-clients`, `min-evaluate-clients`, `min-available-clients`
	- `local-epochs`, `batch-size`, `image-size`, `verbose`
- Local simulation:
	- `options.num-supernodes = 1` (prevents Ray OOM on small machines)

Environment variables (examples):

- `FLOWER_HOST` / `FLOWER_PORT` (server bind address)
- `FRACTION_FIT`, `FRACTION_EVAL`, `MIN_FIT_CLIENTS`, `MIN_EVAL_CLIENTS`, `MIN_AVAILABLE_CLIENTS`
- `IMAGE_SIZE`, `ROUNDS`
- `DATA_DIR` (for clients using local images)

## Results and checkpoints

Saved automatically each round by the server strategy (`SaveableFedAvg`):

- Aggregated metrics (weighted by client num_examples):
	- `results/metrics/fit_metrics.csv` (round, metrics)
	- `results/metrics/eval_metrics.csv` (round, loss, metrics)
- Per-client metrics:
	- `results/metrics/fit_client_metrics.csv` (round, client_id, num_examples, metrics)
	- `results/metrics/eval_client_metrics.csv` (round, client_id, num_examples, loss, metrics)
- Global model parameters:
	- `results/models/round_XXX.npz` (NumPy arrays of weights)

Tip: you can reconstruct a Keras model and load weights from a `.npz` snapshot for analysis.

## Architecture

- Model: `poweranger/server/model.py` defines a compact CNN and FL helpers.
- Server: `poweranger/server/server_app.py` builds the Flower ServerApp and a `SaveableFedAvg` strategy that aggregates metrics and writes snapshots.
- Client: `poweranger/client/client_app.py` implements a TensorFlow `NumPyClient` with low thread counts to reduce memory.
- Web UI: `poweranger/web_ui/web_app_improved.py` provides a minimal dashboard to start a server and clients.

## Troubleshooting

- Ray OOM or processes killed:
	- Keep `options.num-supernodes=1` in `pyproject.toml`.
	- Lower `batch-size`, `image-size`, and set TF threads to 1 (already applied).
- Metric aggregation warnings:
	- Resolved by passing `fit_metrics_aggregation_fn` and `evaluate_metrics_aggregation_fn` (implemented).
- TensorFlow retracing messages:
	- Benign; reduce by keeping shapes/static batch sizes consistent across rounds.
- VS Code/Pylance noise from site-packages:
	- Exclude venv and site-packages in workspace settings to focus diagnostics on your code.

## Optional: Google Drive integration

`drive_connect.py` shows basic OAuth for Google Drive using the official API:

1) Create OAuth client credentials (Desktop) in Google Cloud Console and download `credentials.json` into the project root.
2) First run will open a browser for consent and create `token.json`.

```bash
python drive_connect.py
```

Then extend this script to upload `results/` artifacts as needed.

## Contributing

Issues and PRs are welcome. Please include environment details and reproduction steps.

## License

Specify a license for this project (e.g., MIT). If unsure, add a `LICENSE` file before publishing.
