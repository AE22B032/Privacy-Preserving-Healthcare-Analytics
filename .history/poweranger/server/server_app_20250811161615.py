import os
import csv
from typing import Dict, List, Tuple
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
import poweranger.server.utils as utils
from poweranger.server.model import get_model
import numpy as np

METRICS_DIR = "results/metrics"
MODELS_DIR = "results/models"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # metrics: list of (num_examples, {metric_name: value})
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    agg: Dict[str, float] = {}
    keys = set().union(*(m.keys() for _, m in metrics))
    for k in keys:
        agg[k] = sum(m.get(k, 0.0) * n for n, m in metrics) / max(total, 1)
    return agg


class SaveableFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, image_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_size = image_size

    @staticmethod
    def _append_csv(path: str, row: Dict[str, float | int]) -> None:
        # Ensure header
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        # Save per-client fit metrics
        try:
            if results:
                path = os.path.join(METRICS_DIR, "fit_client_metrics.csv")
                for client, fit_res in results:
                    row = {
                        "round": int(server_round),
                        "client_id": getattr(client, "cid", str(client)),
                        "num_examples": int(getattr(fit_res, "num_examples", 0)),
                    }
                    metrics = getattr(fit_res, "metrics", None) or {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            row[k] = float(v)
                    self._append_csv(path, row)
        except Exception as e:
            print(f"Warning: failed to save per-client fit metrics for round {server_round}: {e}")
        # Save metrics
        if aggregated_metrics:
            row = {"round": int(server_round), **{k: float(v) for k, v in aggregated_metrics.items()}}
            self._append_csv(os.path.join(METRICS_DIR, "fit_metrics.csv"), row)
        # Save global model parameters as compressed numpy
        if aggregated_parameters is not None:
            try:
                nds = parameters_to_ndarrays(aggregated_parameters)
                np.savez_compressed(os.path.join(MODELS_DIR, f"round_{int(server_round):03d}.npz"), *nds)
            except Exception as e:
                print(f"Warning: failed to save model params for round {server_round}: {e}")
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        # Save per-client eval metrics
        try:
            if results:
                path = os.path.join(METRICS_DIR, "eval_client_metrics.csv")
                for client, eval_res in results:
                    row = {
                        "round": int(server_round),
                        "client_id": getattr(client, "cid", str(client)),
                        "num_examples": int(getattr(eval_res, "num_examples", 0)),
                    }
                    loss = getattr(eval_res, "loss", None)
                    if isinstance(loss, (int, float)):
                        row["loss"] = float(loss)
                    metrics = getattr(eval_res, "metrics", None) or {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            row[k] = float(v)
                    self._append_csv(path, row)
        except Exception as e:
            print(f"Warning: failed to save per-client eval metrics for round {server_round}: {e}")
        # Save eval metrics (include loss)
        row = {"round": int(server_round)}
        if aggregated_loss is not None:
            row["loss"] = float(aggregated_loss)
        if aggregated_metrics:
            row.update({k: float(v) for k, v in aggregated_metrics.items()})
        if len(row) > 1:
            self._append_csv(os.path.join(METRICS_DIR, "eval_metrics.csv"), row)
        return aggregated_loss, aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Construct ServerAppComponents: server, config, strategy, client_manager.

    Only config and strategy are customized here; defaults are used for server and client_manager.
    """
    # Read desired rounds and fraction from run_config when present (simulation)
    rc = context.run_config or {}
    num_rounds = int(rc.get("num-server-rounds", os.getenv("ROUNDS", "3")))
    fraction_fit = float(rc.get("fraction-fit", os.getenv("FRACTION_FIT", "0.3")))
    fraction_evaluate = float(rc.get("fraction-evaluate", os.getenv("FRACTION_EVAL", str(fraction_fit))))
    min_fit_clients = int(rc.get("min-fit-clients", os.getenv("MIN_FIT_CLIENTS", "1")))
    min_evaluate_clients = int(rc.get("min-evaluate-clients", os.getenv("MIN_EVAL_CLIENTS", "1")))
    min_available_clients = int(rc.get("min-available-clients", os.getenv("MIN_AVAILABLE_CLIENTS", "1")))
    image_size = int(rc.get("image-size", os.getenv("IMAGE_SIZE", "96")))

    # Initialize global model parameters
    model = get_model(input_shape=(image_size, image_size, 3))
    initial_parameters = ndarrays_to_parameters(model.get_weights())

    # Define strategy with initial parameters
    strategy = SaveableFedAvg(
        image_size=image_size,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    return ServerAppComponents(config=ServerConfig(num_rounds=num_rounds), strategy=strategy)


# Expose a ServerApp for simulations (fl.simulation.run_simulation)
app = ServerApp(server_fn=server_fn)
# Backwards-compatible alias
server = app


if __name__ == "__main__":
    utils.setup_logging()
    # Build strategy/config directly (Context is provided by Flower only in simulation)
    num_rounds = int(os.getenv("ROUNDS", "3"))
    # Initialize global parameters
    image_size = int(os.getenv("IMAGE_SIZE", "96"))
    model = get_model(input_shape=(image_size, image_size, 3))
    initial_parameters = ndarrays_to_parameters(model.get_weights())
    strategy = SaveableFedAvg(
        image_size=image_size,
        fraction_fit=float(os.getenv("FRACTION_FIT", "0.3")),
        fraction_evaluate=float(os.getenv("FRACTION_EVAL", os.getenv("FRACTION_FIT", "0.3"))),
        min_fit_clients=int(os.getenv("MIN_FIT_CLIENTS", "1")),
        min_evaluate_clients=int(os.getenv("MIN_EVAL_CLIENTS", "1")),
        min_available_clients=int(os.getenv("MIN_AVAILABLE_CLIENTS", "1")),
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    host = os.getenv("FLOWER_BIND", os.getenv("FLOWER_HOST", "0.0.0.0"))
    port = int(os.getenv("FLOWER_PORT", "8081"))
    addr = f"{host}:{port}"
    print(f"🚀 Starting Federated Learning Server on {addr}...")
    fl.server.start_server(
        server_address=addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )
