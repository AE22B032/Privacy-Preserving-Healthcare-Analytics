import os
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
import poweranger.server.utils as utils
from poweranger.server.model import get_model
from typing import Dict, List, Tuple

RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    strategy = fl.server.strategy.FedAvg(
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
    strategy = fl.server.strategy.FedAvg(
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
