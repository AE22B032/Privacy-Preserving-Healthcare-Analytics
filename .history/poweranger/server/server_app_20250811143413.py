import os
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
import poweranger.server.utils as utils
from poweranger.server.model import get_model

RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct ServerAppComponents: server, config, strategy, client_manager.

    Only config and strategy are customized here; defaults are used for server and client_manager.
    """
    # Read desired rounds and fraction from run_config when present (simulation)
    rc = context.run_config or {}
    num_rounds = int(rc.get("num-server-rounds", os.getenv("ROUNDS", "3")))
    fraction_fit = float(rc.get("fraction-fit", os.getenv("FRACTION_FIT", "0.3")))
    image_size = int(rc.get("image-size", os.getenv("IMAGE_SIZE", "96")))

    # Initialize global model parameters
    model = get_model(input_shape=(image_size, image_size, 3))
    initial_parameters = ndarrays_to_parameters(model.get_weights())

    # Define strategy with initial parameters
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
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
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
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
