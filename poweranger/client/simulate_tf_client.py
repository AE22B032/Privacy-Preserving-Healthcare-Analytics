"""Run a small Flower simulation using the TensorFlow Client (direct Client API).

Usage:
  python -m poweranger.poweranger.scripts.simulate_tf_client --rounds 2 --clients 3
"""

from __future__ import annotations

import argparse
import os
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from poweranger.client.client_fl_client import client_app


def server_fn(_context):
    rounds = int(os.getenv("ROUNDS", "2"))
    return ServerAppComponents(config=ServerConfig(num_rounds=rounds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "2")))
    parser.add_argument("--clients", type=int, default=int(os.getenv("CLIENTS", "3")))
    args = parser.parse_args()

    server = ServerApp(server_fn=server_fn)

    # Configure client resources; default None -> 2 CPU 0 GPU per client
    backend_config = {"client_resources": None}

    fl.simulation.run_simulation(
        server_app=server,
        client_app=client_app,
        num_supernodes=args.clients,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    main()
