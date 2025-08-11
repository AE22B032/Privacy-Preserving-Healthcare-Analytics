"""Poweranger: Flower ClientApp using TensorFlow and medical image data."""

from __future__ import annotations

import os
import tensorflow as tf
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from poweranger.client.data_loader import load_data
from poweranger.server.model import get_model


class FlowerClient(NumPyClient):
    def __init__(self, model: tf.keras.Model, data, epochs: int, batch_size: int, verbose: int = 0):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Allow server-provided overrides
        epochs = int(config.get("local_epochs", self.epochs)) if isinstance(config, dict) else self.epochs
        batch_size = int(config.get("batch_size", self.batch_size)) if isinstance(config, dict) else self.batch_size
        verbose = int(config.get("verbose", self.verbose)) if isinstance(config, dict) else self.verbose
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # Use return_dict to handle multiple metrics (e.g., loss, accuracy, auc)
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0, return_dict=True)
        loss = float(results.get("loss", float("nan")))
        # Prefer common accuracy keys
        acc = (
            results.get("accuracy")
            or results.get("acc")
            or results.get("binary_accuracy")
            or results.get("sparse_categorical_accuracy")
        )
        metrics = {"accuracy": float(acc) if acc is not None else float("nan")}
        return loss, len(self.x_test), metrics


def client_fn(context: Context):
    # Limit TF intra/inter threads to reduce per-process memory usage during simulation
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    # Hyperparameters from pyproject or defaults
    epochs = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 32))
    verbose = int(context.run_config.get("verbose", 0))
    image_size = int(context.run_config.get("image-size", 96))

    # Optional per-client data directory (simulate hospital silo)
    data_dir = context.node_config.get("data_dir") if isinstance(context.node_config, dict) else os.getenv("DATA_DIR")

    # Load model and data
    partition_id = context.node_config.get("partition-id", 0)
    model = get_model()
    model, x_train, y_train, x_test, y_test = load_data(
        partition_id,
        data_dir=data_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    return FlowerClient(model, (x_train, y_train, x_test, y_test), epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
