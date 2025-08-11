"""Flower Client implementation using the low-level Client API (TensorFlow).

This demonstrates migrating from NumPyClient to Client for federated learning
in the healthcare imaging setting. Datasets below are synthetic for demo; wire
your hospital-specific tf.data pipelines as needed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import flwr as fl
from flwr.client import Client
from flwr.common import (
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import tensorflow as tf

from poweranger.server.model import (
    get_model,
    get_parameters as keras_get_parameters,
    set_parameters as keras_set_parameters,
    train as keras_train,
    evaluate as keras_evaluate,
)
from poweranger.client.data_loader import load_tf_datasets


def _build_synthetic_ds(n: int = 128, image_shape: Tuple[int, int, int] = (128, 128, 3), num_classes: int = 2, batch_size: int = 32) -> tf.data.Dataset:
    x = np.random.rand(n, *image_shape).astype("float32")
    y = np.random.randint(0, num_classes, size=(n,)).astype("float32" if num_classes == 2 else "int32")
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


class TFFlowerClient(Client):
    def __init__(self, partition_id: int, *, image_shape=(128, 128, 3), num_classes: int = 2, epochs: int = 1, batch_size: int = 32, data_dir: str | None = None):
        self.partition_id = partition_id
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.model = get_model(input_shape=image_shape, num_classes=num_classes)
        # Try to load real hospital tf.data pipelines, fallback to synthetic
        try:
            self.train_ds, self.val_ds, _ = load_tf_datasets(
                client_id=partition_id,
                data_dir=data_dir,
                image_size=image_shape[:2],
                batch_size=batch_size,
            )
        except Exception:
            self.train_ds = _build_synthetic_ds(n=128, image_shape=image_shape, num_classes=num_classes, batch_size=batch_size)
            self.val_ds = _build_synthetic_ds(n=64, image_shape=image_shape, num_classes=num_classes, batch_size=batch_size)

    # --- Client API methods ---
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        weights = keras_get_parameters(self.model)
        return GetParametersRes(parameters=ndarrays_to_parameters(weights), status=fl.common.Status(code=fl.common.Code.OK, message=""))

    def fit(self, ins: FitIns) -> FitRes:
        # Set incoming model weights
        ndarrays = parameters_to_ndarrays(ins.parameters)
        keras_set_parameters(self.model, ndarrays)

        # Optional: read server-provided config
        cfg: Dict = ins.config or {}
        epochs = int(cfg.get("local_epochs", self.epochs))
        batch_size = int(cfg.get("batch_size", self.batch_size))

        # Re-batch if server provided a different batch size
        train_ds = self.train_ds if batch_size == self.batch_size else self.train_ds.unbatch().batch(batch_size)
        val_ds = self.val_ds if batch_size == self.batch_size else self.val_ds.unbatch().batch(batch_size)

        keras_train(self.model, train_ds, epochs=epochs, validation_data=val_ds)

        # Return updated weights and number of examples used for training
        new_params = ndarrays_to_parameters(keras_get_parameters(self.model))
        num_examples = sum(1 for _ in train_ds.unbatch())
        return FitRes(parameters=new_params, num_examples=num_examples, status=fl.common.Status(code=fl.common.Code.OK, message=""))

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        keras_set_parameters(self.model, ndarrays)
        loss, accuracy = keras_evaluate(self.model, self.val_ds)
        num_examples = sum(1 for _ in self.val_ds.unbatch())
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(loss=float(loss), num_examples=num_examples, metrics=metrics, status=fl.common.Status(code=fl.common.Code.OK, message=""))


def client_fn(context: Context) -> Client:
    partition_id = int(context.node_config.get("partition-id", 0))
    num_classes = int(context.node_config.get("num_classes", 2))
    data_dir = context.node_config.get("data_dir")
    return TFFlowerClient(partition_id, num_classes=num_classes, data_dir=data_dir)


# Expose a ClientApp for simulations
client_app = fl.client.ClientApp(client_fn=client_fn)
