"""TensorFlow data loading for federated healthcare images.

Supports loading from a directory of images (per hospital silo) with
tf.keras.utils.image_dataset_from_directory, plus synthetic fallback.
Exposes two helpers:
 - load_data: returns (model, x_train, y_train, x_test, y_test) as NumPy
 - load_tf_datasets: returns (train_ds, val_ds, test_ds) as tf.data.Dataset
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf

# Use absolute package import to avoid path issues
from poweranger.server.model import get_model

__all__ = ["load_tf_datasets", "load_data"]


def _dataset_to_numpy(ds: tf.data.Dataset, max_batches: int | None = None):
    xs, ys = [], []
    for i, (xb, yb) in enumerate(ds):
        xs.append(xb.numpy())
        ys.append(yb.numpy())
        if max_batches is not None and (i + 1) >= max_batches:
            break
    if not xs:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_tf_datasets(client_id: str | int | None,
                     data_dir: str | None = None,
                     image_size: tuple[int, int] = (128, 128),
                     batch_size: int = 32):
    data_dir = data_dir or os.getenv("DATA_DIR")
    if data_dir and os.path.isdir(data_dir):
        seed = 42 + int(client_id or 0)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )

        def _norm(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            return x, y

        train_ds = train_ds.map(_norm).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(_norm).prefetch(tf.data.AUTOTUNE)
        test_ds = val_ds
    else:
        # Synthetic fallback
        h, w = image_size
        # Smaller synthetic dataset to lower memory use during simulation
        x_train = np.random.rand(64, h, w, 3).astype(np.float32)
        y_train = np.random.randint(0, 2, 64).astype(np.float32)
        x_test = np.random.rand(16, h, w, 3).astype(np.float32)
        y_test = np.random.randint(0, 2, 16).astype(np.float32)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        test_ds = val_ds

    return train_ds, val_ds, test_ds


def load_data(client_id: str | int | None,
              data_dir: str | None = None,
              image_size: tuple[int, int] = (128, 128),
              batch_size: int = 32):
    """Return (model, x_train, y_train, x_test, y_test) for NumPyClient code paths."""
    model = get_model(input_shape=(*image_size, 3))
    train_ds, val_ds, _ = load_tf_datasets(client_id, data_dir=data_dir, image_size=image_size, batch_size=batch_size)
    x_train, y_train = _dataset_to_numpy(train_ds, max_batches=None)
    x_test, y_test = _dataset_to_numpy(val_ds, max_batches=None)
    return model, x_train, y_train, x_test, y_test
