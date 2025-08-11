"""TensorFlow model and training utilities for federated healthcare analytics.

This module defines a simple CNN suitable for medical image classification and
helper functions to integrate with Flower's federated learning (get/set
parameters, train, evaluate).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

try:  # Helpful error if TensorFlow isn't installed
	import tensorflow as tf
except Exception as _e:  # pragma: no cover
	raise ImportError(
		"TensorFlow is required but not installed. Install a PyPI build compatible with your Python version, e.g. 'pip install tensorflow>=2.17,<2.19' inside your venv.\n"
		f"Original import error: {_e}"
	)


def get_model(input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = 2) -> tf.keras.Model:
	"""Create a small CNN classifier.

	Args:
		input_shape: Image shape (H, W, C).
		num_classes: Number of classes. If 2, uses sigmoid + binary loss; otherwise softmax + categorical loss.

	Returns:
		A compiled tf.keras.Model.
	"""

	model = tf.keras.Sequential(
		[
			tf.keras.layers.InputLayer(input_shape=input_shape),
			tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(256, activation="relu"),
			tf.keras.layers.Dropout(0.3),
			tf.keras.layers.Dense(1 if num_classes == 2 else num_classes,
								  activation="sigmoid" if num_classes == 2 else "softmax"),
		]
	)

	if num_classes == 2:
		loss = "binary_crossentropy"
		metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
	else:
		loss = "sparse_categorical_crossentropy"
		metrics = ["accuracy"]

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=metrics)
	return model


def get_parameters(model: tf.keras.Model) -> List[tf.Tensor | tf.Variable]:
	"""Return model weights as a list (Flower expects NumPy-compatible arrays)."""
	return model.get_weights()


def set_parameters(model: tf.keras.Model, parameters: Iterable) -> None:
	"""Set model weights from a list of NumPy arrays or Tensors."""
	model.set_weights(list(parameters))


def train(model: tf.keras.Model,
		  train_ds: tf.data.Dataset,
		  epochs: int = 1,
		  steps_per_epoch: int | None = None,
		  validation_data: tf.data.Dataset | None = None,
		  validation_steps: int | None = None) -> tf.keras.callbacks.History:
	"""Train the model.

	The datasets should yield (images, labels) with images float32 in [0,1] (or already normalized).
	"""
	history = model.fit(
		train_ds,
		epochs=epochs,
		steps_per_epoch=steps_per_epoch,
		validation_data=validation_data,
		validation_steps=validation_steps,
		verbose=1,
	)
	return history


def evaluate(model: tf.keras.Model,
			 test_ds: tf.data.Dataset,
			 steps: int | None = None) -> Tuple[float, float]:
	"""Evaluate the model and return (loss, accuracy)."""
	results = model.evaluate(test_ds, steps=steps, verbose=0)
	# Keras returns a list [loss, acc, ...]; handle both list and dict forms
	if isinstance(results, (list, tuple)):
		loss = float(results[0])
		acc = float(results[1]) if len(results) > 1 else float("nan")
	elif isinstance(results, dict):
		loss = float(results.get("loss", float("nan")))
		acc = float(results.get("accuracy", float("nan")))
	else:
		loss, acc = float("nan"), float("nan")
	return loss, acc
