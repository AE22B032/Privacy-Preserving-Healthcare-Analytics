# client/local_training.py

def train_local(model, x_train, y_train, epochs=1, batch_size=32):
    """Train the model locally for one or more epochs."""
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
