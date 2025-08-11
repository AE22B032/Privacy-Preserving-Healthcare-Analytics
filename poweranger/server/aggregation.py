import numpy as np

def aggregate_fit(server_round, results, failures):
    weights = [fit_res.parameters for _, fit_res in results]

    avg_weights = []
    for weights_layer in zip(*weights):
        layer_stack = np.array([np.array(w) for w in weights_layer])
        avg_layer = np.mean(layer_stack, axis=0)
        avg_weights.append(avg_layer)

    print(f"✅ Aggregated round {server_round} weights")
    return avg_weights, {}
