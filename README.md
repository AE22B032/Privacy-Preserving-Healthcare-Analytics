# Federated Health (Poweranger)

Federated learning for healthcare images using Flower and TensorFlow.

- Server: `poweranger.server.server_app`
- Client: `poweranger.client.client_app`
- Standalone client runner: `python -m poweranger.client.standalone_client --client-id C1 --server-address 127.0.0.1:8081`

Run a simulation:
- `flwr run . --run-config "num-server-rounds=3 fraction-fit=0.3 batch-size=16 image-size=96"`
