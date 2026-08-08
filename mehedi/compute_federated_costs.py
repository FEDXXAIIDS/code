import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

CONFIG_CLIENTS = 5
CONFIG_ROUNDS = 7
BYTES_PER_PARAM = 4
KB = 1024
MB = KB * KB


def create_mlp_model(input_dim: int = 46, num_classes: int = 6) -> keras.Model:
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def create_cnn_model(input_dim: int = 46, num_classes: int = 6) -> keras.Model:
    model = keras.Sequential([
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def create_cnn_mlp_model(input_dim: int = 46, num_classes: int = 6) -> keras.Model:
    model = keras.Sequential([
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def model_size_bytes(model: keras.Model, bytes_per_param: int = BYTES_PER_PARAM) -> int:
    return int(sum(np.prod(w.shape) for w in model.get_weights()) * bytes_per_param)


def communication_cost_bytes(model: keras.Model, num_clients: int = CONFIG_CLIENTS, rounds: int = CONFIG_ROUNDS) -> int:
    """Estimate total communication cost in bytes for federated averaging.

    Assumes each round includes:
    - server broadcast of the global model to each client
    - each client upload of updated weights back to server

    Total bytes = 2 * num_clients * rounds * model_size
    """
    return 2 * num_clients * rounds * model_size_bytes(model)


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= MB:
        return f"{num_bytes / MB:.3f} MB"
    if num_bytes >= KB:
        return f"{num_bytes / KB:.1f} KB"
    return f"{num_bytes} bytes"


def total_communication_from_size(serialized_size_bytes: int, num_clients: int, rounds: int) -> int:
    return 2 * rounds * num_clients * serialized_size_bytes


def print_model_metrics(name: str, model: keras.Model, num_clients: int, rounds: int) -> None:
    params = model.count_params()
    size_bytes = model_size_bytes(model)
    comm_bytes = communication_cost_bytes(model, num_clients=num_clients, rounds=rounds)
    print(f"{name} architecture")
    print(f"  Parameters: {params:,}")
    print(f"  Model size: {format_bytes(size_bytes)}")
    print(f"  Communication per round: {format_bytes(size_bytes * 2 * num_clients)}")
    print(f"  Total communication over {rounds} rounds and {num_clients} clients: {format_bytes(comm_bytes)}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate federated communication costs for IDS models.')
    parser.add_argument('--clients', type=int, default=CONFIG_CLIENTS, help='Number of clients (K).')
    parser.add_argument('--rounds', type=int, default=CONFIG_ROUNDS, help='Number of federated rounds (R).')
    parser.add_argument('--example-size-kb', type=int, default=720, help='Serialized example model size in KB for CNN+LSTM estimate.')
    args = parser.parse_args()

    num_clients = args.clients
    rounds = args.rounds

    print("Federated communication cost estimates")
    print("--------------------------------------")
    print(f"Configured clients: {num_clients}")
    print(f"Configured rounds: {rounds}\n")

    mlp = create_mlp_model()
    cnn = create_cnn_model()
    cnn_mlp = create_cnn_mlp_model()

    print_model_metrics("Deep MLP", mlp, num_clients=num_clients, rounds=rounds)
    print_model_metrics("Pure 1D-CNN", cnn, num_clients=num_clients, rounds=rounds)
    print_model_metrics("CNN + MLP", cnn_mlp, num_clients=num_clients, rounds=rounds)

    serialized_model_size_kb = args.example_size_kb
    serialized_size_bytes = serialized_model_size_kb * KB
    example_comm = total_communication_from_size(serialized_size_bytes, num_clients, rounds)

    print("Example CNN+LSTM communication estimate")
    print(f"  Serialized model size: {serialized_model_size_kb} KB")
    print(f"  Clients (K): {num_clients}")
    print(f"  Rounds (R): {rounds}")
    print(f"  Total communication: {format_bytes(example_comm)}")
    print()

    print("Training time measurement")
    print("  In your federated training code, use time.time() around the training call:")
    print("    start = time.time()")
    print("    model, history = comparer.train_federated_model(...)")
    print("    elapsed = time.time() - start")
    print("  That elapsed value is the total training time in seconds.")
