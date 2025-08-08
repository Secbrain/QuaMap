# utils/device_config.py

BACKEND_CONFIGS = {
    "ibmq_lima": {
        "num_qubits": 5,
        "coupling_map": [(0, 1), (1, 2), (1, 3), (3, 4)]
    },
    "ibmq_jakarta": {
        "num_qubits": 7,
        "coupling_map": [(1, 0), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)]
    },
    "ibmq_manila": {
        "num_qubits": 5,
        "coupling_map": [(0, 1), (1, 2), (2, 3), (3, 4)]
    },
    "ibmq_belem": {
        "num_qubits": 5,
        "coupling_map": [(0, 1), (1, 2), (1, 3), (3, 4)]
    },
    "ibmq_bogota": {
        "num_qubits": 5,
        "coupling_map": [(0, 1), (1, 2), (2, 3), (3, 4)]
    },
    "ibmq_santiago": {
        "num_qubits": 5,
        "coupling_map": [(0, 1), (1, 2), (2, 3), (3, 4)]
    },
    "ibmq_quito": {
        "num_qubits": 5,
        "coupling_map": [(0, 1), (1, 2), (1, 3), (3, 4)]
    },
    "ibmq_casablanca": {
        "num_qubits": 7,
        "coupling_map": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)]
    },
    "ibmq_perth": {
        "num_qubits": 7,
        "coupling_map": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)]
    },
    "ibmq_lagos": {
        "num_qubits": 7,
        "coupling_map": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)]
    }
}


def get_backend_config(backend_name):
    """
    Get coupling map and qubit count for a given backend
    """
    if backend_name not in BACKEND_CONFIGS:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_CONFIGS[backend_name]


def get_coupling_map(backend_name):
    return get_backend_config(backend_name)["coupling_map"]


def get_num_qubits(backend_name):
    return get_backend_config(backend_name)["num_qubits"]


if __name__ == "__main__":
    name = "ibmq_lima"
    config = get_backend_config(name)
    print(f"{name} config:")
    print("Qubits:", config["num_qubits"])
    print("Coupling map:", config["coupling_map"])
