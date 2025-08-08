# utils/layout_utils.py

import itertools

def enumerate_layouts(physical_qubits, logical_qubits):
    """
    Generate all possible layouts mapping logical_qubits onto physical_qubits.

    Args:
        physical_qubits (int): Total number of physical qubits in backend
        logical_qubits (int): Number of logical qubits in circuit

    Returns:
        List of layouts, each is a list of physical qubit indices (length = logical_qubits)
    """
    assert logical_qubits <= physical_qubits, "Too many logical qubits for this backend!"
    return list(itertools.permutations(range(physical_qubits), logical_qubits))


def encode_layout(layout):
    """
    Encode a layout [4,0,2] → "4_0_2" (used in filename or key)
    """
    return "_".join(str(q) for q in layout)


def decode_layout(layout_str):
    """
    Decode layout string "4_0_2" → [4, 0, 2]
    """
    return list(map(int, layout_str.strip().split("_")))


if __name__ == "__main__":
    # Example usage
    layouts = enumerate_layouts(physical_qubits=5, logical_qubits=3)
    print(f"Generated {len(layouts)} layouts.")
    print("Example:", layouts[:3])

    test_layout = [4, 0, 2]
    encoded = encode_layout(test_layout)
    print("Encoded:", encoded)
    print("Decoded:", decode_layout(encoded))
