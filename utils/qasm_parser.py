# utils/qasm_parser.py

from qiskit import QuantumCircuit
from qiskit.qasm import pi
import os

def load_qasm_file(filepath):
    """
    Load an OpenQASM file and return the gate list and qubit count.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"QASM file not found: {filepath}")
    
    try:
        qc = QuantumCircuit.from_qasm_file(filepath)
    except Exception as e:
        raise ValueError(f"Failed to parse QASM file {filepath}: {e}")

    gate_list = []
    for instr, qargs, _ in qc.data:
        gate_name = instr.name
        qubit_ids = [q.index for q in qargs]
        gate_list.append((gate_name, qubit_ids))

    num_qubits = qc.num_qubits
    return gate_list, num_qubits

if __name__ == "__main__":
    # Example usage
    path = "../origin_circuit/qubits_3/grover_3.qasm"
    gates, qubits = load_qasm_file(path)
    print(f"Qubits: {qubits}")
    print("First 5 gates:")
    print(gates[:5])
