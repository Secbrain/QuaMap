import re
from collections import defaultdict

def analyze_qasm_metrics(qasm_file):
    """
    {
        "depth": int,
        "cx_count": int,
        "swap_count": int,
        "total_gate_count": int,
        "single_qubit_gate_count": int,
        "two_qubit_gate_count": int,
        "multi_qubit_gate_count": int
    }
    """
    skip_prefix = ("OPENQASM", "include", "qreg", "creg", "//")
    skip_ops = {"barrier"}
    
    qubit_time = defaultdict(int)

    cx_count = 0
    swap_count = 0
    total_gate_count = 0
    single_qubit_gate_count = 0
    two_qubit_gate_count = 0
    multi_qubit_gate_count = 0

    with open(qasm_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or any(line.startswith(x + " ") or line.startswith(x + "\t") or line == x for x in skip_prefix):
            continue

        if ';' in line:
            line = line[:line.index(';')]

        parts = line.split()
        if len(parts) < 2:
            continue

        op = parts[0]
        if op in skip_ops:
            continue

        args = parts[1].split(',')
        n = len(args)

        qubits = []
        for arg in args:
            idx = re.findall(r'\[(\d+)\]', arg)
            if idx:
                qubits.append(f"{arg.split('[')[0]}{idx[0]}")
            else:
                qubits.append(arg)

        start_t = max(qubit_time[q] for q in qubits)
        for q in qubits:
            qubit_time[q] = start_t + 1

        total_gate_count += 1

        if op == "cx":
            cx_count += 1
        elif op == "swap":
            swap_count += 1

        if n == 1:
            single_qubit_gate_count += 1
        elif n == 2:
            two_qubit_gate_count += 1
        else:
            multi_qubit_gate_count += 1

    depth = max(qubit_time.values()) if qubit_time else 0

    return {
        "depth": depth,
        "cx_count": cx_count,
        "swap_count": swap_count,
        "total_gate_count": total_gate_count,
        "single_qubit_gate_count": single_qubit_gate_count,
        "two_qubit_gate_count": two_qubit_gate_count,
        "multi_qubit_gate_count": multi_qubit_gate_count
    }
