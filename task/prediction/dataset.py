# tasks/prediction/dataset.py

import os
import torch
from torch.utils.data import Dataset
from utils.metrics_loader import load_all_metrics
from utils.qasm_parser import load_qasm_file

GATE_TYPES = ["h", "x", "y", "z", "cx", "cz", "swap", "t", "tdg", "rx", "ry", "rz"]

def extract_gate_stats(gate_list):
    counter = {g: 0 for g in GATE_TYPES}
    for gname, _ in gate_list:
        if gname in counter:
            counter[gname] += 1
    return [counter[g] for g in GATE_TYPES]


def encode_layout(layout, max_qubits=7):
    """
    Encode layout like [4,0,2,3] into one-hot flat vector
    """
    vec = [0] * (max_qubits * max_qubits)
    for logical, physical in enumerate(layout):
        vec[logical * max_qubits + physical] = 1
    return vec


class MetricPredictionDataset(Dataset):
    def __init__(self, metrics_dir, backend_name, qubit_count, target="depth"):
        self.records = load_all_metrics(metrics_dir, backend_filter=backend_name, qubit_filter=qubit_count)
        self.samples = []
        self.target = target
        self.max_qubits = 7  # for fixed one-hot size

        for rec in self.records:
            try:
                qasm_path = rec["source_qasm"]
                layout = rec["layout"]
                gates, _ = load_qasm_file(qasm_path)

                gate_feat = extract_gate_stats(gates)
                layout_feat = encode_layout(layout, max_qubits=self.max_qubits)
                input_feat = gate_feat + layout_feat
                label = float(rec[target])

                self.samples.append((input_feat, label))
            except Exception as e:
                print(f"Skipping record due to error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
