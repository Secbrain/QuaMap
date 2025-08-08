# tasks/ranking/dataset.py

import torch
from torch.utils.data import Dataset
from utils.metrics_loader import load_all_metrics
from utils.qasm_parser import load_qasm_file
from tasks.prediction.dataset import encode_layout, extract_gate_stats

class LayoutPointwiseDataset(Dataset):
    """
    Each sample is (QASM + layout) → target (e.g., depth or cx_count)
    Used for regression/ranking score prediction
    """
    def __init__(self, metrics_dir, backend_name, qubit_count, target="depth"):
        self.records = load_all_metrics(metrics_dir, backend_filter=backend_name, qubit_filter=qubit_count)
        self.samples = []
        self.max_qubits = 7
        self.target = target

        for rec in self.records:
            try:
                qasm_path = rec["source_qasm"]
                layout = rec["layout"]
                gates, _ = load_qasm_file(qasm_path)

                gate_feat = extract_gate_stats(gates)
                layout_feat = encode_layout(layout, max_qubits=self.max_qubits)
                x = gate_feat + layout_feat
                y = float(rec[target])

                self.samples.append((x, y))
            except Exception as e:
                continue  # skip invalid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LayoutPairwiseDataset(Dataset):
    """
    Each sample is (x1, x2, label):
      x1 = (qasm + layout A), x2 = (qasm + layout B)
      label = 1 if A better than B (e.g., lower depth)
    """
    def __init__(self, metrics_dir, backend_name, qubit_count, target="depth"):
        self.records = load_all_metrics(metrics_dir, backend_filter=backend_name, qubit_filter=qubit_count)
        self.pairs = []
        self.max_qubits = 7
        self.target = target

        # group by circuit name
        circuit_groups = {}
        for rec in self.records:
            name = rec["source_qasm"]
            circuit_groups.setdefault(name, []).append(rec)

        for name, recs in circuit_groups.items():
            if len(recs) < 2:
                continue
            try:
                gates, _ = load_qasm_file(name)
                gate_feat = extract_gate_stats(gates)

                for i in range(len(recs)):
                    for j in range(i+1, len(recs)):
                        a = recs[i]
                        b = recs[j]
                        x1 = gate_feat + encode_layout(a["layout"], self.max_qubits)
                        x2 = gate_feat + encode_layout(b["layout"], self.max_qubits)

                        y1 = float(a[target])
                        y2 = float(b[target])
                        label = 1 if y1 < y2 else 0  # lower is better
                        self.pairs.append((x1, x2, label))
            except:
                continue

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2, y = self.pairs[idx]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
