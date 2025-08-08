# tasks/classification/dataset.py

import os
import torch
import dgl
import networkx as nx
from torch.utils.data import Dataset
from utils.qasm_parser import load_qasm_file
from utils.graph_builder import build_gate_interaction_graph
from utils.metrics_loader import load_all_metrics

ALGO_CATEGORIES = [
    "BV", "Clifford", "Grover", "Ising", "QFT", "QKNN", "QNN", "QPE", "QSVM",
    "QuGAN", "RB", "Shor", "Simon", "VQC", "XEB"
]
CATEGORY_TO_IDX = {name: i for i, name in enumerate(ALGO_CATEGORIES)}

def nx_to_dgl(nx_graph):
    """
    Convert a NetworkX DiGraph to DGLGraph with gate-type one-hot node features
    """
    g = dgl.from_networkx(nx_graph, node_attrs=["gate", "qubits"])
    
    gate_types = list(set(nx.get_node_attributes(nx_graph, "gate").values()))
    gate2idx = {g: i for i, g in enumerate(sorted(gate_types))}

    gate_feat = []
    for _, attr in nx_graph.nodes(data=True):
        one_hot = [0] * len(gate2idx)
        one_hot[gate2idx[attr["gate"]]] = 1
        gate_feat.append(one_hot)

    g.ndata['feat'] = torch.tensor(gate_feat, dtype=torch.float32)
    return g


class CircuitClassificationDataset(Dataset):
    def __init__(self, metrics_dir, backend_name, qubit_count):
        self.records = load_all_metrics(metrics_dir, backend_filter=backend_name, qubit_filter=qubit_count)
        self.graphs = []
        self.labels = []

        for rec in self.records:
            qasm_path = rec["mapped_qasm"]
            algo_name = self._parse_algorithm_name(qasm_path)
            if algo_name not in CATEGORY_TO_IDX:
                continue

            try:
                gates, _ = load_qasm_file(qasm_path)
                nx_g = build_gate_interaction_graph(gates)
                dgl_g = nx_to_dgl(nx_g)
                label = CATEGORY_TO_IDX[algo_name]

                self.graphs.append(dgl_g)
                self.labels.append(label)
            except Exception as e:
                print(f"Error loading {qasm_path}: {e}")

    def _parse_algorithm_name(self, qasm_path):
        filename = os.path.basename(qasm_path)
        return filename.split("_")[0]  # e.g., "grover_0_1_2.qasm" → "grover"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
