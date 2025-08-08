# tasks/classification/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv, SumPooling

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class GINClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=3):
        super(GINClassifier, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mlp = MLP(hidden_dim if i > 0 else in_dim, hidden_dim, hidden_dim)
            conv = GINConv(mlp, learn_eps=True)
            self.layers.append(conv)

        self.readout = SumPooling()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, g, features):
        h = features
        for conv in self.layers:
            h = conv(g, h)
            h = F.relu(h)
        hg = self.readout(g, h)
        out = self.classifier(hg)
        return out
