# tasks/prediction/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_layers=2):
        super(MLPRegressor, self).__init__()
        layers = []

        dim_in = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU())
            dim_in = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # Regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
