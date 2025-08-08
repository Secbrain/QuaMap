# tasks/ranking/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointwiseRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PointwiseRanker, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


class PairwiseRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, shared=True):
        super(PairwiseRanker, self).__init__()
        self.shared = shared

        if shared:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            self.encoder1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.encoder2 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        if self.shared:
            h1 = self.encoder(x1)
            h2 = self.encoder(x2)
        else:
            h1 = self.encoder1(x1)
            h2 = self.encoder2(x2)

        score_diff = h1 - h2
        logits = self.classifier(score_diff).squeeze(-1)  # (batch,)
        prob = torch.sigmoid(logits)
        return prob
