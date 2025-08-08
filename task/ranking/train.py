# tasks/ranking/train.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

from dataset import LayoutPointwiseDataset, LayoutPairwiseDataset
from model import PointwiseRanker, PairwiseRanker

def train_pointwise(args, device):
    dataset = LayoutPointwiseDataset(
        metrics_dir=args.metrics_dir,
        backend_name=args.backend,
        qubit_count=args.qubit,
        target=args.target
    )
    if len(dataset) == 0:
        print("[ERROR] Empty dataset.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    input_dim = dataset[0][0].shape[0]
    model = PointwiseRanker(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                preds.extend(pred.cpu().tolist())
                gts.extend(y.cpu().tolist())

        mae = mean_absolute_error(gts, preds)
        r2 = r2_score(gts, preds)
        print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f} | MAE: {mae:.2f} | R²: {r2:.3f}")

def train_pairwise(args, device):
    dataset = LayoutPairwiseDataset(
        metrics_dir=args.metrics_dir,
        backend_name=args.backend,
        qubit_count=args.qubit,
        target=args.target
    )
    if len(dataset) == 0:
        print("[ERROR] Empty pairwise dataset.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    input_dim = dataset[0][0].shape[0]
    model = PairwiseRanker(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            prob = model(x1, x2)
            loss = loss_fn(prob, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                pred = model(x1, x2).cpu()
                all_preds.extend((pred > 0.5).int().tolist())
                all_labels.extend(y.tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f} | Val Acc: {acc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ibmq_lima")
    parser.add_argument("--qubit", type=int, default=4)
    parser.add_argument("--target", type=str, default="depth")
    parser.add_argument("--metrics-dir", type=str, default="../../metrics")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, choices=["pointwise", "pairwise"], default="pointwise")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "pointwise":
        train_pointwise(args, device)
    else:
        train_pairwise(args, device)

if __name__ == "__main__":
    main()
