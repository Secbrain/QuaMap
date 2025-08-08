# tasks/prediction/train.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from dataset import MetricPredictionDataset
from model import MLPRegressor


def evaluate(model, dataloader, device):
    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            preds.extend(y_pred.cpu().tolist())
            gts.extend(y.cpu().tolist())

    mae = mean_absolute_error(gts, preds)
    mse = mean_squared_error(gts, preds)
    r2 = r2_score(gts, preds)
    return mae, mse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ibmq_lima")
    parser.add_argument("--qubit", type=int, default=4)
    parser.add_argument("--target", type=str, default="cx_count")  # or depth, total_gate_count
    parser.add_argument("--metrics-dir", type=str, default="../../metrics")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading dataset: {args.backend} - {args.qubit} qubits - target={args.target}")
    dataset = MetricPredictionDataset(
        metrics_dir=args.metrics_dir,
        backend_name=args.backend,
        qubit_count=args.qubit,
        target=args.target
    )

    if len(dataset) == 0:
        print("[ERROR] No samples loaded.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    input_dim = dataset[0][0].shape[0]
    model = MLPRegressor(input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print("[INFO] Starting training...")

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

        val_mae, val_mse, val_r2 = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1:02d}] Train Loss: {total_loss:.4f} | Val MAE: {val_mae:.2f} | R²: {val_r2:.3f}")

    print("[INFO] Final Evaluation on Validation Set:")
    print(f"MAE: {val_mae:.2f} | MSE: {val_mse:.2f} | R²: {val_r2:.3f}")

if __name__ == "__main__":
    main()
