# tasks/classification/train.py

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

import dgl
from dgl.dataloading import GraphDataLoader

from dataset import CircuitClassificationDataset
from model import GINClassifier

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            feats = batched_graph.ndata["feat"].to(device)
            labels = labels.to(device)

            logits = model(batched_graph, feats)
            pred = logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ibmq_lima")
    parser.add_argument("--qubit", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metrics-dir", type=str, default="../../metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading dataset: backend={args.backend}, qubits={args.qubit}")
    dataset = CircuitClassificationDataset(
        metrics_dir=args.metrics_dir,
        backend_name=args.backend,
        qubit_count=args.qubit
    )

    if len(dataset) == 0:
        print("[ERROR] No data loaded.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    in_dim = dataset[0][0].ndata["feat"].shape[1]
    model = GINClassifier(in_dim=in_dim, hidden_dim=64, num_classes=15).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("[INFO] Starting training...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batched_graph, labels in train_loader:
            batched_graph = batched_graph.to(device)
            feats = batched_graph.ndata["feat"].to(device)
            labels = labels.to(device)

            logits = model(batched_graph, feats)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")

    print("[INFO] Final Evaluation on Test Set:")
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"[Test] Accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
