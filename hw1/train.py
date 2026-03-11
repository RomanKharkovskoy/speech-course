import logging
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for waveforms, targets in loader:
        waveforms, targets = waveforms.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = criterion(model(waveforms), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for waveforms, targets in loader:
        waveforms, targets = waveforms.to(device), targets.to(device)
        predictions = (model(waveforms) > 0.0).float()
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return correct / total if total > 0 else 0.0


def train_model(model, train_loader, val_loader, num_epochs=15, lr=1e-3, device="cpu"):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_accuracy": [], "epoch_time": []}

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["epoch_time"].append(elapsed)

        logger.info(
            f"epoch {epoch + 1:2d}/{num_epochs}, "
            f"loss: {train_loss:.4f}, "
            f"val acc: {val_acc:.4f}, "
            f"time: {elapsed:.1f}"
        )

    return history
