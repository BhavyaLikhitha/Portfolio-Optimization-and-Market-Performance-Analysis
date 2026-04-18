"""Shared training loop for all PyTorch models."""

import time
import numpy as np
import torch
import torch.nn as nn
from config import DEVICE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, get_logger

logger = get_logger(__name__)


def train_model(model, train_loader, val_loader, task="regression",
                epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE):
    """Generic training loop for PyTorch models.

    Args:
        model: nn.Module
        train_loader, val_loader: DataLoaders
        task: 'regression' or 'classification'
        epochs, lr, patience: training hyperparameters

    Returns:
        model: trained model (best checkpoint)
        history: dict with 'train_loss' and 'val_loss' lists
        training_time: total training time in seconds
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    t0 = time.time()

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            if task == "classification":
                y_batch = y_batch.long().to(DEVICE)
            else:
                y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(X_batch)

            if task == "regression":
                output = output.squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                if task == "classification":
                    y_batch = y_batch.long().to(DEVICE)
                else:
                    y_batch = y_batch.to(DEVICE)

                output = model(X_batch)
                if task == "regression":
                    output = output.squeeze()
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} — train_loss: {avg_train:.6f}, val_loss: {avg_val:.6f}")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - t0

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    return model, history, training_time


def predict_model(model, data_loader, task="regression"):
    """Run inference and return predictions + true values."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)

            if task == "classification":
                probs = torch.softmax(output, dim=1)
                preds = probs.cpu().numpy()
            else:
                preds = output.squeeze().cpu().numpy()

            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_preds, all_targets
