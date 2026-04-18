import torch.nn as nn

from config import DROPOUT, EPOCHS, LEARNING_RATE, MLFLOW_EXPERIMENTS
from evaluation.metrics import classification_metrics, regression_metrics
from evaluation.visualization import plot_training_curves
from models.deep_learning.trainer import predict_model, train_model
from tracking.mlflow_tracking import log_pytorch_model


class CNN1DModel(nn.Module):
    """1D Convolutional Neural Network for regression or classification.

    Input: (batch, seq_len, num_features)
    Permuted to: (batch, num_features, seq_len) for Conv1d (channels-first)
    """

    def __init__(self, input_size, output_size=1, dropout=DROPOUT):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # (batch, 64)
        return self.fc(x)


def train_cnn(train_loader, val_loader, input_size, task="regression", epochs=EPOCHS, lr=LEARNING_RATE):
    output_size = 1 if task == "regression" else 3
    model = CNN1DModel(input_size=input_size, output_size=output_size)
    model, history, training_time = train_model(model, train_loader, val_loader, task=task, epochs=epochs, lr=lr)
    curve_path = plot_training_curves(history, f"cnn_{task}")
    metrics = {
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "training_time": training_time,
    }
    log_pytorch_model(
        model,
        f"cnn_{task}",
        {"task": task, "input_size": input_size, "epochs": epochs, "lr": lr},
        {k: v for k, v in metrics.items() if v is not None},
        MLFLOW_EXPERIMENTS["regression" if task == "regression" else "classification"],
        artifacts={"training_curve": curve_path} if curve_path else None,
    )
    return model, history, metrics


def evaluate_cnn(model, data_loader, task):
    preds, targets = predict_model(model, data_loader, task=task)
    if task == "classification":
        labels = preds.argmax(axis=1)
        return classification_metrics(targets, labels, preds)
    return regression_metrics(targets, preds)
