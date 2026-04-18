import torch.nn as nn

from config import DROPOUT, EPOCHS, LEARNING_RATE, MLFLOW_EXPERIMENTS
from evaluation.metrics import classification_metrics, regression_metrics
from evaluation.visualization import plot_training_curves
from models.deep_learning.trainer import predict_model, train_model
from tracking.mlflow_tracking import log_pytorch_model


class MLPModel(nn.Module):
    """Multi-Layer Perceptron for regression or classification.

    Input is flattened: (batch, seq_len * num_features) for sequence data
    or (batch, num_features) for tabular data.
    """

    def __init__(self, input_size, output_size=1, dropout=DROPOUT):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


def train_mlp(train_loader, val_loader, input_size, task="regression", epochs=EPOCHS, lr=LEARNING_RATE):
    seq_len = getattr(getattr(train_loader, "dataset", None), "seq_len", 1)
    output_size = 1 if task == "regression" else 3
    model = MLPModel(input_size=input_size * seq_len, output_size=output_size)
    model, history, training_time = train_model(model, train_loader, val_loader, task=task, epochs=epochs, lr=lr)
    curve_path = plot_training_curves(history, f"mlp_{task}")
    metrics = {
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "training_time": training_time,
    }
    log_pytorch_model(
        model,
        f"mlp_{task}",
        {"task": task, "input_size": input_size * seq_len, "epochs": epochs, "lr": lr},
        {k: v for k, v in metrics.items() if v is not None},
        MLFLOW_EXPERIMENTS["regression" if task == "regression" else "classification"],
        artifacts={"training_curve": curve_path} if curve_path else None,
    )
    return model, history, metrics


def evaluate_mlp(model, data_loader, task):
    preds, targets = predict_model(model, data_loader, task=task)
    if task == "classification":
        labels = preds.argmax(axis=1)
        return classification_metrics(targets, labels, preds)
    return regression_metrics(targets, preds)
