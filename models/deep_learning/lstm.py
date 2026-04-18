import torch.nn as nn

from config import DROPOUT, EPOCHS, LEARNING_RATE, MLFLOW_EXPERIMENTS
from evaluation.metrics import classification_metrics, regression_metrics
from evaluation.visualization import plot_training_curves
from models.deep_learning.trainer import predict_model, train_model
from tracking.mlflow_tracking import log_pytorch_model


class LSTMModel(nn.Module):
    """2-layer LSTM for regression or classification.

    Input: (batch, seq_len, num_features)
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def train_lstm(train_loader, val_loader, input_size, task="regression", epochs=EPOCHS, lr=LEARNING_RATE):
    output_size = 1 if task == "regression" else 3
    model = LSTMModel(input_size=input_size, output_size=output_size)
    model, history, training_time = train_model(model, train_loader, val_loader, task=task, epochs=epochs, lr=lr)
    curve_path = plot_training_curves(history, f"lstm_{task}")
    metrics = {
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "training_time": training_time,
    }
    log_pytorch_model(
        model,
        f"lstm_{task}",
        {"task": task, "input_size": input_size, "epochs": epochs, "lr": lr},
        {k: v for k, v in metrics.items() if v is not None},
        MLFLOW_EXPERIMENTS["regression" if task == "regression" else "classification"],
        artifacts={"training_curve": curve_path} if curve_path else None,
    )
    return model, history, metrics


def evaluate_lstm(model, data_loader, task):
    preds, targets = predict_model(model, data_loader, task=task)
    if task == "classification":
        labels = preds.argmax(axis=1)
        return classification_metrics(targets, labels, preds)
    return regression_metrics(targets, preds)
