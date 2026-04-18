import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import DEVICE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, BATCH_SIZE, get_logger

logger = get_logger(__name__)


class Autoencoder(nn.Module):
    """Symmetric autoencoder for unsupervised feature learning.

    Input: (batch, num_features)
    Latent: 32-dim representation
    """

    def __init__(self, input_size, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train_autoencoder(X_train, X_val, input_size, latent_dim=32,
                      epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE):
    """Train autoencoder on feature data (unsupervised).

    Args:
        X_train, X_val: numpy arrays of features
        input_size: number of features
        latent_dim: dimension of latent space

    Returns:
        model, history, training_time
    """
    model = Autoencoder(input_size, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience_counter = 0

    import time
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                output = model(batch)
                loss = criterion(output, batch)
                val_losses.append(loss.item())

        avg_train = np.mean(losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"  AE Epoch {epoch+1}/{epochs} — train: {avg_train:.6f}, val: {avg_val:.6f}")

        if patience_counter >= patience:
            logger.info(f"  AE Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    return model, history, training_time


def extract_latent_features(model, X):
    """Encode data into latent space.

    Args:
        model: trained Autoencoder
        X: numpy array of features

    Returns:
        numpy array of latent representations (N x latent_dim)
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        latent = model.encode(tensor).cpu().numpy()
    return latent
