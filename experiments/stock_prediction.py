import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from torchmetrics import MeanAbsolutePercentageError

from data.dataloader import dl_from_numpy, dataloader_info
from models.predictor import GRU
from utils.utils import load_yaml_config

# Load configurations
configs = load_yaml_config("configs/stock.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset info
dl_info_train = dataloader_info(configs)
dl_info_test = dataloader_info(configs, train=False)
dataset = dl_info_train['dataset']
seq_length, feature_dim = dataset.window, dataset.feature_dim
batch_size = configs["dataloader"]["batch_size"]
lr = 0.001

# Load data
ori_dl = dl_from_numpy(os.path.join(dataset.dir, f"stock_ground_truth_{seq_length}_train.npy"), batch_size=batch_size)
fake_dl = dl_from_numpy(os.path.join(dataset.dir, f"ddpm_fake_stock.npy"), batch_size=batch_size)
dl_test = dl_info_test["dataloader"]

# Model training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=100, description=""):
    model.train()
    with tqdm(range(num_epochs), total=num_epochs) as pbar:
        for e in pbar:
            for data in dataloader:
                x_train = data[:, :seq_length - 1, :].float().to(device)
                y_train = data[:, seq_length - 1, 0].view(-1, 1).float().to(device)

                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            pbar.set_description(f"{description} loss: {loss.item():.6f}")
    

# Train on synthetic data
model_fake = GRU(input_dim=feature_dim, hidden_dim=50, num_layers=2).to(device)
criterion_fake = nn.MSELoss()
optimizer_fake = optim.Adam(model_fake.parameters(), lr=lr)
train_model(model_fake, fake_dl, criterion_fake, optimizer_fake, num_epochs=100, description="Synthetic")

# Train on original data
model_ori = GRU(input_dim=feature_dim, hidden_dim=50, num_layers=2).to(device)
criterion_ori = nn.MSELoss()
optimizer_ori = optim.Adam(model_ori.parameters(), lr=lr)
train_model(model_ori, ori_dl, criterion_ori, optimizer_ori, num_epochs=100, description="Original")

# Evaluate models on the test dataset
def evaluate_model(model, dataloader):
    model.eval()
    criterion = nn.MSELoss()
    mapeloss = MeanAbsolutePercentageError().to(device)
    total_loss = 0
    predictions, true_vals = [], []
    with torch.no_grad():
        for data in dataloader:
            x_test = data[:, :seq_length - 1, :].float().to(device)
            y_test = data[:, seq_length - 1, 0].view(-1, 1).float().to(device)
            y_pred = model(x_test)

            total_loss += criterion(y_pred, y_test) * len(data)
            predictions.append(y_pred.cpu().numpy())
            true_vals.append(y_test.cpu().numpy())

    total_loss /= len(dataloader.dataset)
    predictions = np.concatenate(predictions)
    true_vals = np.concatenate(true_vals)
    mape_loss = mapeloss(torch.tensor(predictions), torch.tensor(true_vals)).item()

    
    return total_loss, mape_loss, predictions, true_vals

loss_ori, mape_loss_ori, pred_y_ori, true_y = evaluate_model(model_ori, dl_test)
loss_fake, mape_loss_fake, pred_y_fake, _ = evaluate_model(model_fake, dl_test)

print(f"Test loss on original data: {loss_ori:0.5f}")
print(f"Test MAPE loss on original data: {mape_loss_ori:0.5f}")
print(f"Test loss on synthetic data: {loss_fake:0.5f}")
print(f"Test MAPE loss on synthetic data: {mape_loss_fake:0.5f}")

# Visualize predictions
plt.figure()
plt.plot(true_y[100:200], label='Original Data')
plt.plot(pred_y_ori[100:200], label='Predictions (Original Model)', linestyle='-', color='r')
plt.plot(pred_y_fake[100:200], label='Predictions (Synthetic Model)', linestyle='--', color='b')
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Time Step")
plt.ylabel("Closing Price")
plt.savefig("test.png")

