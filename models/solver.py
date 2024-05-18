import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.optim import Adam
from utils.utils import instantiate_from_config, cycle
from torchmetrics import MeanAbsolutePercentageError
from torch import nn
import numpy as np

class Trainer(object):
    def __init__(self, config_solver, model, dataloader):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_epochs = config_solver['train_epochs']
        self.dl = cycle(dataloader)        
        self.results_folder = Path(config_solver['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        # Only optimizer for Transformer parameters
        start_lr = config_solver.get('base_lr', 1.0e-4)
        self.opt = Adam(self.model.transformer.parameters(), lr=start_lr, betas=[0.9, 0.96])

        # scheduler 
        sc_cfg = config_solver['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

    def train(self):
        self.model.train()
        curr_epoch = 0
        with tqdm(initial=curr_epoch, total=self.train_epochs) as pbar:
            while curr_epoch < self.train_epochs:
                data = next(self.dl).to(self.device)
                loss = self.model(data)
                loss.backward()
                loss = loss.item()
                pbar.set_description(f'loss: {loss:.6f}')
                
                self.opt.step()
                self.sch.step(loss)
                self.opt.zero_grad()
                curr_epoch += 1

                pbar.update(1)
                
                if (curr_epoch) % 5000 == 0 :
                    path = os.path.join(self.results_folder, f"model_{curr_epoch}.pth")
                    torch.save(self.model.state_dict(), path)
                
    def train_decomp(self):
        self.model.train()
        curr_epoch = 0
        with tqdm(total=self.train_epochs) as pbar:
            while curr_epoch < self.train_epochs:
                data = next(self.dl).to(self.device)
                combined_loss, l1_loss, fourier_loss = self.model(data)
                combined_loss.backward()
                combined_loss = combined_loss.item()

                description = f'combiend_loss: {combined_loss:.6f} l1_loss : {l1_loss.item():.6f} fourier_loss : {fourier_loss.item():.6f}'
                pbar.set_description(description)
                
                self.opt.step()
                self.sch.step(combined_loss)
                self.opt.zero_grad()
                curr_epoch += 1
                pbar.update(1)

                if curr_epoch % 5000 == 0 :
                    path = os.path.join(self.results_folder, f"model_{curr_epoch}.pth")
                    torch.save(self.model.state_dict(), path)


def train_prediction_model(model, dataloader, criterion, optimizer, device, epochs=100, description="", save_path=""):
    model.train()
    with tqdm(range(epochs), total=epochs) as pbar:
        for _ in pbar:
            for data in dataloader:
                x_train = data[:,:-1,:].float().to(device)
                y_train = data[:,-1:,0].float().to(device)
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            pbar.set_description(f"{description} loss: {loss.item():.6f}")
    torch.save(model.state_dict(), save_path)
                


def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()
    criterion3 = MeanAbsolutePercentageError().to(device)
    total_loss_L1 = 0
    total_loss_MSE = 0
    total_loss_mape = 0

    predictions, true_vals = [], []
    with torch.no_grad():
        try:
            for data, mean, std in dataloader:
                x_test = data[:, :-1, :].float().to(device)
                y_test = data[:, -1:, :1].float().to(device)
                mean = mean[:, :, :1].float().to(device)
                std = std[:, :, :1].float().to(device)
                y_pred = model(x_test).view(-1,1,1)
                
                total_loss_L1 += criterion(y_pred, y_test) * len(data)
                total_loss_MSE += criterion2(y_pred, y_test) * len(data)
                total_loss_mape += criterion3(y_pred, y_test).item() * len(data)

                y_test_unnorm = y_test * std + mean
                y_pred_unnorm = y_pred * std + mean

                predictions.append(y_pred_unnorm.cpu().numpy())
                true_vals.append(y_test_unnorm.cpu().numpy())
        except:
            for data in dataloader:
                x_test = data[:, :-1, :].float().to(device)
                y_test = data[:, -1:, :1].float().to(device)
                y_pred = model(x_test).view(-1,1,1)
                
                total_loss_L1 += criterion(y_pred, y_test) * len(data)
                total_loss_MSE += criterion2(y_pred, y_test) * len(data)
                total_loss_mape += criterion3(y_pred, y_test).item() * len(data)

                predictions.append(y_pred.cpu().numpy())
                true_vals.append(y_test.cpu().numpy())

    total_loss_L1 /= len(dataloader.dataset)
    total_loss_MSE /= len(dataloader.dataset)
    total_loss_mape /= len(dataloader.dataset)
    
    predictions = np.concatenate(predictions).squeeze()
    true_vals = np.concatenate(true_vals).squeeze()
    
    return total_loss_L1, total_loss_MSE, total_loss_mape, predictions, true_vals