import os
import torch
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam
from utils.utils import instantiate_from_config, cycle

class Trainer(object):
    def __init__(self, config, model, dataloader):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_epochs = config['solver']['train_epochs']
        self.save_cycle = self.train_epochs // 10
        self.dl = cycle(dataloader)        
        self.milestone = 0
        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        # optimizer
        start_lr = config['solver'].get('base_lr', 1.0e-4)
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        
        # scheduler 
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

    def train(self):
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
                
                if (curr_epoch+1) % 5000 == 0 :
                    path = os.path.join(self.results_folder, f"model_{curr_epoch+1}.pth")
                    torch.save(self.model.state_dict(), path)
                
    def train_decomp(self):
        curr_epoch = 0
        with tqdm(initial=curr_epoch, total=self.train_epochs) as pbar:
            while curr_epoch < self.train_epochs:
                data = next(self.dl).to(self.device)
                combined_loss, l1_loss, fourier_loss, adv_loss = self.model(data)
                combined_loss.backward()
                combined_loss = combined_loss.item()
                pbar.set_description(f'combiend_loss: {combined_loss:.6f} \n \
                                     l1_loss : {l1_loss:.6f} \n \
                                     fourier_loss : {fourier_loss:.6f} \n \
                                     adv_loss : {adv_loss:.6f} \n')
                
                self.opt.step()
                self.sch.step(combined_loss)
                self.opt.zero_grad()
                curr_epoch += 1

                pbar.update(1)
    
    
def train_prediction_model(model, dataloader, criterion, optimizer, device, epochs=100, description=""):
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
