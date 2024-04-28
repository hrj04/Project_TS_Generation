import os
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