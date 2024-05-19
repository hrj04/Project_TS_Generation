import os
import torch
import numpy as np
import FinanceDataReader as fdr

from torch import Tensor
from tqdm.auto import tqdm
from torch.utils.data import Dataset

        
class SineDataset(Dataset):
    def __init__(self, 
                 n_samples=10000, 
                 window=24, 
                 feature_dim=5, 
                 save_ground_truth=True, 
                 seed=2024,
                 period='train',
                 ):
        super().__init__()
        self.dir = './output'
        os.makedirs(self.dir, exist_ok=True)
        self.window = window
        self.feature_dim = feature_dim
        self.data = self.sine_data_generation(n_samples=n_samples, 
                                              window=window, 
                                              feature_dim=feature_dim, 
                                              seed=seed)
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"sine_ground_truth_{window}_{period}.npy"), self.data)

    def sine_data_generation(self,
                             n_samples : int, 
                             window : int, 
                             feature_dim : int, 
                             seed : int, 
                             ):
        np.random.seed(seed)
        sine_data = list()
        for _ in tqdm(range(n_samples), total=n_samples, desc="Sampling sine-dataset"):
            sine = list()
            for _ in range(feature_dim):
                freq, phase = np.random.uniform(0, 0.1, 2)            
                sine_sequence = [np.sin(freq * j + phase) for j in range(window)]
                sine.append(sine_sequence)
            sine_data.append(np.array(sine).T)  
        sine_data = torch.from_numpy(np.array(sine_data)).float()
        
        return sine_data
  
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class StockDataset(Dataset):
    def __init__(self, 
                 symbol : str = "AAPL", 
                 sdate : str = "2000", 
                 edate : str = "2024",
                 window : int = 24,
                 save_ground_truth=True, 
                 normalize=True,
                 period='train',
                 ):
        
        raw_df = fdr.DataReader(symbol, sdate, edate)
        self.data = self.generate_stock_sample(raw_df, window)
        self.window = window
        self.feature_dim = self.data.shape[-1]
        self.dir = './output'
        os.makedirs(self.dir, exist_ok=True)

        if normalize:
            self.data, self.mean, self.std = self._mean_std_scale(self.data)
            
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"stock_ground_truth_data_{window}_{period}.npy"), self.data)
            np.save(os.path.join(self.dir, f"stock_ground_truth_mean_{window}_{period}.npy"), self.mean)
            np.save(os.path.join(self.dir, f"stock_ground_truth_std_{window}_{period}.npy"), self.std)

    def generate_stock_sample(self, df, window):
        raw_data = torch.from_numpy(df.to_numpy()).float()
        data = self._extract_sliding_windows(raw_data, window)
        
        return data
    
    def _extract_sliding_windows(self, raw_data, window):
        sample_n = len(raw_data)-window+1
        n_feature = raw_data.shape[-1]
        data = torch.zeros(sample_n, window, n_feature)
        for i in range(sample_n):
            start = i
            end = i + window    
            data[i, :, :] = raw_data[start:end]
            
        return data

    def _mean_std_scale(self, data):
        epsilon = 1e-8
        std = data.std(dim=1, keepdim=True)
        mean = data.mean(dim=1, keepdim=True)
        scaled_data = (data-mean)/(std+epsilon)
        
        return scaled_data, mean, std

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SyntheticDataset(Dataset):
    def __init__(self, 
                 n_samples : int = 10000, 
                 window : int = 24, 
                 feature_dim : int = 5,
                 noise_level : float = 2, 
                 save_ground_truth=True, 
                 normalize=True,
                 seed=2024,
                 period='train',
                 ):
        self.window = window
        self.feature_dim = feature_dim
        self.dir = './output'
        os.makedirs(self.dir, exist_ok=True)
        self.data = self.generate_synthetic_ts(n_samples, 
                                               window,
                                               feature_dim,
                                               noise_level,
                                               seed)
        self.normalize = normalize
        if self.normalize:
            self.data, self.min_val, self.max_val = self._min_max_scale(self.data)
        
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"synthetic_ground_truth_{window}_{period}.npy"), self.data)

    def generate_synthetic_ts(self, 
                              n_samples,
                              window,
                              feature_dim,
                              noise_level,
                              seed
                              ) -> Tensor:
        np.random.seed(seed)
        syn_data = list()
        for _ in tqdm(range(n_samples), total=n_samples, desc="Sampling synthetic-dataset"):
            synthetic = list()
            for _ in range(feature_dim):
                freq, phase = np.random.uniform(0, 3, 2)            
                linear_trend = np.linspace(0, np.random.uniform(-6,6), window)
                seasonal = [np.sin(freq * j + phase) for j in range(window)]
                noise = np.random.normal(0, noise_level, window)
                synthetic.append(linear_trend+seasonal+noise)
            syn_data.append(np.array(synthetic).T)    
        syn_data = torch.from_numpy(np.array(syn_data)).float()

        return syn_data
    
    def _min_max_scale(self, data : Tensor):
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        scaled_data = (data-min_val)/(max_val - min_val)
        
        return scaled_data, min_val, max_val
    
    def _inverse_min_max_scale(scaled_data : Tensor,
                            min_val : Tensor, 
                            max_val : Tensor):
        origin_data = scaled_data*(max_val-min_val)+ min_val

        return origin_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
class FromNumpyDataset(Dataset):
    def __init__(self, 
                 path
                 ):
        self.data = np.load(path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Stock(Dataset):
    def __init__(self,
                 symbol : str = "AAPL, MSFT, NVDA, AMZN, COST", 
                 sdate : str = "20000101", 
                 edate : str = "20131231",
                 window : int = 24,
                 save_ground_truth : bool =True, 
                 period : str ='train'):
        super().__init__()
        raw_df = fdr.DataReader(symbol, sdate, edate)
        self.data, self.mean, self.std = self._generate_stock_sequence(raw_df, window)
        self.window = window
        self.feature_dim = self.data.shape[-1]
        self.dir = './dataset/stock'
        os.makedirs(self.dir, exist_ok=True)
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"origin_norm_{window}_{period}.npy"), self.data)
            np.save(os.path.join(self.dir, f"origin_mean_{window}_{period}.npy"), self.mean)
            np.save(os.path.join(self.dir, f"origin_std_{window}_{period}.npy"), self.std)
    
    def _generate_stock_sequence(self, df, window):
        raw_data = torch.from_numpy(df.to_numpy()).float()
        data = self._extract_sliding_windows(raw_data, window)
        data, mean, std = self._mean_std_scale(data)

        return data, mean, std
    
    def _extract_sliding_windows(self, raw_data, window):
        sample_n = len(raw_data) - window + 1
        n_feature = raw_data.shape[-1]
        data = torch.zeros(sample_n, window, n_feature)
        for i in range(sample_n):
            start = i
            end = i + window    
            data[i, :, :] = raw_data[start:end]
            
        return data

    def _mean_std_scale(self, data):
        epsilon = 1e-8
        past_data = data[:, :-1, :]
        target_data = data[:, -1:, :]
        mean = torch.mean(past_data, dim=1, keepdim=True)
        std = torch.std(past_data, dim=1, keepdim=True)
        
        past_data_normalized = (past_data - mean) / (std + epsilon)
        target_data_normalized = (target_data - mean) / (std + epsilon)
        data_normalized = torch.cat((past_data_normalized, target_data_normalized), dim=1)
        
        return data_normalized, mean, std
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.mean[idx], self.std[idx]

    
class StockDifferencing(Dataset):
    def __init__(self,
                 symbol : str = "AAPL, MSFT, NVDA, AMZN, COST", 
                 sdate : str = "20000101", 
                 edate : str = "20131231",
                 window : int = 24,
                 save_ground_truth : bool =True, 
                 period : str ='train'):
        super().__init__()
        raw_df = fdr.DataReader(symbol, sdate, edate)
        self.data_diff, self.data, self.mean, self.std = self._generate_stock_sequence(raw_df, window)
        self.window = window
        self.feature_dim = self.data.shape[-1]
        self.dir = './dataset/stock_diff'
        os.makedirs(self.dir, exist_ok=True)
        
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"origin_diff_{window}_{period}.npy"), self.data_diff)
            np.save(os.path.join(self.dir, f"origin_norm_{window}_{period}.npy"), self.data)
            np.save(os.path.join(self.dir, f"origin_mean_{window}_{period}.npy"), self.mean)
            np.save(os.path.join(self.dir, f"origin_std_{window}_{period}.npy"), self.std)
    
    def _generate_stock_sequence(self, df, window):
        raw_data = torch.from_numpy(df.to_numpy()).float()
        data = self._extract_sliding_windows(raw_data, window)
        data, mean, std = self._mean_std_scale(data)
        data_diff = self._differencing(data, n_order=1, axis=1)

        return data_diff, data, mean, std
    
    def _differencing(self, raw_data, n_order, axis):
        data_diff = np.diff(raw_data, n_order, axis)
        
        return data_diff
    
    def _extract_sliding_windows(self, raw_data, window):
        sample_n = len(raw_data) - window + 1
        n_feature = raw_data.shape[-1]
        data = torch.zeros(sample_n, window, n_feature)
        for i in range(sample_n):
            start = i
            end = i + window    
            data[i, :, :] = raw_data[start:end]
            
        return data
    
    def _mean_std_scale(self, data):
        epsilon = 1e-8
        past_data = data[:, :-1, :]
        target_data = data[:, -1:, :]
        mean = torch.mean(past_data, dim=1, keepdim=True)
        std = torch.std(past_data, dim=1, keepdim=True)
        
        past_data_normalized = (past_data - mean) / (std + epsilon)
        target_data_normalized = (target_data - mean) / (std + epsilon)
        data_normalized = torch.cat((past_data_normalized, target_data_normalized), dim=1)
        
        return data_normalized, mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_diff[idx], self.data[idx], self.mean[idx], self.std[idx]

