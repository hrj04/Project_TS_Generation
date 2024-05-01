import math
import yaml
import torch
import importlib
import numpy as np
import scipy

def cycle(dl):
    while True:
        for data in dl:
            yield data

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    
    return config

def instantiate_from_config(config):
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    
    return cls(**config.get("params", dict()))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps).float()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0, 0.999)


def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
  #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
   print(f'Final Score: {mean:0.5f} \xB1 {sigma:0.5f}')
   