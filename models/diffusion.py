import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from models.transformer import Transformer
from utils.utils import extract, cosine_beta_schedule


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            n_feat,
            n_embd=None,
            timesteps=1000,
            loss_type='l1',
            n_heads=4,
            mlp_hidden_times=4,
    ):
        super().__init__()
        
        # model
        self.transformer = Transformer(n_feat=n_feat, 
                                       n_embd=n_embd,
                                       n_heads=n_heads, 
                                       mlp_hidden_times=mlp_hidden_times)
        self.timesteps = int(timesteps)
        self.loss_type = loss_type
        self.loss_fn = F.l1_loss if self.loss_type == "l1" else F.l2_loss
        self.seq_length = seq_length
        self.n_feat = n_feat

        # To enhance computing performance
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # diffusion
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Used in Forward Process
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # calculations for post q(x_{t-1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def forward(self, x_0):
        batch, device = x_0.shape[0], x_0.device
        t = torch.randint(0, self.timesteps, (batch,), device=device).long()
        
        return self._train_loss(x_0=x_0, t=t)

    def _train_loss(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self._forward_process(x_0=x_0, t=t, noise=noise)
        x_0_pred = self.transformer(x_t)
        train_loss = self.loss_fn(x_0_pred, x_0, reduction='mean')
        
        return train_loss

    def _forward_process(self, x_0, t, noise):
        coef1 = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        coef2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = coef1 * x_0 + coef2 * noise
        
        return x_t
    
    @torch.no_grad()
    def generate_mts(self, batch_size):
        device = self.betas.device
        shape = (batch_size, self.seq_length, self.n_feat)
        
        z = torch.randn(shape, device=device)
        synthetic_mts = self._reverse_process(x_t=z)

        return synthetic_mts.detach().cpu().numpy()
    
    def _reverse_process(self, x_t):
        desc='reverse step from x_t to x_0'
        for t in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps, desc=desc):
            x_t = self._p_reverse(x_t, t)
        x_0_hat = x_t
        
        return x_0_hat
        
    def _p_reverse(self, x_t, t: int):
        batched_t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x_0_hat = self.transformer(x_t)
        x_0_hat.clamp_(-1., 1.)
        
        post_mean = self._post_mean(x_0=x_0_hat, x_t=x_t, t=batched_t)
        post_log_variance = self._post_log_var(t=batched_t, shape=x_t.shape)
        
        noise = torch.randn_like(x_t) if t > 0 else 0.
        x_t_1 = post_mean + (0.5 * post_log_variance).exp() * noise
        
        return x_t_1
        
    def _post_log_var(self, t, shape):
        post_log_variance = extract(self.posterior_log_variance_clipped, t, shape)
        
        return post_log_variance

    def _post_mean(self, x_0, x_t, t):
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        post_mean = coef1 * x_0 + coef2 * x_t
        
        return post_mean

    