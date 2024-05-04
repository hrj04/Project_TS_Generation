import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import reduce
from tqdm.auto import tqdm
from models.transformer import Transformer
from utils.utils import extract, cosine_beta_schedule, linear_beta_schedule


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            n_feat,
            n_embd,
            timesteps,
            loss_type,
            beta_sch,
            n_heads,
            mlp_hidden_times,
            n_layer_enc,
            n_layer_dec,
            use_ff
    ):
        super().__init__()
        self.transformer = Transformer(n_feat=n_feat,
                                       seq_len=seq_length, 
                                       n_embd=n_embd,
                                       n_heads=n_heads, 
                                       mlp_hidden_times=mlp_hidden_times,
                                       n_layer_enc=n_layer_enc,
                                       n_layer_dec=n_layer_dec)
        self.timesteps = int(timesteps)
        self.loss_fn = F.l1_loss if loss_type == "l1" else F.l2_loss
        self.seq_length = seq_length
        self.n_feat = n_feat
        self.ff_weight = math.sqrt(self.seq_length) / 5
        self.use_ff = use_ff

        # To enhance computing performance
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # diffusion
        register_buffer('betas', cosine_beta_schedule(timesteps) if beta_sch=="cosine" else linear_beta_schedule(timesteps))
        register_buffer('alphas', 1. - self.betas)
        register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.))
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        
        # calculations for post q(x_{t-1} | x_t, x_0)
        register_buffer('posterior_variance', self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        register_buffer('posterior_log_variance', torch.log(self.posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))

        register_buffer('loss_weight', torch.sqrt(self.alphas) * torch.sqrt(1. - self.alphas_cumprod) / self.betas / 100)

    def forward(self, x_0):
        batch, device = x_0.shape[0], x_0.device
        t = torch.randint(0, self.timesteps, (batch,), device=device).long()
        
        return self._train_loss(x_0=x_0, t=t)

    def _train_loss(self, x_0, t):
        
        noise = torch.randn_like(x_0)
        x_t = self._forward_process(x_0=x_0, t=t, noise=noise)
        x_0_pred = self.transformer(x_t)
        
        # l1 loss
        train_loss = self.loss_fn(x_0_pred, x_0, reduction='none')
        
        # fourier_loss
        if self.use_ff:
            fourier_loss = torch.tensor([0.])
            fft1 = torch.fft.fft(x_0_pred.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(x_0.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_real = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
            fourier_img = self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            fourier_loss = fourier_real + fourier_img
            
            # combine loss function
            train_loss +=  self.ff_weight * fourier_loss
            train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        
        return train_loss.mean()

    def _forward_process(self, x_0, t, noise):
        coef1 = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        coef2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = coef1 * x_0 + coef2 * noise
        
        return x_t
    
    @torch.no_grad()
    def generate_mts(self, batch_size):
        device = self.betas.device
        shape = (batch_size, self.seq_length, self.n_feat)
        
        x_T = torch.randn(shape, device=device)
        synthetic_mts = self._reverse_process(x_T=x_T)

        return synthetic_mts.detach().cpu().numpy()
    
    def _reverse_process(self, x_T):
        desc='reverse step from x_T to x_0'
        for t in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps, desc=desc):
            x_T = self._posterior_q(x_t=x_T, t=t)
        x_0_hat = x_T
        
        return x_0_hat
        
    def _posterior_q(self, x_t, t: int):
        '''posterior q(x_{t-1} | x_t, x_0)'''
        batched_t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x_0 = self.transformer(x_t)
        x_0.clamp_(min=-1., max=1.)
        
        post_mean = self._post_mean(x_0=x_0, x_t=x_t, t=batched_t)
        post_log_variance = self._post_log_var(t=batched_t, shape=x_t.shape)
        z = torch.randn_like(x_t) if t > 0 else 0.
        x_t_1 = post_mean + (0.5 * post_log_variance).exp() * z
        
        return x_t_1

    def _post_mean(self, x_0, x_t, t):
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        post_mean = coef1 * x_0 + coef2 * x_t
        
        return post_mean

    def _post_log_var(self, t, shape):
        post_log_variance = extract(self.posterior_log_variance, t, shape)
        
        return post_log_variance

