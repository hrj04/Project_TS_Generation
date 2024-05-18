import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from utils.utils import extract, cosine_beta_schedule, linear_beta_schedule
from einops import reduce, rearrange, repeat
from torch.optim import Adam
import copy


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
            use_ff,
            loss_decomposition,
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
        self.loss_fn = F.l1_loss if loss_type == "l1" else F.mse_loss
        self.seq_length = seq_length
        self.n_feat = n_feat
        self.ff_weight = math.sqrt(self.seq_length) / 5
        self.use_ff = use_ff
        self.loss_decomposition = loss_decomposition

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

    def predict_x_0(self, x_t, t):
        trend, season = self.transformer(x_t, t)
        x_0_hat = trend + season
        
        return x_0_hat
    
    def _train_loss(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self._forward_process(x_0=x_0, t=t, noise=noise)
        x_0_pred = self.predict_x_0(x_t, t)
        
        # l1 loss
        l1_loss = self.loss_fn(x_0_pred, x_0, reduction='none')

        # fourier_loss
        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fourier_loss = torch.tensor([0.])
            fft1 = torch.fft.fft(x_0_pred.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(x_0.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_real = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
            fourier_img = self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            fourier_loss = fourier_real + fourier_img
            fourier_loss = self.ff_weight * fourier_loss
            combined_loss = l1_loss + fourier_loss
            combined_loss = reduce(combined_loss, 'b ... -> b (...)', 'mean')
        else:
            combined_loss = l1_loss
        combined_loss = combined_loss * extract(self.loss_weight, t, combined_loss.shape)

        if self.loss_decomposition :
            return combined_loss.mean(), l1_loss.mean(), fourier_loss.mean()
        
        return combined_loss.mean()
    
    def set_predictor(self, predictor):
        self.predictor = copy.deepcopy(predictor)

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
        x_0 = self.predict_x_0(x_t, t=batched_t)
        x_0.clamp_(min=-4., max=4.)
        
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce, rearrange, repeat


class Transformer(nn.Module):
    def __init__(self,
                 n_feat,
                 seq_len,
                 n_embd,
                 n_heads,
                 mlp_hidden_times,
                 n_layer_enc,
                 n_layer_dec
                 ):
        super().__init__()
        self.pe_encoder = LearnablePositionalEncoding(embd_dim=n_embd,seq_len=seq_len,dropout=0.1)
        self.pe_decoder = LearnablePositionalEncoding(embd_dim=n_embd,seq_len=seq_len,dropout=0.1)
        self.combine_m = nn.Conv1d(in_channels=n_layer_dec, 
                                   out_channels=1, 
                                   kernel_size=1, 
                                   padding_mode='circular', 
                                   bias=False)
        self.combine_s = nn.Conv1d(in_channels=n_embd,
                                   out_channels=n_feat,
                                   kernel_size=1, 
                                   padding_mode='circular', 
                                   bias=False)
        self.embedding = Conv_MLP(n_feat, n_embd)
        self.inverse = Conv_MLP(n_embd, n_feat)
        self.encoder = Encoder(n_embd=n_embd, 
                               n_heads=n_heads, 
                               mlp_hidden_times=mlp_hidden_times,
                               n_layer=n_layer_enc)
        self.decoder = Decoder(n_embd=n_embd,
                               n_feat=n_feat,
                               seq_len=seq_len,
                               n_heads=n_heads, 
                               mlp_hidden_times=mlp_hidden_times,
                               condition_dim=n_embd,
                               n_layer=n_layer_dec)

    def forward(self, input, t):
        # encoding
        embedding = self.embedding(input)
        inp_enc = self.pe_encoder(embedding)
        enc_cond = self.encoder(inp_enc)

        # decoding
        inp_dec = self.pe_decoder(embedding)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1,2)).transpose(1,2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend
        
        return trend, season_error


class Encoder(nn.Module):
    def __init__(self,
                 n_embd,
                 n_heads,
                 mlp_hidden_times,
                 n_layer
                 ):
        super().__init__()
        self.blocks = nn.Sequential(*[EncoderBlock(n_embd=n_embd,
                                                   n_heads=n_heads,
                                                   mlp_hidden_times=mlp_hidden_times
                                                   ) for _ in range(n_layer)])
        
    def forward(self, x_emb):
        for block_idx in range(len(self.blocks)):
            x_emb = self.blocks[block_idx](x_emb)
        
        return x_emb
  

class EncoderBlock(nn.Module):
    def __init__(self,
                 n_embd,
                 n_heads,
                 mlp_hidden_times):
        super().__init__()
        self.hidden_dim = mlp_hidden_times * n_embd
        self.ln = nn.LayerNorm(n_embd)
        self.attn = FullAttention(n_embd=n_embd,
                                  n_heads=n_heads)   
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, n_embd),
            )

    def forward(self, x_emb):
        a, _ = self.attn(self.ln(x_emb))
        x_emb = x_emb + a
        x_emb = x_emb + self.mlp(self.ln(x_emb))
        
        return x_emb

    
class Decoder(nn.Module):
    def __init__(self,
                 n_embd,
                 n_feat,
                 seq_len,
                 n_heads,
                 mlp_hidden_times,
                 condition_dim,
                 n_layer,
                 ):
        super().__init__()
        self.n_embd = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd=n_embd,
                                                   n_feat=n_feat,
                                                   seq_len=seq_len,
                                                   n_heads=n_heads,
                                                   mlp_hidden_times=mlp_hidden_times,
                                                   condition_dim=condition_dim) 
                                      for _ in range(n_layer)])

    def forward(self, x_emb, t, encoder_output):
        b, c, _ = x_emb.shape
        mean = []
        season = torch.zeros((b, c, self.n_embd), device=x_emb.device)
        trend = torch.zeros((b, c, self.n_feat), device=x_emb.device)
        for block_idx in range(len(self.blocks)):
            x_emb, residual_mean, residual_trend, residual_season = self.blocks[block_idx](x_emb, t, encoder_output)
            trend += residual_trend
            season += residual_season
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        
        return x_emb, mean, trend, season


class DecoderBlock(nn.Module):
    def __init__(self,
                 seq_len,
                 n_feat,
                 n_embd,
                 n_heads,
                 mlp_hidden_times,
                 condition_dim):
        super().__init__()
        self.hidden_dim = mlp_hidden_times * n_embd
        self.layernorm = nn.LayerNorm(n_embd)
        self.adalayernorm1 = AdaLayerNorm(n_embd)
        self.adalayernorm2 = AdaLayerNorm(n_embd)
        self.full_attn = FullAttention(n_embd=n_embd,
                                       n_heads=n_heads)
        self.cross_attn = CrossAttention(n_embd=n_embd,
                                         condition_embd=condition_dim,
                                         n_heads=n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, n_embd))
        
        self.trend = TrendBlock(in_dim=seq_len, 
                                out_dim=seq_len,
                                in_feat=n_embd,
                                out_feat=n_feat)
        self.seasonal = FourierLayer(n_embd=n_embd)
        self.proj = nn.Conv1d(in_channels=seq_len, 
                              out_channels=seq_len * 2, 
                              kernel_size=1)
        self.linear = nn.Linear(in_features=n_embd,
                                out_features=n_feat)
        
    def forward(self, x_emb, t, encoder_output):
        a, _ = self.full_attn(self.adalayernorm1(x_emb, t))
        x_emb = x_emb + a
        a, _ = self.cross_attn(self.adalayernorm2(x_emb, t), encoder_output)
        x_emb = x_emb + a
        
        # decomposition
        x1, x2 = self.proj(x_emb).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x_emb = x_emb + self.mlp(self.layernorm(x_emb))
        m = torch.mean(x_emb, dim=1, keepdim=True)

        return x_emb - m, self.linear(m), trend, season
 
 
class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 n_heads
    ):
        super().__init__()
        self.n_heads = n_heads
        assert n_embd % n_heads == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.proj(y)
        
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 condition_embd,
                 n_heads,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_heads = n_heads

    def forward(self, x, encoder_output):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.proj(y)
        
        return y, att


class TrendBlock(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 in_feat, 
                 out_feat):
        super().__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, 
                      out_channels=trend_poly, 
                      kernel_size=3, 
                      padding=1),
            nn.GELU(),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_feat, 
                      out_channels=out_feat, 
                      kernel_size=3, 
                      padding=1)
        )
        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        x = self.trend(input)
        trend_vals = torch.matmul(x, self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        
        return trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, 
                 n_embd, 
                 low_freq=1, 
                 factor=1):
        super().__init__()
        self.n_embd = n_embd
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        
        return x_freq, index_tuple
  

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, 
                 embd_dim, 
                 seq_len,
                 dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoder = nn.Parameter(torch.empty(1, seq_len, embd_dim))
        nn.init.uniform_(self.positional_encoder, -0.02, 0.02)

    def forward(self, x):
        x = x + self.positional_encoder
        
        return self.dropout(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb
  
    
class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.emb(timestep)
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        
        return x


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module"""
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
