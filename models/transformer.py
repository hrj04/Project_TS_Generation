import math
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,
                 n_feat,
                 n_embd,
                 n_heads,
                 mlp_hidden_times):
        super().__init__()
        self.embedding = Conv_MLP(n_feat, n_embd)
        self.inverse = Conv_MLP(n_embd, n_feat)
        self.encoder = Encoder(n_embd=n_embd, 
                               n_heads=n_heads, 
                               mlp_hidden_times=mlp_hidden_times)
        self.decoder = Decoder(n_feat=n_feat, 
                               n_embd=n_embd, 
                               n_heads=n_heads, 
                               mlp_hidden_times=mlp_hidden_times,
                               condition_dim=n_embd)

    def forward(self, input):
        # encoding
        embedding = self.embedding(input)
        enc_cond = self.encoder(embedding)

        # decoding
        output = self.decoder(embedding, enc_cond)
        out = self.inverse(output)
        
        return out


class Encoder(nn.Module):
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
                 n_feat,
                 n_embd,
                 n_heads,
                 mlp_hidden_times,
                 condition_dim):
        super().__init__()
        self.hidden_dim = mlp_hidden_times * n_embd
        self.ln = nn.LayerNorm(n_embd)
        self.full_attn = FullAttention(n_embd=n_embd,
                                       n_heads=n_heads,)
        self.cross_attn = CrossAttention(n_embd=n_embd,
                                         condition_embd=condition_dim,
                                         n_heads=n_heads,)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, n_embd),
        )
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x_emb, encoder_output):
        a, _ = self.full_attn(self.ln(x_emb))
        x_emb = x_emb + a
        a, _ = self.cross_attn(self.ln(x_emb), encoder_output)
        x_emb = x_emb + a
        x_emb = x_emb + self.mlp(self.ln(x_emb))

        return x_emb
    

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

