import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        self.embedding = Conv_MLP(n_feat, n_embd)
        self.inverse = Conv_MLP(n_embd, n_feat)
        self.encoder = Encoder(n_embd=n_embd, 
                               n_heads=n_heads, 
                               mlp_hidden_times=mlp_hidden_times,
                               n_layer=n_layer_enc)
        self.decoder = Decoder(n_embd=n_embd,
                               seq_len=seq_len,
                               n_heads=n_heads, 
                               mlp_hidden_times=mlp_hidden_times,
                               condition_dim=n_embd,
                               n_layer=n_layer_dec)

    def forward(self, input):
        # encoding
        embedding = self.embedding(input)
        inp_enc = self.pe_encoder(embedding)
        enc_cond = self.encoder(inp_enc)

        # decoding
        inp_dec = self.pe_decoder(embedding)
        output = self.decoder(inp_dec, enc_cond)
        out = self.inverse(output)
        
        return out


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
                 seq_len,
                 n_heads,
                 mlp_hidden_times,
                 condition_dim,
                 n_layer,
                 ):
        super().__init__()
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd=n_embd,
                                                   seq_len=seq_len,
                                                   n_heads=n_heads,
                                                   mlp_hidden_times=mlp_hidden_times,
                                                   condition_dim=condition_dim) 
                                      for _ in range(n_layer)])

    def forward(self, x_emb, encoder_output):
        for block_idx in range(len(self.blocks)):
            x_emb = self.blocks[block_idx](x_emb, encoder_output)
        
        return x_emb


class DecoderBlock(nn.Module):
    def __init__(self,
                 seq_len,
                 n_embd,
                 n_heads,
                 mlp_hidden_times,
                 condition_dim):
        super().__init__()
        self.hidden_dim = mlp_hidden_times * n_embd
        self.ln = nn.LayerNorm(n_embd)
        self.full_attn = FullAttention(n_embd=n_embd,
                                       n_heads=n_heads)
        self.cross_attn = CrossAttention(n_embd=n_embd,
                                         condition_embd=condition_dim,
                                         n_heads=n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, n_embd))
        
        self.proj = nn.Conv1d(in_channels=seq_len, 
                              out_channels=seq_len * 2, 
                              kernel_size=1)

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
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals


