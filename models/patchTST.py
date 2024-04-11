from typing import ParamSpecKwargs
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from x_transformers import Encoder
from models.revIN import RevIN
'''
Future embedding strategies:
1) patchify each chanel into several tokens, concat tokens
2) make spectrogram, then pass to vit or CNN
'''


'''
Future embedding strategies:
1) patchify each chanel into several tokens, concat tokens
2) make spectrogram, then pass to vit or CNN
'''


class Embedding(nn.Module):
    def __init__(self, input_size, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        x = x.float()
        return self.embed(x)

class PatchTSTEncoder(nn.Module):
    def __init__(self, seq_len,  num_channels, embed_dim, heads, depth, patch_len=8, dropout=0.0, embed_strat='patch', decay=0.9, ema=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.heads = heads
        self.depth = depth
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.dropout = dropout
        self.embed_strat = embed_strat
        self.decay = decay
        self.ema = ema
        
        if self.embed_strat == 'patch':
          # learnable embeddings for each channel
          self.embed = Embedding(patch_len, embed_dim)
          # learnable positional encoding
          self.pe = nn.Parameter(torch.randn(1, (seq_len // patch_len), embed_dim))

        elif self.embed_strat == 'learned_table':
          self.embed = nn.Embedding(num_embeddings=2001, embedding_dim=embed_dim)
          self.pe = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        elif self.embed_strat == 'naive_linear':
          self.embed = nn.Linear(1, embed_dim)
          self.pe = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        elif self.embed_strat == 'max' or self.embed_strat == 'mean' or self.embed_strat == 'sum':
          self.embed = nn.Linear(1, embed_dim)
          self.pe = nn.Parameter(torch.randn(1, (seq_len // patch_len), embed_dim))


        else:
          pass
        # transformer encoder
        self.encoder = Encoder(
            dim = embed_dim,
            depth = depth,
            heads = heads,
            dropout = self.dropout,
            sandwich_norm = True
        )

    '''
    patchify each channel into several tokens,
    '''
    def patchify(self, x):
        # shape x: (batch_size, seq_len, num_channels)
        x = rearrange(x, 'b (s patch_len) c -> b c s patch_len', patch_len=self.patch_len)
        return x 

    def forward(self, x):
        # instance norm
        if self.embed_strat == 'patch' or self.embed_strat == 'max' or self.embed_strat == 'sum' or self.embed_strat == 'mean':
          # if ssl we do everything separately
          x = self.patchify(x)

          if self.embed_strat == 'max':
            x = x.max(dim=-1).values
            x = x.unsqueeze(-1)
            x = x.float()
          elif self.embed_strat == 'sum':
            x = x.sum(dim=-1)
            x = x.unsqueeze(-1)
            x = x.float()
          elif self.embed_strat == 'mean':
            x = x.mean(dim=-1)
            x = x.unsqueeze(-1)
            x = x.float()

        elif self.embed_strat == 'learned_table':
          # Clip the tensor to the range [-1, 1]
          x = torch.clamp(x, min=-1, max=1)
          # Add 1 to x, then multiply by 1000
          x = x * 1000
          # Now x is in the range [0, 2000]
          # Convert to integers
          x = x % 2000
          x = x.long()
        elif self.embed_strat == 'naive_linear':
          x = x.float()
          x = x.unsqueeze(-1)    

        # embed tokens
        x = self.embed(x)

        if self.embed_strat == 'patch' or self.embed_strat == 'max' or self.embed_strat == 'sum' or self.embed_strat == 'mean':
          # reshape for transformer so that channels are passed independently
          x = rearrange(x, 'b c num_patch emb_dim -> (b c) num_patch emb_dim')
        elif self.embed_strat == "learned_table" or self.embed_strat == 'naive_linear':
        # apply positional encoding on last 2 dims
          x = rearrange(x, 'b seq_len c emb_dim -> (b c) seq_len emb_dim')

        if self.ema == True:
          # Applying EMA-like residual addition
          ema_x = torch.zeros_like(x[:, 0, :])
          for i in range(x.shape[1]):
              ema_x = self.decay * ema_x + (1 - self.decay) * x[:, i, :]
              x[:, i, :] = ema_x

        x = x + self.pe

        x = self.encoder(x)

        if self.embed_strat == 'patch' or self.embed_strat == 'max' or self.embed_strat == 'sum' or self.embed_strat == 'mean':
          x = rearrange(x, '(b c) num_patch emb_dim -> b c num_patch emb_dim', c=self.num_channels)
        else:
          x = rearrange(x, '(b c) seq_len emb_dim -> b c seq_len emb_dim', c=self.num_channels)

        return x
     
class PatchTSTDecoder(nn.Module):
    def __init__(self, num_patches, num_channels, embed_dim, target_seq_size, patch_len=8, dropout=0.0, embed_strat='patch'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.dropout = dropout
        self.target_seq_size = target_seq_size
        self.num_patches = num_patches
        self.embed_strat = embed_strat

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(int(embed_dim * num_patches), self.target_seq_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    def __init__(self, seq_len, num_channels, embed_dim, heads, depth, target_seq_size, patch_len=8, dropout=0.0, embed_strat='patch',
                 decay=0.9, ema=False):
        super().__init__()
        self.encoder = PatchTSTEncoder(seq_len, num_channels, embed_dim, heads, depth, patch_len, dropout, embed_strat)
        if embed_strat == "patch" or embed_strat == "max" or embed_strat == "sum" or embed_strat == "mean":
          self.decoder = PatchTSTDecoder(seq_len // patch_len, num_channels, embed_dim, target_seq_size, patch_len, dropout)
        else:
          self.decoder = PatchTSTDecoder(seq_len, num_channels, embed_dim, target_seq_size, patch_len, dropout)

        self.revIN = RevIN(num_channels, affine=True, subtract_last=False)

    def forward(self, x):
        # x shape is: (batch_size, seq_len, num_channels)
        x = self.revIN(x, 'norm')
        x = self.encoder(x)
        x = self.decoder(x)
        x = rearrange(x, 'b c s -> b s c')
        x = self.revIN(x, 'denorm')
        return x # shape: (batch_size, target_seq_size, num_channels)
