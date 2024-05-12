from typing import ParamSpecKwargs
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from x_transformers import Encoder
from models.revIN import RevIN
from torch.fft import rfftn, fftshift
import torch.nn.functional as F

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
    def __init__(self, input_size, embed_dim, mode='linear'):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.input_size = input_size

        self.embed = nn.Sequential(
            nn.Linear(input_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        if mode == 'cnn':
          self.embed = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        elif mode == 'fft':
            # Parameters for STFT
            self.win_length = input_size  # window length for STFT
            self.hop_length = self.win_length // 8  # stride; set it so that seq_len is reduced by 8




    def forward(self, x):
        x = x.float()

        if self.mode == 'cnn':
          batch_size, num_channels, seq_len, patch_len = x.shape
          # Using einops to rearrange the input
          x = rearrange(x, 'b c s p -> (b c s) 1 p')
          # Apply the 1D CNN
          x = self.embed(x)
          x = rearrange(x, '(b c s) e p -> b c s (e p)', b=batch_size, c=num_channels, s=seq_len)
          return x

        elif self.mode == 'fft':
          x = rearrange(x, 'b c s p -> b c (s p)')
          # Assuming x is of shape (batch_size, channels, seq_len)
          # Apply STFT
          x = rfftn(x, dim=(-1,))  # Perform FFT on the last dimension
          x = fftshift(x, dim=-1)  # Center the frequencies
          # Take the magnitude (absolute value) of the complex output
          x = x.abs()
          # Reshape to original batch and channel dimensions, adjust seq_len and set embed_dim
          # Optionally, linearly project to a specific embed_dim if different from output of rfftn
          if x.shape[-1] != self.embed_dim:
              projection = nn.Linear(x.shape[-1], self.embed_dim).to(x.device)
              x = projection(x)
          return x


        return self.embed(x)

class PatchTSTEncoder(nn.Module):
    def __init__(self, seq_len,  num_channels, embed_dim, heads, depth, patch_len=8, dropout=0.0, embed_strat='patch', 
                 decay=0.9, ema=False, residual=False, embed_mode = 'linear'):
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
        self.residual = residual
        self.embed_mode = embed_mode

        if self.embed_strat == 'patch':
          # learnable embeddings for each channel
          self.embed = Embedding(patch_len, embed_dim, mode=self.embed_mode)
          # learnable positional encoding
          self.pe = nn.Parameter(torch.randn(1, (seq_len // patch_len), embed_dim))

        elif self.embed_strat == 'learned_table':
          self.embed = nn.Embedding(num_embeddings=2001, embedding_dim=embed_dim)
          self.pe = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        elif self.embed_strat == 'naive_linear':
          self.embed = nn.Linear(1, embed_dim)
          self.pe = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        elif self.embed_strat == 'patch++':
          self.embed = Embedding(patch_len, embed_dim, mode=self.embed_mode)
          self.pe = nn.Parameter(torch.randn(1, (seq_len // patch_len), embed_dim))
          # add another linear layer that takes the entire sequence as input and outputs custom positional encoding
          self.pe2 = nn.Linear(seq_len, seq_len)

        else:
          self.embed = nn.Linear(1, embed_dim)
          self.pe = nn.Parameter(torch.randn(1, (seq_len // patch_len), embed_dim))


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
        elif self.embed_strat == 'patch++':
          pe_x = self.pe2(rearrange(x, 'b s c -> b c s').float())
          pe_x = rearrange(pe_x, 'b c (s patch_len) -> b c s patch_len', patch_len=self.patch_len)
          x = self.patchify(x)
          x += pe_x
        else:
           pass
        # embed tokens
        x = self.embed(x)

        if self.embed_strat == 'patch' or self.embed_strat == 'max' or self.embed_strat == 'sum' or self.embed_strat == 'mean' or self.embed_strat == 'patch++':
          # reshape for transformer so that channels are passed independently
          x = rearrange(x, 'b c num_patch emb_dim -> (b c) num_patch emb_dim')
        elif self.embed_strat == "learned_table" or self.embed_strat == 'naive_linear':
        # apply positional encoding on last 2 dims
          x = rearrange(x, 'b seq_len c emb_dim -> (b c) seq_len emb_dim')

        if self.ema == True:
          alpha = 0.1  # Smoothing factor, adjust as needed
          # Initialize EMA tensor, same shape as x, starting with the first token's values
          ema = torch.zeros_like(x)
          ema[:, 0, :] = x[:, 0, :]  # Start EMA with the first token for each batch and feature

          # Calculate EMA across each token for each batch and feature
          for i in range(1, x.shape[1]):  # Start from second token as first is already initialized
              ema[:, i, :] = alpha * x[:, i, :] + (1 - alpha) * ema[:, i - 1, :]

          # Add EMA to original tokens
          x += ema


        
        if self.residual == True:
          padded_x = F.pad(x, (0, 0, 1, 1), "constant", 0)  # No padding for features, pad tokens dimension

          # Slicing to get 'before' and 'after' tensors
          before = padded_x[:, :-2, :]  
          after = padded_x[:, 2:, :]    
          # Calculate the residuals
          residual_tokens = 0.1 * before + 0.8 * x + 0.1 * after  # All have shape [110, 12, 256]
          x = residual_tokens



        x = x + self.pe

        x = self.encoder(x)

        if self.embed_strat == 'patch' or self.embed_strat == 'max' or self.embed_strat == 'sum' or self.embed_strat == 'mean' or self.embed_strat == 'patch++':
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
                 decay=0.9, ema=False, residual=False, embed_mode='linear'):
        super().__init__()
        self.encoder = PatchTSTEncoder(seq_len, num_channels, embed_dim, heads, depth, patch_len, dropout, 
                                       embed_strat, residual=residual, embed_mode=embed_mode)

        if embed_strat == "patch" or embed_strat == "max" or embed_strat == "sum" or embed_strat == "mean" or embed_strat == "patch++":
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
