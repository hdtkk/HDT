"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import random

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from past_trans import Past_transformer, Final_transformer

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present  # TODO: check that this does not break anything

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]





class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present:
            assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, vocab_size, block_size, enc_in, pred_len=24, codebook_embed=None, e_layers=3,
                 d_layers=3, n_head=8, n_embd=32, init_embed = 32,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, past_embed=None,
                 ):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           codebook_embed=codebook_embed, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.pred_len = pred_len
        # input embedding stem
        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1
            )
            )
        self.past_transformer = Past_transformer(pred_len=pred_len, enc_in=enc_in,
                                                 d_model=n_embd, c_out=n_embd, e_layers=e_layers,
                         d_layers=d_layers, init_embed = init_embed, output_attention=None)

        self.tok_emb = nn.Embedding(config.vocab_size + 1, n_embd)
        self.tok_emb.weight.data.uniform_(-1.0 / (config.vocab_size + 1), 1.0 / (config.vocab_size + 1))

        self.past_tok_emb = nn.Embedding(config.vocab_size, n_embd)
        self.past_tok_emb.weight.data.uniform_(-1.0 / config.vocab_size, 1.0 / config.vocab_size)

        self.past_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, n_embd))

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, n_embd))  # 512 x 1024

        self.drop = nn.Dropout(config.embd_pdrop)

        self.ln_f = nn.LayerNorm(n_embd)


        self.head = nn.Sequential(nn.Linear(n_embd, n_embd),
                                  nn.ReLU(),
                                  nn.Linear(n_embd, config.vocab_size, bias=True)
                                  )

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def forward(self, idx, x_past, embeddings=None):
    def forward(self, idx, x_past, x_past_past):
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector  # (64, 12, 64)
        # print('token', token_embeddings.shape)
        x_past_new = torch.cat((x_past_past, x_past), dim=-1)
        past_token_embeddings = self.past_tok_emb(x_past_new)


        t = token_embeddings.shape[1]
        past_t = past_token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # position_embeddings = self.pos_emb(all_embeddings)
        position_embeddings = self.pos_emb[:, :t, :]
        past_position_embeddings = self.past_pos_emb[:, :past_t, :]
        x_past = self.drop(past_token_embeddings + past_position_embeddings)
        # x_past = self.projection(torch.cat((x_past.permute(0, 2, 1), x_past_past), dim=1)).permute(0, 2 , 1)
        x = self.drop(token_embeddings + position_embeddings)
        x = self.past_transformer(x_dec=x, x_enc=x_past)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None
        # return logits,


class Final_GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, vocab_size, block_size, enc_in, pred_len=24, codebook_embed=None, e_layers=3,
                 d_layers=3, n_head=8, n_embd=32, init_embed = 32,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, past_embed=None,
                 ):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           codebook_embed=codebook_embed, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.pred_len = pred_len
        # input embedding stem
        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=n_embd, out_channels=n_embd, kernel_size=3, stride=1, padding=1
            )
            )

        self.final_transformer = Final_transformer(pred_len=pred_len, enc_in=enc_in,
                                                 d_model=n_embd, c_out=n_embd, e_layers=e_layers,
                         d_layers=d_layers, init_embed = init_embed, output_attention=None)


        self.tok_emb = nn.Embedding(config.vocab_size + 1, n_embd)
        self.tok_emb.weight.data.uniform_(-1.0 / (config.vocab_size + 1), 1.0 / (config.vocab_size + 1))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, n_embd))  # 512 x 1024

        self.past_tok_emb = nn.Embedding(config.vocab_size, n_embd)
        self.past_tok_emb.weight.data.uniform_(-1.0 / config.vocab_size, 1.0 / config.vocab_size)
        self.past_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, n_embd))

        self.mov_token_emb = nn.Embedding(config.vocab_size + 1, n_embd)
        self.mov_token_emb.weight.data.uniform_(-1.0 / (config.vocab_size + 1), 1.0 / (config.vocab_size + 1))
        self.mov_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)

        self.ln_f = nn.LayerNorm(n_embd)


        self.head = nn.Sequential(nn.Linear(n_embd, n_embd),
                                  nn.ReLU(),
                                  nn.Linear(n_embd, config.vocab_size, bias=True)
                                  )

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def forward(self, idx, x_past, embeddings=None):
    def forward(self, idx, x_past, x_past_past, mov_indces):
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector  # (64, 12, 64)


        x_past_new = torch.cat((x_past_past, x_past), dim=-1)
        past_token_embeddings = self.past_tok_emb(x_past_new)
        mov_token_embeddings = self.mov_token_emb(mov_indces)


        t = token_embeddings.shape[1]
        past_t = past_token_embeddings.shape[1]
        mov_t = mov_token_embeddings.shape[1]

        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # position_embeddings = self.pos_emb(all_embeddings)
        position_embeddings = self.pos_emb[:, :t, :]
        past_position_embeddings = self.past_pos_emb[:, :past_t, :]
        mov_position_embeddings = self.mov_pos_emb[:, :mov_t, :]

        x_past = self.drop(past_token_embeddings + past_position_embeddings)
        # x_past = self.projection(torch.cat((x_past.permute(0, 2, 1), x_past_past), dim=1)).permute(0, 2 , 1)
        x = self.drop(token_embeddings + position_embeddings)

        mov_x = self.drop(mov_token_embeddings + mov_position_embeddings)

        x = self.final_transformer(x_dec=x, x_enc=x_past, mov_embed=mov_x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None
        # return logits,

