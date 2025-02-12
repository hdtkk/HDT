import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from math import sqrt
import numpy as np
import random

class TriangularCausalMask():
    def __init__(self, B, L, device):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask




class CompressionEmbedding(nn.Module):
    def __init__(self, d_model):
        super(CompressionEmbedding, self).__init__()

        self.past_embed = d_model
        # self.d_model = d_model

        self.network_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.past_embed, out_channels=self.past_embed , kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1),
        )

        self.layernorm1 =  nn.LayerNorm(self.past_embed)

        self.network_2 =  nn.Sequential(
            nn.Conv1d(
                in_channels=self.past_embed, out_channels=self.past_embed , kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1),
        )
        self.layernorm2 = nn.LayerNorm(self.past_embed)

        self.network_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.past_embed, out_channels=self.past_embed, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
            nn.Dropout(p=0.1),
        )


    def forward(self, x):
        x = self.network_1(x)
        x = self.layernorm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.network_2(x)
        x = self.layernorm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.network_3(x)

        return x



class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.d_model  = d_model

        self.embed_layer = nn.ModuleList()
        kernels = [3, 5, 7, 9]
        paddings = [1, 2, 3, 4]
        for _ in range(16):
            for k, p in zip(kernels, paddings):
                conv = nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=p)
                # print('cov', conv.weight)
                self.embed_layer.append(conv)
        #


    def forward(self, x):

        # x = x.unsqueeze(1)
        emb = []
        for i in range(self.d_model):
            emb.append(self.embed_layer[i](x))
            # print('emb', emb[i])
        x = torch.cat(emb, dim=1)

        return x



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Past_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Past_PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe_past', pe)

    def forward(self, x):
        return self.pe_past[:, :x.size(1)]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, init_embed, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.d_model = d_model
        self.init_embed = init_embed
        self.value_embedding = TokenEmbedding(c_in=c_in, init_embed=init_embed, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=init_embed)
        self.position_embedding = nn.Parameter(torch.zeros(1, 512, init_embed))
        self.compression_embedding = CompressionEmbedding(init_embed=init_embed, d_model=d_model, history_length=192)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.value_embedding(x) # (B, d_model, 96)
        x = x.permute(0, 2, 1)
        position_embed = self.position_embedding[:, :x.shape[1], :]
        x = x + position_embed
        x = self.compression_embedding(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)
        return x
        # return self.dropout(x)



class PastEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PastEmbedding, self).__init__()
        self.d_model = d_model
        self.value_embedding = TokenEmbedding(d_model=d_model)

        self.pos_embed = Past_PositionalEmbedding(d_model=d_model)
        self.compression_embedding = CompressionEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) # (B, d_model, 96)
        x = x.permute(0, 2, 1)
        position_embed = self.pos_embed(x)
        x = x + position_embed
        x = self.compression_embedding(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
        # return self.dropout(x)






class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer


    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU()
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.mov_attention = mov_attention

    def forward(self, x, cross=None, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Final_DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", mov_attention=None):
        super(Final_DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU()
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.mov_attention = mov_attention
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x, cross=None, mov_indices=None, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x = self.norm4(x)
        x =  x + self.dropout(self.mov_attention(
            x, mov_indices, mov_indices,
            attn_mask=cross_mask
        )[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):


        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x



class Final_Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Final_Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, mov_embed, x_mask=None, cross_mask=None):

        for layer in self.layers:
            x = layer(x, cross, mov_embed, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x





class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries,   keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, queries.device)
                # attn_mask = attn_mask.to(queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class Past_transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, pred_len, enc_in, d_model, c_out, dropout=0.1, activation='gelu',
                 nheads=8, e_layers=3, d_layers=1, init_embed = 32, output_attention=None):
        super(Past_transformer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = nheads
        self.activation = activation
        self.e_layers =e_layers
        self.d_layers = d_layers
        self.c_out = c_out
        self.d_ff = 4 * self.d_model
        self.init_embed = init_embed
        # Embedding
        # self.enc_embedding = DataEmbedding(enc_in, self.init_embed, self.d_model, self.dropout)

        # self.enc_input_proj = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.ReLU(),
        # )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False,scale=None, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=True, scale=None, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        FullAttention(mask_flag=False, scale=None, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    #
    def forward(self, x_dec, x_enc,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)
        dec_out = self.decoder(x_dec, cross=enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)


        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]





class Final_transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, pred_len, enc_in, d_model, c_out, dropout=0.1, activation='gelu',
                 nheads=8, e_layers=3, d_layers=1, init_embed = 32, output_attention=None):
        super(Final_transformer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = nheads
        self.activation = activation
        self.e_layers =e_layers
        self.d_layers = d_layers
        self.c_out = c_out
        self.d_ff = 4 * self.d_model
        self.init_embed = init_embed
        # Embedding
        # self.enc_embedding = DataEmbedding(enc_in, self.init_embed, self.d_model, self.dropout)

        # self.enc_input_proj = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=1, padding=1
        #     ),
        #     nn.ReLU(),
        # )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False,scale=None, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )


        self.decoder = Final_Decoder(
            [
                Final_DecoderLayer(
                    self_attention=AttentionLayer(
                        FullAttention(mask_flag=True, scale=None, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    cross_attention=AttentionLayer(
                        FullAttention(mask_flag=False, scale=None, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    mov_attention=AttentionLayer(
                            FullAttention(mask_flag=False, scale=None, attention_dropout=self.dropout,
                                          output_attention=False),
                            self.d_model, self.n_heads),
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )


    #
    def forward(self, x_dec, x_enc, mov_embed,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc = x_enc.permute(0, 2, 1)

        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)
        dec_out = self.decoder(x_dec, cross=enc_out, mov_embed=mov_embed, x_mask=dec_self_mask, cross_mask=dec_enc_mask)


        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
