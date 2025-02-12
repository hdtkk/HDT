from torch import nn
from torch import autograd
from torch import linalg
import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
from cross_attn import Cross_AttentionLayer, Cross_FullAttention, Cross_Attention_Trans

# TODO : should we use ?
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


# class Encoder(nn.Module):
#     def __init__(self, dim_in, history_length):
#         super().__init__()
#         self.history_length = history_length
#         # self.network = nn.Sequential(
#         #     nn.Conv1d(
#         #         in_channels=dim_in, out_channels=128, kernel_size=4, stride=2, padding=1
#         #     ),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.1),
#         #     nn.LayerNorm(normalized_shape=[128, 12]),
#         #     nn.Conv1d(
#         #         in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
#         #     ),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.1),
#         #     nn.LayerNorm(normalized_shape=[64, 12]),
#         #     nn.Conv1d(
#         #         in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#         #     ),
#         #     nn.Tanh(),
#         # )
#
#         # self.network = nn.Sequential(
#         #     nn.Conv1d(
#         #             in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
#         #     ),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.1),
#         #     nn.LayerNorm(normalized_shape=[64, self.input_length // 2]),
#         #     # nn.LayerNorm(normalized_shape=[32, self.input_length // 4]),
#         #     nn.Conv1d(
#         #         in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#         #     ),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.1),
#         #     nn.LayerNorm(normalized_shape=[64, self.input_length // 2]),
#         #     nn.Conv1d(
#         #         in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#         #     ),
#         #     nn.Tanh(),
#         # )
#         self.func1= nn.Sequential(
#             nn.Conv1d(
#                 in_channels=dim_in, out_channels=64, kernel_size=4, stride=2, padding=1
#             ),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#         )
#
#         self.func2= nn.Sequential(
#             nn.Conv1d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#             ),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#         )
#
#         self.func3= nn.Sequential(
#             nn.Conv1d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#             ),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#         )
#         self.layernorm1 = nn.LayerNorm(64)
#         self.layernorm2 = nn.LayerNorm(64)
#         self.activation = nn.Tanh()
#
#
#     def forward(self, X):
#         X_1 = self.func1(X)
#         X_1 = self.layernorm1(X_1.permute(0, 2, 1)).permute(0, 2, 1)
#         X_2 = self.func2(X_1)
#         X_2 = self.layernorm2(X_2.permute(0, 2, 1)).permute(0, 2, 1)
#         X_3 = self.func3(X_2)
#         output = self.activation(X_3.permute(0, 2, 1)).permute(0, 2, 1)
#         return output

        # X_past = X[:, :, :self.history_length]
        # X_future = X[:, :, self.history_length:]
        #
        # X_past_paddings = F.pad(X_past, (1, 1), mode='constant', value=0.0)
        # X_future_paddings = F.pad(X_future, (1, 1), mode='constant', value=0.0)
        #
        # X_past_1 = self.func1(X_past_paddings)
        # X_future_1 = self.func1(X_future_paddings)
        # combine_1 = torch.cat((X_past_1, X_future_1), dim=-1)
        # X_1 = self.layernorm1(combine_1.permute(0, 2, 1)).permute(0, 2, 1)
        #
        # X_1_past_paddings = F.pad(X_1[:,:,:self.history_length // 2], (1, 1),  mode='constant', value=0.0)
        # X_1_future_paddings = F.pad(X_1[:, :, self.history_length // 2: ], (1, 1), mode='constant', value=0.0)
        #
        # X_past_2 = self.func2(X_1_past_paddings)
        # X_future_2 = self.func2(X_1_future_paddings)
        # combine_2 = torch.cat((X_past_2, X_future_2), dim=-1)
        # X_2 = self.layernorm2(combine_2.permute(0, 2, 1)).permute(0, 2, 1)
        #
        # X_2_past_paddings = F.pad(X_2[:, :, :self.history_length // 2], (1, 1), mode='constant', value=0.0)
        # X_2_future_paddings = F.pad(X_2[:, :, self.history_length // 2:], (1, 1), mode='constant', value=0.0)
        #
        # X_past_3 = self.func3(X_2_past_paddings)
        # X_future_3 = self.func3(X_2_future_paddings)
        #
        # outputs = torch.cat((X_past_3, X_future_3), dim=-1)
        # outputs = self.activation(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        # return outputs
        # return self.network(X)
class Encoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=dim_in, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(normalized_shape=[256, 24]),
            nn.Conv1d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(normalized_shape=[256, 24]),
            nn.Conv1d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.network(X)

# class Decoder(nn.Module):
#     def __init__(self, dim_out, history_length):
#         super().__init__()
#         self.history_length = history_length
#         # self.network = nn.Sequential(
#         #     nn.ConvTranspose1d(
#         #         in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#         #     ),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.1),
#         #     nn.LayerNorm(normalized_shape=[64, self.inputs_length // 2]),
#         #     nn.ConvTranspose1d(
#         #         in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
#         #     ),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.1),
#         #     nn.LayerNorm(normalized_shape=[32, self.inputs_length]),
#         #     nn.ConvTranspose1d(
#         #         in_channels=32,
#         #         out_channels=1,
#         #         kernel_size=3,
#         #         stride=1,
#         #         padding=1,
#         #     ),
#         # )
#         self.d_func1 = nn.Sequential(
#             nn.ConvTranspose1d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
#             ),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#         )
#         self.d_layernorm1 = nn.LayerNorm(64)
#         self.d_func2 = nn.Sequential(
#             nn.ConvTranspose1d(
#                 in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
#             ),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#         )
#         self.d_layernorm2 =nn.LayerNorm(64)
#
#         self.pred = nn.ConvTranspose1d(
#                 in_channels=64,
#                 out_channels=dim_out,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             )
#
#
#     def forward(self, H):
#
#         H_1 = self.d_func1(H)
#         H_1 = self.d_layernorm1(H_1.permute(0, 2, 1)).permute(0, 2, 1)
#         H_2 = self.d_func2(H_1)
#         H_2 = self.d_layernorm2(H_2.permute(0, 2, 1)).permute(0, 2, 1)
#         outputs = self.pred(H_2)
#         return outputs
#         # H_past = H[:, :, :self.history_length // 2]
#         # H_future = H[:, :, self.history_length // 2 :]
#         #
#         # # H_past_paddings = F.pad(H_past, (1, 1), mode='constant', value=0.0)
#         # # H_future_paddings = F.pad(H_future, (1, 1), mode='constant', value=0.0)
#         #
#         # H_past_1 = self.d_func1(H_past)
#         # H_future_1 = self.d_func1(H_future)
#         #
#         #
#         # combine_1 = torch.cat((H_past_1, H_future_1), dim=-1)
#         # H_1 = self.d_layernorm1(combine_1.permute(0, 2, 1)).permute(0, 2, 1)
#         #
#         #
#         # H_past_2 = self.d_func2(H_1[:, :, :self.history_length // 2])
#         # H_future_2 = self.d_func2(H_1[:, :, self.history_length // 2:])
#         # combine_2 = torch.cat((H_past_2, H_future_2), dim=-1)
#         # H_2 = self.d_layernorm2(combine_2.permute(0, 2, 1)).permute(0, 2, 1)
#         # # print('H2',H_2.shape)
#         #
#         # # H_2_past_paddings = F.pad(H_2[:, :, :self.history_length], (1, 1), mode='constant', value=0.0)
#         # # H_2_future_paddings = F.pad(H_2[:, :, self.history_length:], (1, 1), mode='constant', value=0.0)
#         #
#         # H_past_3 = self.pred(H_2[:, :, :self.history_length])
#         # H_future_3 = self.pred(H_2[:, :, self.history_length:])
#         #
#         # outputs = torch.cat((H_past_3, H_future_3), dim=-1)
#         #return self.network(H)

class Decoder(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(normalized_shape=[256, 24]),
            nn.ConvTranspose1d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(normalized_shape=[256, 24]),
            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=dim_out,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, H):
        return self.network(H)


class Discriminator(nn.Module):
    def __init__(self, target_dim, d_model):
        super().__init__()
        self.backbone = Encoder(dim_in=d_model)
        self.backbone.apply(weights_init)
        self.fc = nn.Sequential(nn.Linear(in_features=128, out_features=1), nn.Sigmoid())
        # self.fc = nn.Linear(in_features=64, out_features=1)  # removed sigmoid
        self.fc2 = nn.Linear(in_features=target_dim, out_features=d_model)

    def forward(self, X):
        X = self.fc2(X)   # B, L, target_dim -> B, L, dim_in
        # print('X.shape', X.shape)
        backbone_out = self.backbone(X.permute(0, 2, 1))
        # backbone_out = self.backbone(X.permute(0, 2, 1))
        # backbone_out = self.backbone(X.permute(0, 2, 1), X_past.permute(0, 2, 1))  # B， dim_in, L, -> B, 64, L/n
        # print('1', backbone_out.shape)
        backbone_out = backbone_out.transpose(1, 2)  # B * L/n * d
        output = self.fc(backbone_out)  # B * L/n * 1
        # print('output',output.shape)
        output = output.transpose(1, 2)  # B * 1 * L/n
        return output


class VQGANloss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1e-4, discriminator_weight=0.8) -> None:
        super().__init__()
        # TODO : gamma = 1e-4 ? C.f github de VQGAN
        self.reconstruction_loss = nn.MSELoss()
        self.discriminator_weight = discriminator_weight
        self.alpha = alpha
        self.gamma = gamma


    def calc_adaptive_weight(self, loss_rec, loss_d, last_layer):
        # VQGAN recommands to set lambda = 0 for at least 1 epoch
        # They set lambda to 0 in an initial warm-up phase
        # They found that longer warm-ups generally lead to better reconstructions
        rec_grads = autograd.grad(loss_rec, last_layer, retain_graph=True)[0]
        d_grads = autograd.grad(loss_d, last_layer, retain_graph=True)[0]

        weight = linalg.norm(rec_grads) / (linalg.norm(d_grads) + self.gamma)

        weight = torch.clamp(weight, 0.0, 1e4)

        return weight.detach()

    def forward(self, Xhat, X, Dhat=None, decoder=None, lmbda=None):


        # loss_rec = self.reconstruction_loss(Xhat, X)
        loss_rec = self.reconstruction_loss(Xhat, X)
        return loss_rec

        # loss_d = -Dhat.mean()
        #
        # decoder_last_layer = decoder.network[-1].weight
        #
        # if lmbda is None:
        #     lmbda = self.calc_adaptive_weight(loss_rec, loss_d, decoder_last_layer)
        #
        # loss = (
        #         loss_rec + self.discriminator_weight * lmbda * loss_d
        # )
        #
        # return loss, (loss_rec, loss_d, lmbda)
        # loss_d = -Dhat.mean()
        #
        # # decoder_last_layer_1 = decoder.pred_network.weight
        # decoder_last_layer_2 = decoder.pred_network.weight
        #
        #
        # if lmbda is None:
        #     lmbda = self.calc_adaptive_weight(loss_rec, loss_d, decoder_last_layer_2)
        #
        # loss = (
        #         loss_rec + self.discriminator_weight * lmbda * loss_d
        # )
        # return loss, (loss_rec, loss_d, lmbda)


class DiscriminatorLoss(nn.Module):
    def __init__(self, weight=0.8) -> None:
        super().__init__()
        self.weight = weight

    # def forward(self, Dhat, D):
    #     loss_D = torch.mean(F.relu(1.0 - D))
    #     loss_Dhat = torch.mean(F.relu(1.0 + Dhat))
    #     loss = self.weight * 0.5 * (loss_D + loss_Dhat)
    #     return loss, (loss_D, loss_Dhat)

    def forward(self, Dhat, D):
        # TODO : je ne comprends pas l'intérêt des max() vu que D et Dhat sont dans [0,1]
        max_D = torch.maximum(torch.zeros_like(D), 1 - D)
        max_Dhat = torch.maximum(torch.zeros_like(Dhat), 1 + Dhat)
        loss = self.weight * torch.mean(max_D + max_Dhat)
        return loss, (max_D, max_Dhat)


class RevIN(nn.Module):
    def __init__(self, num_features: int, context_length: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.context_length = context_length
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x[:, :self.context_length, :])
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


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

class New_FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(New_FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = 0
                # attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class New_AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(New_AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
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
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class AttentionLayer_Embed(nn.Module):
    def __init__(self, attention, n_heads):
        super(AttentionLayer_Embed, self).__init__()

        # d_keys = d_keys
        # d_values = d_values

        self.inner_attention = attention
        # self.query_projection = nn.Linear(input_size, d_keys * n_heads)
        # self.key_projection = nn.Linear(input_size, d_keys * n_heads)
        # self.value_projection = nn.Linear(input_size, d_values * n_heads)
        # self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        queries = keys = values = x
        # print('1', queries.shape)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        # queries = self.query_projection(queries).view(B, L, H, -1)
        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)
        # out = nn.AvgPool1d

        return out, attn


class AttentionLayer_Cross(nn.Module):
    def __init__(self, attention, n_heads, d_model):
        super(AttentionLayer_Cross, self).__init__()
        #
        # d_keys = d_keys
        # d_values = d_values
        self.d_model = d_model
        self.inner_attention = attention
        self.pooling = nn.AdaptiveAvgPool1d(self.d_model)
        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self,x, attn_mask=None, tau=None, delta=None):
        x = x.permute(0, 2, 1)  # b, k, l
        queries = keys = values = x
        # print('2', queries.shape)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)
        out = out.permute(0, 2, 1) # b, l, k
        out = self.pooling(out)
        return out, attn
        # return self.out_projection(out), attn



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class New_EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(New_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class New_Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(New_Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)

        return x, attns

class Transformer_EncModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff,  factor, pred_length,
                 dropout=0.1, output_attention=False, activation="relu", e_layers=3, target_dim=None):
        super(Transformer_EncModel, self).__init__()
        self.encoder = New_Encoder(
            [
                New_EncoderLayer(
                    New_AttentionLayer(
                        New_FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer= torch.nn.BatchNorm1d(d_model)
        )
        self.pos = PositionalEmbedding(d_model=d_model, max_len=5000)
        self.projection = nn.Sequential(nn.Linear(target_dim, d_model * 2, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(d_model * 2, d_model, bias=True)
                                        )
        # self.crossformer = Cross_Attention_Trans(pred_length=pred_length, factor=factor,
        #                                          d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        #                                          dropout=dropout)
        # self.crossformer_2 = Cross_Attention_Trans(pred_length=pred_length, factor=factor,
        #                                          d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        #                                          dropout=dropout)
    def forward(self, x_enc):
        x_enc = self.projection(x_enc)
        B, L, E = x_enc.shape
        base_shape = x_enc.shape
        pos_emb = self.pos(x_enc)   # 1, L, 1, E

        x_inputs = x_enc + pos_emb
        # x_inputs = x_inputs.reshape(B, E, K, L).permute(0, 2, 3, 1).reshape(B * K, L, E)
        enc_out, attns = self.encoder(x_inputs) # B * K, L, E
        enc_out = enc_out.reshape(B, L, E)
        # final_out = self.crossformer(enc_out)

        ##################
        # enc_out_traffic = enc_out[:, :, :274, :]
        # enc_out_eletricity = enc_out[:, : , 274:, :]
        # final_out_traffic = self.crossformer(enc_out_traffic)
        # final_out_eletricity = self.crossformer_2(enc_out_eletricity)
        # final_out = torch.cat((final_out_traffic, final_out_eletricity), dim=2)
        #####################
        # final_out = self.crossformer(enc_out)  # B, L, ts_d, d_model
        final_out = enc_out.reshape(B, E, L)
        # print(final_out.shape)
        return final_out, attns
        # return enc_out, attns


class Transformer_DecModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, pred_length,
                 dropout=0.1, output_attention=False, activation="relu", d_layers=3):
        super(Transformer_DecModel, self).__init__()
        self.encoder = New_Encoder(
            [
                New_EncoderLayer(
                    New_AttentionLayer(
                        New_FullAttention(False, factor, attention_dropout=dropout,
                                          output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(d_model)
        )
        self.pos = PositionalEmbedding(d_model=d_model, max_len=5000)
        # self.crossformer = Cross_Attention_Trans(pred_length=pred_length, factor=factor,
        #                                          d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        #                                          dropout=dropout)
        # self.crossformer_2 = Cross_Attention_Trans(pred_length=pred_length, factor=factor,
        #                                          d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        #                                          dropout=dropout)
        # self.pred_network = nn.Linear(d_model, target, bias=True)
        self.pred_network = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1
        )
        # self.pred_network =  nn.Conv1d(
        #         in_channels=d_model, out_channels=1, kernel_size=3, stride=1, padding=1
        #     )
        # self.pred_network_2 =  nn.Conv1d(
        #         in_channels=d_model, out_channels=target[1], kernel_size=3, stride=1, padding=1
        #     )
        nn.init.kaiming_normal_(self.pred_network.weight)
    def forward(self, x_enc, token=None):
        B, L, E = x_enc.shape
        # base_shape = x_enc.shape

        pos_emb = self.pos(x_enc)  # 1, L, 1, E

        x_inputs = x_enc + pos_emb
        # x_inputs = x_inputs.reshape(B, E, K, L).permute(0, 2, 3, 1).reshape(B * K, L, E)
        enc_out, attns = self.encoder(x_inputs)  # B * K, L, E
        enc_out = enc_out.reshape(B, E, L)
        # enc_out = self.mid_network(enc_out)
        # enc_out = F.relu(enc_out)
        final_out = self.pred_network(enc_out)
        #####################
        # enc_out_traffic = enc_out[:, :, :274, :]
        # enc_out_eletricity = enc_out[:, :, 274:, :]
        # final_out_traffic = self.crossformer(enc_out_traffic)
        # final_out_eletricity = self.crossformer_2(enc_out_eletricity)
        # out = torch.cat((final_out_traffic, final_out_eletricity), dim=2)
        ######################
        # out = self.crossformer(enc_out) # B, L, ts_d, d_model

        # out = enc_out.permute(0, 2, 1, 3).reshape(B, K, L * E)
        # if token == 'ele':
        #     final_out = self.pred_network_2(enc_out)
        # else:
        #     final_out = self.pred_network(enc_out)

        final_out = final_out.permute(0, 2, 1)

        return final_out, attns
        # return enc_out, attns

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x



