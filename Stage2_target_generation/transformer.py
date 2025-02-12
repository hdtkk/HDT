import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT, Final_GPT
import numpy as np
# from vqgan import VQGAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

class VQGANTransformer(nn.Module):
    def __init__(self, encoder, input_encoder, decoder, codebook, quant_conv,
                 post_quant_conv, num_codebook, pkeep, target_dim, e_layers, d_layers, init_embed, sos_token: int=0,):
        super(VQGANTransformer, self).__init__()

        self.sos_token = num_codebook


        self.input_encoder = input_encoder
        for params in self.input_encoder.parameters():
            params.requires_grad = False
        self.input_encoder.eval()

        self.encoder = encoder
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.encoder.eval()
        self.decoder = decoder
        for params in self.decoder.parameters():
            params.requires_grad = False
        self.decoder.eval()

        self.codebook = codebook
        for params in self.codebook.parameters():
            params.requires_grad = False
        self.codebook.eval()

        self.quant_conv = quant_conv
        for params in self.quant_conv.parameters():
            params.requires_grad = False
        self.quant_conv.eval()

        self.post_quant_conv = post_quant_conv
        for params in self.post_quant_conv.parameters():
            params.requires_grad = False
        self.post_quant_conv.eval()

        self.num_codebook = num_codebook
        self.gamma = self.gamma_func(mode='square')
        transformer_config = {
            "vocab_size": self.num_codebook,
            "block_size": 512,
            "n_layer": 3,
            "n_head": 8,
            "n_embed": 256,

        }
        self.block_size = 512
        self.init_embed = init_embed
        self.codebook1 = self.codebook.embedding.weight.data.clone()
        # self.transformer = GPT(**transformer_config)
        self.transformer = GPT(vocab_size=self.num_codebook, block_size=self.block_size, enc_in=target_dim,
                               pred_len=24, codebook_embed=self.codebook1, e_layers=e_layers,
                               d_layers=d_layers, n_head=transformer_config['n_head'], init_embed = init_embed,
                               n_embd=transformer_config['n_embed'],  embd_pdrop=0.1, resid_pdrop=0.,
                               attn_pdrop=0., n_unmasked=0, past_embed=self.init_embed)

        # if not training:
        # for params in self.transformer.parameters():
        #     params.requires_grad = False
        # self.transformer.eval()

        self.pkeep = pkeep

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    @torch.no_grad()
    def encode_to_z(self, x):
        encoded_inputs, _ = self.encoder(x.permute(0, 2, 1))
        encoded_inputs = self.input_encoder(encoded_inputs)
        quant_enc_out = self.quant_conv(encoded_inputs)

        quant_z, indices, _ = self.codebook(quant_enc_out)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices, quant_enc_out


    @torch.no_grad()
    def encode_past(self, x):
        encoded_inputs, _ = self.encoder(x.permute(0, 2, 1))
        encoded_inputs = self.input_encoder(encoded_inputs)
        quant_enc_out = self.quant_conv(encoded_inputs)

        quant_z, indices, _ = self.codebook(quant_enc_out)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices, quant_enc_out




    @torch.no_grad()
    def z_to_image(self, indices, sampling_shape):
        ix_to_vectors = self.codebook.embedding(indices).reshape(sampling_shape)
        ix_to_vectors = self.post_quant_conv(ix_to_vectors)
        targets = self.decoder(ix_to_vectors)
        targets = targets.permute(0, 2, 1)
        return targets


    def token_embed(self, x_past):
        x_past = x_past.unsqueeze(1)
        emb = []
        for i in range(self.init_embed):
            emb.append(self.embed_model[i](x_past))
            # print('emb', emb[i])
        x_past = torch.cat(emb, dim=1)
        return x_past





    def forward(self, x, x_past, x_past_past):
        _, indices, _ = self.encode_to_z(x)  # indices Batch, 2 (64, 2)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(indices.device)
        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  # torch.bernoulli([0.5 ... 0.5]) -> [1, 0, 1, 1, 0, 0] ; p(1) - 0.5
        mask = mask.round().to(dtype=torch.int64)

        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1], x_past, x_past_past)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, x_past, x_past_past, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x, x_past, x_past_past)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def sample_moving(self, x_past, x_past_past):
        sos_tokens = torch.ones(x_past.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x_past.device)
        indices = torch.zeros(sos_tokens.shape).long().to(x_past.device)
        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, x_past, x_past_past, steps=24)

        return sample_indices

    @torch.no_grad()
    def loglog_series(self, x_past, sampling_shape):

        sos_tokens = torch.ones(x_past.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        indices = torch.zeros(sos_tokens.shape).long().to("cuda")
        start_indices = indices[:, :0]
        print(start_indices.shape)
        sample_indices = self.sample(start_indices, sos_tokens, x_past, steps=12)

        targets = self.z_to_image(indices=sample_indices, sampling_shape=sampling_shape)

        print(targets.shape)
        return targets



    @torch.no_grad()
    def log_images(self, x, x_past, x_past_past):
        # log = dict()
        # sampling_shape = [64, 64, 12]
        _, indices, _ = self.encode_to_z(x)
        print('indice', indices[:20, :])
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x.device)

        # start_indices = indices[:, :indices.shape[1] // 2]
        # sample_indices = self.sample(start_indices, sos_tokens, x_past, steps=indices.shape[1] - start_indices.shape[1])
        #
        # half_sample = self.z_to_image(indices=sample_indices, sampling_shape=sampling_shape)
        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, x_past, x_past_past, steps=indices.shape[1])
        print('full_indice', sample_indices[:20, :])
        dsad

        return half_sample, full_sample, target_sample
        # return log, torch.concat((x, x_rec, half_sample, full_sample))





class Target_VQGANTransformer(nn.Module):
    def __init__(self, encoder, input_encoder, decoder, codebook, quant_conv,
                 post_quant_conv, num_codebook, pkeep, target_dim, e_layers, d_layers, init_embed, pred_length):
        super(Target_VQGANTransformer, self).__init__()

        self.sos_token = num_codebook
        self.pred_length = pred_length
        # self.mask_token_id = num_codebook

        self.input_encoder = input_encoder
        for params in self.input_encoder.parameters():
            params.requires_grad = False
        self.input_encoder.eval()

        self.encoder = encoder
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.encoder.eval()
        self.decoder = decoder
        for params in self.decoder.parameters():
            params.requires_grad = False
        self.decoder.eval()

        # self.embed_model = embed_model
        # for params in self.embed_model.parameters():
        #     params.requires_grad = False
        # self.embed_model.eval()

        self.codebook = codebook
        for params in self.codebook.parameters():
            params.requires_grad = False
        self.codebook.eval()

        self.quant_conv = quant_conv
        for params in self.quant_conv.parameters():
            params.requires_grad = False
        self.quant_conv.eval()
        self.post_quant_conv = post_quant_conv
        for params in self.post_quant_conv.parameters():
            params.requires_grad = False
        self.post_quant_conv.eval()
        self.num_codebook = num_codebook
        # self.vqgan = self.load_vqgan(args)
        self.gamma = self.gamma_func(mode='square')
        transformer_config = {
            "vocab_size": self.num_codebook,
            "block_size": 512,
            "n_layer": 3,
            "n_head": 8,
            "n_embed": 256,

        }
        self.block_size = 512
        self.init_embed = init_embed
        self.codebook1 = self.codebook.embedding.weight.data.clone()
        # self.transformer = GPT(**transformer_config)
        self.fine_tuning = nn.Sequential(
            nn.Conv1d(
                in_channels=transformer_config['n_embed'], out_channels=transformer_config['n_embed'], kernel_size=3,
                stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(
                in_channels=transformer_config['n_embed'], out_channels=transformer_config['n_embed'], kernel_size=3,
                stride=1, padding=1
            ),
        )
        for params in self.fine_tuning.parameters():
            params.requires_grad = False
        self.fine_tuning.eval()

        # self.transformer = GPT(vocab_size=self.num_codebook, block_size=self.block_size, enc_in=target_dim,
        #                        pred_len=self.pred_length, codebook_embed=self.codebook1, e_layers=e_layers,
        #                        d_layers=d_layers, n_head=transformer_config['n_head'], init_embed = init_embed,
        #                        n_embd=transformer_config['n_embed'],  embd_pdrop=0.1, resid_pdrop=0.,
        #                        attn_pdrop=0., n_unmasked=0, past_embed=self.init_embed, ismoving_embed=True, istarget=True)

        self.transformer = Final_GPT(vocab_size=self.num_codebook, block_size=self.block_size, enc_in=target_dim,
                               pred_len=48, codebook_embed=self.codebook1, e_layers=e_layers,
                               d_layers=d_layers, n_head=transformer_config['n_head'], init_embed = init_embed,
                               n_embd=transformer_config['n_embed'],  embd_pdrop=0.1, resid_pdrop=0.,
                               attn_pdrop=0., n_unmasked=0, past_embed=self.init_embed)
        #
        # self.transformer.eval()
        for params in self.transformer.parameters():
            params.requires_grad = False
        self.transformer.eval()
        self.pkeep = pkeep

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    @torch.no_grad()
    def encode_to_z(self, x):
        encoded_inputs, _ = self.encoder(x.permute(0, 2, 1))
        encoded_inputs = self.input_encoder(encoded_inputs)
        quant_enc_out = self.quant_conv(encoded_inputs)

        quant_z, indices, _ = self.codebook(quant_enc_out)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices, quant_enc_out



    @torch.no_grad()
    def z_to_image(self, indices, sampling_shape):
        ix_to_vectors = self.codebook.embedding(indices).reshape(sampling_shape)
        ix_to_vectors = self.post_quant_conv(ix_to_vectors)
        targets = self.decoder(ix_to_vectors)
        targets = targets.permute(0, 2, 1)
        return targets

    def z_to_series(self, indices, sampling_shape):
        if sampling_shape is not None:
            ix_to_vectors = self.codebook.embedding(indices).reshape(sampling_shape)
        else:
            ix_to_vectors = self.codebook.embedding(indices).permute(0, 2, 1)
        ix_to_vectors = self.post_quant_conv(ix_to_vectors)
        ix_to_vectors = self.fine_tuning(ix_to_vectors)
        targets = self.decoder(ix_to_vectors)
        targets = targets.permute(0, 2, 1)
        # print('target',targets.shape)
        return targets


    def forward(self, x, x_past, x_past_past, mov_indice):
        _, indices, _ = self.encode_to_z(x)  # indices Batch, 2 (64, 2)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(indices.device)
        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  # torch.bernoulli([0.5 ... 0.5]) -> [1, 0, 1, 1, 0, 0] ; p(1) - 0.5
        mask = mask.round().to(dtype=torch.int64)

        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1], x_past, x_past_past, mov_indice)
        # print('logits', logits.shape)
        # print('target', target.shape)
        return logits, target



    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, x_past, x_past_past, steps, temperature=5.0, top_k=100, moving_indice=None):
        self.transformer.eval()

        x = torch.cat((c, x), dim=1)

        for k in range(steps):
            # x_embed = self.codebook.embedding(x)
            logits, _ = self.transformer(x, x_past, x_past_past, moving_indice)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        x = x[:, c.shape[1]:]
        # self.transformer.train()
        return x

    @torch.no_grad()
    def log_series(self, x_past, sampling_shape):

        sos_tokens = torch.ones(x_past.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        indices = torch.zeros(sos_tokens.shape).long().to("cuda")
        start_indices = indices[:, :0]
        print(start_indices.shape)
        sample_indices = self.sample(start_indices, sos_tokens, x_past, steps=12)

        targets = self.z_to_image(indices=sample_indices, sampling_shape=sampling_shape)

        print(targets.shape)
        return targets


    def sample_final(self, x_past, x_past_past, moving_indice, sampling_shape=None):

        sos_tokens = torch.ones(x_past.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x_past.device)

        indices = torch.zeros(sos_tokens.shape).long().to(x_past.device)
        start_indices = indices[:, :0]


        sample_indices = self.sample(start_indices, sos_tokens, x_past, x_past_past, steps=48,
                                     moving_indice=moving_indice)

        target = self.z_to_series(indices=sample_indices, sampling_shape=sampling_shape)
        return target


    @torch.no_grad()
    def log_images(self, x, x_past, x_past_past, moving_indice=None):
        # log = dict()
        # sampling_shape = [64, 64, 12]
        _, indices, _ = self.encode_to_z(x)
        print('indice', indices[:20, :])
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x.device)

        # start_indices = indices[:, :indices.shape[1] // 2]
        # sample_indices = self.sample(start_indices, sos_tokens, x_past, steps=indices.shape[1] - start_indices.shape[1])
        #
        # half_sample = self.z_to_image(indices=sample_indices, sampling_shape=sampling_shape)
        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, x_past, x_past_past, steps=indices.shape[1], moving_indice=moving_indice)
        print('full_indice', sample_indices[:20, :])
        dsad

        return half_sample, full_sample, target_sample
        # return log, torch.concat((x, x_rec, half_sample, full_sample))
