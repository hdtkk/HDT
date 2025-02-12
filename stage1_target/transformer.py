import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
import numpy as np
# from vqgan import VQGAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

class VQGANTransformer(nn.Module):
    def __init__(self, embed_model, encoder, input_encoder, decoder, codebook, quant_conv,
                 post_quant_conv, num_codebook, pkeep, target_dim, e_layers, d_layers, init_embed, sos_token: int=0,):
        super(VQGANTransformer, self).__init__()

        self.sos_token = num_codebook + 1

        self.mask_token_id = num_codebook

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

        self.embed_model = embed_model
        for params in self.embed_model.parameters():
            params.requires_grad = False
        self.embed_model.eval()

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
            "n_embed": 128,

        }
        self.block_size = 512
        self.init_embed = init_embed
        self.codebook1 = self.codebook.embedding.weight.data.clone()
        # self.transformer = GPT(**transformer_config)
        self.transformer = GPT(vocab_size=self.num_codebook, block_size=self.block_size, enc_in=target_dim,
                               pred_len=12, codebook_embed=self.codebook1, e_layers=e_layers,
                               d_layers=d_layers, n_head=transformer_config['n_head'], init_embed = init_embed,
                               n_embd=transformer_config['n_embed'],  embd_pdrop=0.1, resid_pdrop=0.,
                               attn_pdrop=0., n_unmasked=0, past_embed=self.init_embed)
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
    def z_to_image(self, indices, sampling_shape):
        ix_to_vectors = self.codebook.embedding(indices).reshape(sampling_shape)
        ix_to_vectors = self.post_quant_conv(ix_to_vectors)
        targets = self.decoder(ix_to_vectors)
        targets = targets.permute(0, 2, 1)
        return targets
        # image = self.vqgan.decode(ix_to_vectors)
        # return image

    def token_embed(self, x_past):
        x_past = x_past.unsqueeze(1)
        emb = []
        for i in range(self.init_embed):
            emb.append(self.embed_model[i](x_past))
            # print('emb', emb[i])
        x_past = torch.cat(emb, dim=1)
        return x_past

    def draw_image(self, x, discriminator_x):
        _, idx = self.encode_to_z(x)
        xx = discriminator_x.cpu().numpy()
        x_indice = idx.cpu().numpy()
        # past_indice = past_indice.cpu().numpy()

        # Plotting the time series data
        # plt.figure(figsize=(15, 8))
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        for i in range(10):
            ax1.plot(xx[i][-24:], label=f'Series {i + 1}')

        fig1.savefig('series' + str(i) + '.png')
        for i in range(10):
            ax2.plot(x_indice[i][-12:], label=f'indice_Series {i + 1}')
        fig2.savefig('indice' + str(i) + '.png')

        #
        # print('x', xx.shape)
        # print('y', x_indice.shape)
        # y1 = x
        # y2 = y
        # label_x = np.linspace(1, 144, 144)
        # label_y = np.linspace(1, 12, 12)
        # for i in range(15):
        #     fig1, ax1 = plt.subplots()
        #     ax1.plot(label_x, y1[i])
        #     ax1.legend()
        #     plt.savefig('X_past_Test_64'+str(i)+'.png')
        # for i in range(15):
        #     fig1, ax1 = plt.subplots()
        #     ax1.plot(label_y, y2[i])
        #     ax1.legend()
        #     plt.savefig('future_Test_64'+str(i)+'.png')
    def draw_codebook(self, idx):
        indices = torch.arange(0, 32).to("cuda:1")
        x_indice = self.codebook.embedding(indices).cpu().numpy()
        tsne = TSNE(n_components=2, random_state=0, perplexity=3)
        codebook_2d = tsne.fit_transform(x_indice)
        fig1, ax1 = plt.subplots()
        ax1.scatter(codebook_2d[:, 0], codebook_2d[:, 1], marker='o')
        for i, (x, y) in enumerate(codebook_2d):
            ax1.text(x, y, str(i + 1), fontsize=9)  # 编号从1开始
        ax1.set_title("Codebook Embeddings Visualization")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        fig1.savefig('codebook_3' +  '.png')

    # def distribution(self, idx):
    #     # print(idx[:10, :])
    #     # das
    #     # print(self.codebook.embedding.weight.shape)
    #     # # dsad
    #     from mpl_toolkits.mplot3d import Axes3D
    #     indices = torch.arange(0, 32).to("cuda:1")
    #
    #     x_indice = self.codebook.embedding(indices).cpu().numpy()
    #
    #     # embeddings = self.codebook.embedding(idx)
    #     # embeddings = x_indice.reshape(-1, 64)
    #     # # print(embeddings.shape)
    #     # embeddings = embeddings.cpu().numpy()
    #
    #     # 使用PCA进行降维，目标维度设置为3
    #     pca = PCA(n_components=3)
    #     reduced_data = pca.fit_transform(x_indice)
    #     # print(reduced_data.shape)
    #
    #     # 绘制降维后的数据
    #     fig = plt.figure(figsize=(12, 10))
    #     ax1 = fig.add_subplot(111, projection='3d')
    #     ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.7)
    #     for i, (x, y, z) in enumerate(reduced_data):
    #         ax1.text(x, y, z, str(i), fontsize=8)  # 编号从1开始
    #     ax1.set_title("3D Visualization of the Data Distribution")
    #     ax1.set_xlabel("Principal Component 1")
    #     ax1.set_ylabel("Principal Component 2")
    #     ax1.set_zlabel("Principal Component 3")
    #     plt.grid(True)
    #     plt.show()

    def forward(self, x, x_past):
        _, indices, _ = self.encode_to_z(x)  # indices Batch, 2 (64, 2)
        # past_indices = []
        # past_embeds = []
        # # print(len(x_past_list))
        # for p in x_past_list:
        #     _, p_indices, p_embeds = self.encode_to_z(p)
        #     past_indices.append(p_indices)
        #     past_embeds.append(p_embeds)
        # # print('p_indice', past_indices)
        # past_indices = torch.cat(past_indices, dim=1)
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=indices.device) * self.sos_token

        r = math.floor(self.gamma(np.random.uniform()) * indices.shape[1])
        # print('r', r)
        sample = torch.rand(indices.shape, device=indices.device).topk(r, dim=1).indices
        #print('sample', sample)
        mask = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
        mask.scatter_(dim=1, index=sample, value=True)
        masked_indices = self.mask_token_id * torch.ones_like(indices, device=indices.device)
        #print('mask_indices', masked_indices)
        a_indices = mask * indices + (~mask) * masked_indices
        # print('a_indices', a_indices)

        a_indices = torch.cat((sos_tokens, a_indices), dim=1)
        target = torch.cat((sos_tokens, indices), dim=1)

        # a_indices = torch.cat((past_indices, a_indices), dim=1)

        # test_indice = torch.cat((past_indices, indices), dim=1)

        # print('target', target)
        # dasd
        # # sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        # #sos_tokens = sos_tokens.long().to("cuda:1")
        # # print('indice', indices)
        # mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        # mask = mask.round().to(dtype=torch.int64)
        # #print('mask', mask)
        # random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        # new_indices = mask * indices + (1 - mask) * random_indices
        # new_indices = torch.cat((sos_tokens, new_indices), dim=1)
        #
        # target = indices

        # logits, _ = self.transformer(sos_tokens, x_dec_emb, x_past)
        # logits, _ = self.transformer(new_indices[:, :-1], x_past)
        logits, _ = self.transformer(a_indices, x_past)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, x_past, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        # for param in self.transformer.parameters():
        #     param.requires_grad = False
        # x = torch.cat((c, x), dim=1)

        # x = self.codebook.embedding(x)
        x_past = self.token_embed(x_past)
        for k in range(steps):

            x_embed = self.codebook.embedding(x)
            logits, _ = self.transformer(c, x_embed, x_past)

            # logits, _ = self.transformer(c, x, x_past)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)
            if k ==0:
                x = torch.cat((c, x), dim=1)
            x = torch.cat((x, ix), dim=1)
            # print('xx', x[:20, :])
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



    @torch.no_grad()
    def log_images(self, x, x_past):
        log = dict()
        sampling_shape = [64, 64, 12]
        _, indices = self.encode_to_z(x)
        print('indice', indices[:20, :])
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        # start_indices = indices[:, :indices.shape[1] // 2]
        # sample_indices = self.sample(start_indices, sos_tokens, x_past, steps=indices.shape[1] - start_indices.shape[1])
        #
        # half_sample = self.z_to_image(indices=sample_indices, sampling_shape=sampling_shape)
        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, x_past, steps=indices.shape[1])
        print('full_indice', sample_indices[:20, :])
        dsad

        return half_sample, full_sample, target_sample
        # return log, torch.concat((x, x_rec, half_sample, full_sample))
