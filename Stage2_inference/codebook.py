import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, beta):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)
        # self.embedding.eval()

    def forward(self, z):
        # z = z.permute()
        # z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim) # B, 64, L -> B*L, 64

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))  # B*L, num_codebook
        min_encoding_indices = torch.argmin(d, dim=1)   # B * L
        z_q = self.embedding(min_encoding_indices).view(z.shape) # B*L*target_dim, 64

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        # z_q = z_q.view

        # z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss