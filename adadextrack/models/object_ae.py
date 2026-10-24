"""Original PointNet object autoencoder, with checkpoint-compatible tensor names."""
import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(True), nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True), nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.ReLU(True))
        self.fc_mu = nn.Linear(1024, latent_dim)

    def forward(self, x):
        return self.fc_mu(self.mlp(x).max(dim=1).values)


class PointDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(True),
                                 nn.Linear(512, 1024), nn.ReLU(True),
                                 nn.Linear(1024, num_points * 3))

    def forward(self, z):
        return torch.tanh(self.mlp(z).view(z.size(0), self.num_points, 3))


class PointAE(nn.Module):
    def __init__(self, latent_dim=64, num_points=2048):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointDecoder(latent_dim, num_points)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def chamfer_l2(pred, target, chunk=0):
    """Sum of the two mean nearest-neighbor squared Euclidean distances."""
    if not chunk:
        distances = torch.cdist(pred, target, p=2)
        return (distances.min(-1).values.square().mean(1)
                + distances.min(-2).values.square().mean(1)).mean()
    def directional(first, second):
        return torch.cat([torch.cdist(part, second, p=2).min(-1).values
                          for part in first.split(chunk, dim=1)], dim=1).square().mean(1)
    return (directional(pred, target) + directional(target, pred)).mean()
