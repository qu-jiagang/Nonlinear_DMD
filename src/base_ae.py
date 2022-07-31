import torch.nn as nn
from src.base_mlp import BaseMLP
from src.config import *


class BaseAE(nn.Module):
    def __init__(self, args: ConfigBaseAE):
        super(BaseAE, self).__init__()
        self.encoder = BaseMLP(args.encoder)
        self.latent = BaseMLP(args.latent)
        self.decoder = BaseMLP(args.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x
