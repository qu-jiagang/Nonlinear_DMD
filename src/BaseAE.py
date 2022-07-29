import torch.nn as nn
import numpy as np
from BaseMLP import BaseMLP
from config import *


class BaseAE(nn.Module):
    def __init__(self, args: ConfigBaseAE):
        super(BaseAE, self).__init__()

        self.encoder = BaseMLP(args.encoder)
        latent_structure = [args.encoder.output_dim, 128, args.decoder.input_dim]
        args_latent = ConfigBase(latent_structure)
        self.latent = BaseMLP(args_latent)
        self.decoder = BaseMLP(args.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x


args = ConfigBaseAE()
net = BaseAE(args)
print(net)