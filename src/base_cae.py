import torch.nn as nn
from src.base_mlp import BaseMLP
from src.base_cnn import BaseCNN
from src.config import *


class BaseCAE(nn.Module):
    def __init__(self, args: ConfigBaseCAE):
        super(BaseCAE, self).__init__()

        self.args = args

        self.encoder = BaseCNN(args.encoder)
        self.encoder_mlp = BaseMLP(args.encoder_MLP)
        latent_structure = [args.latent_dim, 128, args.latent_dim]
        args_latent = ConfigBase(latent_structure)
        self.latent = BaseMLP(args_latent)
        self.decoder_mlp = BaseMLP(args.decoder_MLP)
        self.decoder = BaseCNN(args.decoder)

    def forward(self, x):
        channels = x.size(0)
        x = self.encoder(x)
        x = x.reshape([channels,-1])
        x = self.encoder_mlp(x)
        x = self.latent(x)
        x = self.decoder_mlp(x)
        x = x.reshape([channels, self.decoder.input_dim, self.args.datasize_pooled[0],self.args.datasize_pooled[1]])
        x = self.decoder(x)
        return x
