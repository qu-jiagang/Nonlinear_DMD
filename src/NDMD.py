import numpy as np
import torch.nn as nn
from AEBaseCNN import *
from BaseMLP import BaseMLP
from BaseCNN import BaseCNN
from config import *
import torch


class NDMD(nn.Module):
    def __init__(self, args: ConfigNDMD):
        super(NDMD, self).__init__()
        '''
        args: ConfigNDMD
        encoder:     [1 -> 64 -> 128]
        encoder_MLP: [encoder_output_dim -> Latent_dim]
        latent:      [Latent_dim -> 128 -> Latent_dim]
        decoder_MLP: [Latent_dim -> decoder_input_dim]
        decoder:     [128 -> 64 -> 1]
        '''
        self.latent_dim = args.latent_dim

        self.encoder = MaxPoolCNN(args.encoder)
        self.encoder_mlp = BaseMLP(args.encoder_MLP)
        self.latent = BaseMLP(args.latent)
        self.decoder_mlps = nn.ModuleList()
        for i in range(self.latent_dim):
            self.decoder_mlps.append(BaseMLP(args.decoder_MLP))
        self.decoder = UpsampleCNN(args.decoder)

    def forward(self, x):
        channels = x.size(0)
        x = self.encoder(x)
        x = x.reshape([channels, -1])
        x = self.encoder_mlp(x)
        x = self.latent(x)
        tempz = []
        for i in range(self.latent_dim):
            z = x[:,i:i+1]
            tempz.append(self.decoder_mlps[i](z))
        tempx = []
        for i in range(self.latent_dim):
            tempx.append(self.decoder(tempz[i]))
        x_reconst = torch.zeros_like(x)
        for i in range(self.latent_dim):
            x_reconst += tempx[i]
        return x_reconst


args = ConfigNDMD(input_datasize=[192,384])
Net = NDMD(args)
print(Net)

