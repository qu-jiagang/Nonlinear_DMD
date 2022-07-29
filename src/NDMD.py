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

    def forward(self, x, x_shift):
        channels = x.size(0)
        out = self.encoder(x)
        z1 = self.encoder_mlp(out.reshape([channels, -1]))

        tempz = []
        for i in range(self.latent_dim):
            tempz.append(self.decoder_mlps[i](x[:,i:i+1]))
        tempx = []
        for i in range(self.latent_dim):
            tempx.append(self.decoder(tempz[i]))
        x_reconst = torch.zeros_like(x)
        for i in range(self.latent_dim):
            x_reconst += tempx[i]

        z2 = self.latent(z1)
        tempz_shift = []
        for i in range(self.latent_dim):
            tempz_shift.append(self.decoder_mlps[i](z2[:,i:i+1]))
        tempx_shift = []
        for i in range(self.latent_dim):
            tempx_shift.append(self.decoder(tempz_shift[i]))
        x_reconst_shift = torch.zeros_like(x)
        for i in range(self.latent_dim):
            x_reconst_shift += tempx_shift[i]

        z2_from_shift = self.encoder_mlp(self.encoder(x_shift).reshape([channels,-1]))

        return x_reconst, x_reconst_shift, z2, z2_from_shift

