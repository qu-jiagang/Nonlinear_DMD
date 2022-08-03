from src.ae_base_cnn import *
from src.base_mlp import BaseMLP
from src.config import *
import torch
import torch.nn.functional as F


class NDMD(nn.Module):
    def __init__(self, args: ConfigNDMD):
        super(NDMD, self).__init__()
        '''
        args: ConfigNDMD     [default]
        input_datasize:      [height, width]
        encoder (CNN):       [1 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512]
        encoder_MLP:         [encoder_output_dim -> 2048 -> Latent_dim]
        latent:              [Latent_dim -> 2048 -> Latent_dim]
        decoder_MLP (multi): [1 -> 2048 -> decoder_input_dim]
        decoder (CNN):       [512 -> 512 -> 512 -> 256 -> 128 -> 64 -> 1]
        '''
        self.args = args
        self.latent_dim = args.latent_dim
        self.independent_decoder = args.independent_decoder
        if args.activation == 'ReLU':
            self.activation = nn.ReLU()
        elif args.activation == 'GELU':
            self.activation = nn.GELU()

        self.encoder = MaxPoolCNN(args.encoder, args.resnet)
        self.encoder_mlp = BaseMLP(args.encoder_MLP)
        self.latent = BaseMLP(args.latent)
        self.decoder_mlps = nn.ModuleList()
        for i in range(self.latent_dim):
            self.decoder_mlps.append(BaseMLP(args.decoder_MLP))
        if args.independent_decoder:
            self.decoder = nn.ModuleList()
            for i in range(self.latent_dim):
                self.decoder.append(UpsampleCNN(args.decoder, args.resnet))
        else:
            self.decoder = UpsampleCNN(args.decoder, args.resnet)

    def forward(self, x, x_shift):
        channels = x.size(0)
        out = self.encoder(x)
        z1 = self.encoder_mlp(out.reshape([channels, -1]))

        tempz = []
        for i in range(self.latent_dim):
            tempz.append(self.decoder_mlps[i](z1[:, i:i + 1]))
        tempx = []
        for i in range(self.latent_dim):
            temp = tempz[i].reshape([channels, -1, self.args.datasize_pooled[0], self.args.datasize_pooled[1]])
            if self.independent_decoder:
                tempx.append(self.decoder[i](temp))
            else:
                tempx.append(self.decoder(temp))
        x_reconst = torch.zeros_like(x)
        for i in range(self.latent_dim):
            x_reconst += tempx[i]

        z2 = self.latent(z1)
        tempz_shift = []
        for i in range(self.latent_dim):
            tempz_shift.append(self.decoder_mlps[i](z2[:, i:i + 1]))
        tempx_shift = []
        for i in range(self.latent_dim):
            temp = tempz_shift[i].reshape([channels, -1, self.args.datasize_pooled[0], self.args.datasize_pooled[1]])
            if self.independent_decoder:
                tempx_shift.append(self.decoder[i](temp))
            else:
                tempx_shift.append(self.decoder(temp))
        x_reconst_shift = torch.zeros_like(x)
        for i in range(self.latent_dim):
            x_reconst_shift += tempx_shift[i]

        z2_from_shift = self.encoder_mlp(self.encoder(x_shift).reshape([channels, -1]))

        return x_reconst, x_reconst_shift, z2, z2_from_shift

    def loss_func(self, x_reconst, x, x_reconst_shift, x_shift, z2, z2_from_shift):
        loss1 = F.mse_loss(x_reconst, x)
        loss2 = F.mse_loss(x_reconst_shift, x_shift)
        loss3 = F.mse_loss(z2, z2_from_shift)
        sum_loss = loss1 + loss2 + loss3
        reconst_loss = loss1 + loss2
        represent_loss = loss3
        return sum_loss, reconst_loss, represent_loss

    def decomposition(self, x):
        channels = x.size(0)
        out = self.encoder(x)
        z1 = self.encoder_mlp(out.reshape([channels, -1]))

        tempz = []
        for i in range(self.latent_dim):
            tempz.append(self.decoder_mlps[i](z1[:, i:i + 1]))
        tempx = []
        for i in range(self.latent_dim):
            temp = tempz[i].reshape([channels, -1, self.args.datasize_pooled[0], self.args.datasize_pooled[1]])
            if self.independent_decoder:
                tempx.append(self.decoder[i](temp))
            else:
                tempx.append(self.decoder(temp))
        x_reconst = torch.zeros_like(x)
        for i in range(self.latent_dim):
            x_reconst += tempx[i]

        decomposed_fields = tempx
        coefs = z1

        return decomposed_fields, coefs
