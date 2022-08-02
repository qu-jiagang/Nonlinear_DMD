import numpy as np


class ConfigBase:
    # configurations for baseMLP and baseCNN
    # structure: list
    # e.g. config_base([1,128,1])
    # refer to a multi-layer neural network: 1->128->1
    def __init__(self, structure: list = None, batch_normalization=False):
        if structure is None:
            structure = [1, 128, 128, 128, 1]
        self.structure = structure
        self.input_dim = structure[0]
        self.output_dim = structure[-1]
        self.batch_normalization = batch_normalization


class ConfigBaseAE:
    def __init__(self,
                 encoder: list = None,
                 latent: list = None,
                 decoder: list = None):
        if encoder is None:
            self.encoder = ConfigBase()
        if decoder is None:
            self.decoder = ConfigBase()
        if latent is None:
            self.latent = ConfigBase([self.encoder.output_dim, 128, self.decoder.input_dim])


class ConfigBaseCAE:
    def __init__(self,
                 input_datasize: list,
                 latent_dim: int = 2,
                 encoder: list = None,
                 encoder_mlp: list = None,
                 decoder: list = None,
                 decoder_mlp: list = None):
        if encoder is None:
            encoder = [1, 32, 64, 32, 1]
        if decoder is None:
            decoder = [1, 32, 64, 32, 1]
        self.datasize_pooled = [int(x / (2 ** (len(encoder) - 1))) for x in input_datasize]
        encoder_output_dim = encoder[-1] * int(np.prod(self.datasize_pooled))
        self.datasize_sampled = [int(x / (2 ** (len(decoder) - 1))) for x in input_datasize]
        decoder_input_dim = decoder[0] * int(np.prod(self.datasize_sampled))
        if encoder_mlp is None:
            encoder_mlp = [encoder_output_dim, 2048, latent_dim]
        else:
            encoder_mlp = [encoder_output_dim] + encoder_mlp + [latent_dim]
        if decoder_mlp is None:
            decoder_mlp = [latent_dim, 2048, decoder_input_dim]
        else:
            decoder_mlp = [latent_dim] + decoder_mlp + [decoder_input_dim]

        self.encoder = ConfigBase(encoder)
        self.encoder_MLP = ConfigBase(encoder_mlp)
        self.decoder_MLP = ConfigBase(decoder_mlp)
        self.decoder = ConfigBase(decoder)

        self.latent_dim = latent_dim


class ConfigNDMD:
    def __init__(self,
                 input_datasize: list,  # [height, width]
                 latent_dim: int = 2,
                 encoder: list = None,
                 encoder_mlp: list = None,
                 latent: list = None,
                 decoder: list = None,
                 decoder_mlp: list = None,
                 batch_normalization=False,
                 resnet=False,
                 independent_decoder=False,):
        if encoder is None:
            encoder = [1, 64, 128, 256, 512, 512, 512]
        if decoder is None:
            decoder = [512, 512, 512, 256, 128, 64, 1]
        self.datasize_pooled = [int(x / (2 ** (len(encoder) - 1))) for x in input_datasize]
        encoder_output_dim = encoder[-1] * int(np.prod(self.datasize_pooled))
        self.datasize_sampled = [int(x / (2 ** (len(decoder) - 1))) for x in input_datasize]
        decoder_input_dim = decoder[0] * int(np.prod(self.datasize_sampled))
        if encoder_mlp is None:
            encoder_mlp = [encoder_output_dim, 2048, latent_dim]
        else:
            encoder_mlp = [encoder_output_dim] + encoder_mlp + [latent_dim]
        if decoder_mlp is None:
            decoder_mlp = [1, 2048, decoder_input_dim]
        else:
            decoder_mlp = [1] + decoder_mlp + [decoder_input_dim]
        if latent is None:
            latent = [latent_dim, 2048, latent_dim]
        else:
            latent = [latent_dim] + latent + [latent_dim]

        self.encoder = ConfigBase(encoder, batch_normalization)
        self.encoder_MLP = ConfigBase(encoder_mlp, batch_normalization)
        self.latent = ConfigBase(latent, batch_normalization)
        self.decoder_MLP = ConfigBase(decoder_mlp, batch_normalization)
        self.decoder = ConfigBase(decoder, batch_normalization)

        self.latent_dim = latent_dim
        self.resnet = resnet
        self.independent_decoder = independent_decoder

