from typing import Dict, Tuple, Union
import numpy as np


class ConfigBase:
    # configurations for baseMLP and baseCNN
    # structure: list
    # e.g. config_base([1,128,1])
    # refer to a multi-layer neural network: 1->128->1
    def __init__(self, structure: list = None):
        if structure is None:
            structure = [1, 128, 128, 128, 1]
        self.structure = structure
        self.input_dim = structure[0]
        self.output_dim = structure[-1]


class ConfigBaseAE:
    def __init__(self, encoder: ConfigBase = None, decoder: ConfigBase = None):
        if encoder is None:
            encoder = ConfigBase()
        if decoder is None:
            decoder = ConfigBase()
        self.encoder = encoder
        self.decoder = decoder


class ConfigBaseCAE(ConfigBaseAE):
    def __init__(self,
                 input_datasize: list,
                 latent_dim: int = 2,
                 encoder: ConfigBase = None,
                 encoder_MLP: ConfigBase = None,
                 decoder: ConfigBase = None,
                 decoder_MLP: ConfigBase = None):
        super(ConfigBaseAE, self).__init__(encoder, decoder)
        if encoder is None:
            encoder = ConfigBase()
        if decoder is None:
            decoder = ConfigBase()
        datasize_pooled = [x / (2 ** (len(encoder.structure) - 1)) for x in input_datasize]
        if encoder_MLP is None:
            input_dim = encoder.output_dim * int(np.prod(datasize_pooled))
            encoder_MLP = ConfigBase([input_dim, 128, latent_dim])
        if decoder_MLP is None:
            output_dim = encoder.output_dim * int(np.prod(datasize_pooled))
            decoder_MLP = ConfigBase([latent_dim, 128, output_dim])
        self.encoder_MLP = encoder_MLP
        self.decoder_MLP = decoder_MLP
        self.latent_dim = latent_dim
        self.datasize_pooled = datasize_pooled
    

class ConfigNDMD:
    def __init__(self,
                 input_datasize: list, # [height, width]
                 latent_dim: int = 2,
                 encoder: ConfigBase = ConfigBase([1,64,128,256,512,512,512]),
                 encoder_MLP: ConfigBase = None,
                 decoder: ConfigBase = ConfigBase([512,512,512,256,128,64,1]),
                 decoder_MLP: ConfigBase = None):
        datasize_pooled = [x / (2 ** (len(encoder.structure) - 1)) for x in input_datasize]
        encoder_output_dim = encoder.output_dim * int(np.prod(datasize_pooled))
        datasize_sampled = [x / (2 ** (len(decoder.structure) - 1)) for x in input_datasize]
        decoder_input_dim = encoder.output_dim * int(np.prod(datasize_sampled))
        if encoder_MLP is None:
            encoder_MLP = ConfigBase([encoder_output_dim, 128, latent_dim])
        if decoder_MLP is None:
            decoder_MLP = ConfigBase([1, 128, decoder_input_dim])
        self.encoder = encoder
        self.encoder_MLP = encoder_MLP
        self.latent = ConfigBase([latent_dim, 128, latent_dim])
        self.decoder_MLP = decoder_MLP
        self.decoder = decoder

        self.latent_dim = latent_dim
