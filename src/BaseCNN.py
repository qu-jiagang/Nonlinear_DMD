import torch.nn as nn
from config import *


class BaseCNN(nn.Module):
    def __init__(self, args:ConfigBase):
        super(BaseCNN, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim

        # CNN_layers: [1-128-128-128-1]
        CNN_Layers = nn.ModuleList()
        for i in range(len(args.structure) - 1):
            if i<len(args.structure) - 2:
                CNN_Layers.append(
                    nn.Sequential(
                        nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                        nn.ReLU()
                    )
                )
            else:
                CNN_Layers.append(
                    nn.Sequential(
                        nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                    )
                )
        self.CNN_layers = CNN_Layers

    def forward(self, x):
        x = self.CNN_layers(x)
        return x

