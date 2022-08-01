import torch.nn as nn
from src.config import *


class MaxPoolCNN(nn.Module):
    def __init__(self, args:ConfigBase):
        super(MaxPoolCNN, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim

        # CNN_layers: [1-128-128-128-1]
        CNN_Layers = nn.ModuleList()
        for i in range(len(args.structure) - 1):
            if i<len(args.structure) - 2:
                if args.batch_normalization:
                    CNN_Layers.append(
                        nn.Sequential(
                            nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                            nn.BatchNorm2d(args.structure[i+1]),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )
                    )
                else:
                    CNN_Layers.append(
                        nn.Sequential(
                            nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )
                    )
            else:
                CNN_Layers.append(
                    nn.Sequential(
                        nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                        nn.MaxPool2d(2),
                    )
                )
        self.CNN_layers = CNN_Layers

    def forward(self, x):
        for layer in self.CNN_layers:
            x = layer(x)
        return x


class UpsampleCNN(nn.Module):
    def __init__(self, args:ConfigBase):
        super(UpsampleCNN, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim

        # CNN_layers: [1-128-128-128-1]
        CNN_Layers = nn.ModuleList()
        for i in range(len(args.structure) - 1):
            if i<len(args.structure) - 2:
                if args.batch_normalization:
                    CNN_Layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                            nn.BatchNorm2d(args.structure[i+1]),
                            nn.ReLU(),
                        )
                    )
                else:
                    CNN_Layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                            nn.ReLU(),
                        )
                    )
            else:
                CNN_Layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                    )
                )
        self.CNN_layers = CNN_Layers

    def forward(self, x):
        for layer in self.CNN_layers:
            x = layer(x)
        return x

