import torch.nn as nn
from src.config import *
from src.residual_block import ResidualBlock


class _Conv2Block(nn.Module):
    def __init__(self, in_channel, out_channel, resnet=False):
        super(_Conv2Block, self).__init__()
        if resnet:
            self.conv_block = ResidualBlock(in_channel, out_channel)
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm2d(out_channel),
                nn.GELU()
            )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class MaxPoolCNN(nn.Module):
    def __init__(self, args: ConfigBase, resnet=False):
        super(MaxPoolCNN, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.activation = args.activation

        if args.activation == 'GELU':
            self.activation = nn.GELU()
        elif args.activation == 'ReLU':
            self.activation = nn.ReLU()

        # CNN_layers: [1-128-128-128-1]
        cnn_layers = nn.ModuleList()
        for i in range(len(args.structure) - 1):
            if i < len(args.structure) - 2:
                if args.batch_normalization:
                    cnn_layers.append(
                        nn.Sequential(
                            nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                            nn.BatchNorm2d(args.structure[i+1]),
                            self.activation,
                            # _Conv2Block(args.structure[i], args.structure[i + 1], resnet),
                            nn.MaxPool2d(2),
                        )
                    )
                else:
                    cnn_layers.append(
                        nn.Sequential(
                            nn.Conv2d(args.structure[i], args.structure[i + 1], 3, padding=1),
                            self.activation,
                            nn.MaxPool2d(2),
                        )
                    )
            else:
                cnn_layers.append(
                    nn.Sequential(
                        nn.Conv2d(args.structure[i], args.structure[i + 1], 3, padding=1),
                        nn.MaxPool2d(2),
                    )
                )
        self.CNN_layers = cnn_layers

    def forward(self, x):
        for layer in self.CNN_layers:
            x = layer(x)
        return x


class UpsampleCNN(nn.Module):
    def __init__(self, args: ConfigBase, resnet=False):
        super(UpsampleCNN, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.activation = args.activation

        if args.activation == 'GELU':
            self.activation = nn.GELU()
        elif args.activation == 'ReLU':
            self.activation = nn.ReLU()

        # CNN_layers: [1-128-128-128-1]
        CNN_Layers = nn.ModuleList()
        for i in range(len(args.structure) - 1):
            if i < len(args.structure) - 2:
                if args.batch_normalization:
                    CNN_Layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            # _Conv2Block(args.structure[i], args.structure[i + 1], resnet),
                            nn.Conv2d(args.structure[i],args.structure[i+1],3,padding=1),
                            nn.BatchNorm2d(args.structure[i+1]),
                            self.activation
                        )
                    )
                else:
                    CNN_Layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(args.structure[i], args.structure[i + 1], 3, padding=1),
                            self.activation
                        )
                    )
            else:
                CNN_Layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(args.structure[i], args.structure[i + 1], 3, padding=1),
                    )
                )
        self.CNN_layers = CNN_Layers

    def forward(self, x):
        for layer in self.CNN_layers:
            x = layer(x)
        return x
