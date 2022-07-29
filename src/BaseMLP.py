import torch.nn as nn
from src.config import *


class BaseMLP(nn.Module):
    def __init__(self, args: ConfigBase):
        super(BaseMLP, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim

        # CNN_layers: [1-128-128-128-1]
        MLP_Layers = nn.ModuleList()
        for i in range(len(args.structure) - 1):
            if i < len(args.structure) - 2:
                MLP_Layers.append(
                    nn.Sequential(
                        nn.Linear(args.structure[i], args.structure[i + 1]),
                        nn.ReLU()
                    )
                )
            else:
                MLP_Layers.append(
                    nn.Sequential(
                        nn.Linear(args.structure[i], args.structure[i + 1]),
                    )
                )
        self.MLP_Layers = MLP_Layers

    def forward(self, x):
        x = self.MLP_Layers(x)
        return x

