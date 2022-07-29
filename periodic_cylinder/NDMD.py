import torch
import numpy as np
from src.config import Config

# Nx, Ny = 192,384
# periodic = np.fromfile('../dataset/periodic.dat').reshape([1000, Nx, Ny])

config = {'input_dim':1, 'latent_dim':10}
a = Config.from_dict(config)

def func(args:dict):
    print(args.latent_dim)

func(a)
