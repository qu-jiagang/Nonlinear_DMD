import torch
import numpy as np


Nx, Ny = 192,384
periodic = np.fromfile('../dataset/periodic.dat').reshape([1000, Nx, Ny])


