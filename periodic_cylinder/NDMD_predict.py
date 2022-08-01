import matplotlib.pyplot as plt
import torch
from src.config import *


model = torch.load('NDMD_small.net')

Nx, Ny = 192, 384
Nt = 1000
C = 1
data = np.fromfile('../dataset/periodic.dat').reshape([Nt, C, Nx, Ny])

latent_dim = 2
coefs = np.zeros([Nt, latent_dim])
for i in range(Nt):
    x = torch.from_numpy(data[i:i + 1]).float().cuda()
    _, c = model.decomposition(x)
    coefs[i:i + 1] = c.cpu().data.numpy()

plt.plot(coefs[:, 0], coefs[:, 1], '.')
plt.show()
