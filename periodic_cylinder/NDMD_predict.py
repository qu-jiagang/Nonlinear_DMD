import matplotlib.pyplot as plt
import torch
from src.config import *


model = torch.load('NDMD_small.net')
model.eval()

Nx, Ny = 192, 384
Nt = 1000
C = 1
data = np.fromfile('../dataset/periodic.dat').reshape([Nt, C, Nx, Ny])

latent_dim = 2
coefs = np.zeros([Nt, latent_dim])
decomposed_fields = np.zeros([Nt, latent_dim, Nx, Ny])
for i in range(Nt):
    x = torch.from_numpy(data[i:i + 1]).float().cuda()
    d, c = model.decomposition(x)
    coefs[i:i + 1] = c.cpu().data.numpy()
    for j in range(latent_dim):
        decomposed_fields[i:i+1, j] = d[j].cpu().data.numpy()

plt.figure()
plt.plot(coefs[:, 0], coefs[:, 1], '.')

plt.figure(figsize=(8,6))
T = 100
plt.subplot(221)
plt.imshow(data[T,0],cmap=plt.cm.RdBu_r, vmax=3, vmin=-3)
plt.subplot(222)
plt.imshow(decomposed_fields[T,0]+decomposed_fields[T,1],cmap=plt.cm.RdBu_r, vmax=3, vmin=-3)
plt.subplot(223)
plt.imshow(decomposed_fields[T,0],cmap=plt.cm.RdBu_r, vmax=1.5, vmin=-1.5)
plt.subplot(224)
plt.imshow(decomposed_fields[T,1],cmap=plt.cm.RdBu_r, vmax=1.5, vmin=-1.5)
plt.show()
