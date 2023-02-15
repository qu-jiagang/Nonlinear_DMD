import numpy as np
from src.pod import snapshot_pod
import matplotlib.pyplot as plt


Nx, Ny = 192,384
periodic = np.fromfile('../dataset/periodic.dat').reshape([1000, Nx, Ny])

modes, coeffs = snapshot_pod(periodic, dask=True)

for i in range(4):
    plt.subplot(411+i)
    plt.imshow(modes[i], cmap=plt.cm.RdBu_r)
plt.show()
