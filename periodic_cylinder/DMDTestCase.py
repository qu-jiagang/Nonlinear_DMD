import numpy as np
from source.DMD import snapshot_dmd
import matplotlib.pyplot as plt


Nx, Ny = 192,384
periodic = np.fromfile('../dataset/periodic.dat').reshape([1000, Nx, Ny])

modes, coefs = snapshot_dmd(periodic[:-1],periodic[1:], dask=True)
modes_real = np.real(modes)
modes_imag = np.imag(modes)

coefs_real = np.real(coefs)
coefs_imag = np.imag(coefs)

plt.figure()
for i in range(4):
    plt.subplot(411+i)
    plt.imshow(modes_real[i], cmap=plt.cm.RdBu_r)

plt.figure()
for i in range(4):
    plt.plot(coefs_real[i])
plt.show()
