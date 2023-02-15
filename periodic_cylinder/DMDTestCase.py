import numpy as np
from src.dmd import snapshot_dmd
import matplotlib.pyplot as plt


Nx, Ny = 192,384
periodic = np.fromfile('../dataset/periodic.dat').reshape([1000, Nx, Ny])

modes, coeffs = snapshot_dmd(periodic[:-1],periodic[1:], rank=5, dask=True)
modes_real = np.real(modes)
modes_imag = np.imag(modes)

coefs_real = np.real(coeffs)
coefs_imag = np.imag(coeffs)

plt.figure()
for i in range(5):
    plt.subplot(511+i)
    plt.imshow(modes_real[i], cmap=plt.cm.RdBu_r)

plt.figure()
for i in range(5):
    plt.plot(coefs_real[i])
plt.show()
