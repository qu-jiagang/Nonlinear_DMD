import matplotlib.pyplot as plt
import numpy as np
from src.DMD import snapshot_dmd
from src.POD import pod


# Data-Driven Science and Engineering Machine Learning Dynamical Systems and Control
# Steven L. Brunton, J. Nathan Kutz
# Page 396

# space
n = 200
L = 20
x = np.linspace(-L,L,n)

# time
m = 41
T = 10
t = np.linspace(0,T,m)

# Wave speed
c = 3

# dataset
X = np.zeros([n,m])
for j in range(m):
    X[:,j] = np.exp(-(x+15-c*t[j])**2)

# SVD/POD/PCA
U,S,Vh = np.linalg.svd(X)
V = Vh.conj().T

# POD
modes, coefs = pod(X,dask=True)
plt.figure()
plt.subplot(211)
plt.plot(modes[:4].T)
plt.subplot(212)
plt.plot(coefs[:4].T)


# DMD
Phi,Psi = snapshot_dmd(X[:,:-1],X[:,1:],4)

plt.figure()
for i in range(4):
    plt.subplot(411+i)
    plt.plot(np.real(Psi)[i])
    plt.plot(np.imag(Psi)[i])

plt.figure()
plt.subplot(211)
plt.plot(U[:,:4])
plt.subplot(212)
plt.plot(V[:,:4])
plt.show()