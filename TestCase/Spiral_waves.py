import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, cos, tanh, exp, angle
from src.pod import snapshot_pod


# spiral waves
n = 101
L = 20
x = np.linspace(-L,L,n)
y = np.linspace(-L,L,n)
X,Y = np.meshgrid(x,y)

Xd = np.zeros([100,n,n])
for i in range(100):
    u = tanh(sqrt(X**2+Y**2))*cos(angle(X+Y*1j)-sqrt(X**2+Y**2)+i/10)
    f = u*exp(-0.01*(X**2+Y**2))
    Xd[i] = f

modes, coefs = snapshot_pod(Xd)

plt.imshow(modes[0])
plt.show()

# recurrent the Test Case

# n = 101
# L = 20
# x = np.linspace(-L,L,n)
# y = np.linspace(-L,L,n)
# X,Y = np.meshgrid(x,y)

# Xd = np.zeros([n**2,100])
# for i in range(100):
#     u = tanh(sqrt(X**2+Y**2))*cos(angle(X+Y*1j)-sqrt(X**2+Y**2)+i/10)
#     f = u*exp(-0.01*(X**2+Y**2))
#     Xd[:,i] = f.reshape([n**2])
#
# U,S,Vh = np.linalg.svd(Xd,False)
# V = Vh.conj().T
#
# plt.figure()
# plt.subplot2grid((4,1),(0,0),rowspan=2)
# plt.plot(V[:,:4])
# plt.subplot2grid((4,1),(2,0))
# plt.plot(100*S/np.sum(S),'.')
# plt.subplot2grid((4,1),(3,0))
# plt.semilogy(100*S/np.sum(S),'.')
#
# plt.figure()
# for i in range(2):
#     for j in range(2):
#         plt.subplot2grid((2,2),(i,j))
#         mode = U[:,i*2+j].reshape([n,n])
#         plt.imshow(mode,cmap=plt.cm.gray,vmin=-0.03,vmax=0.03)
#
# plt.show()

# # PLOT spiral wave
# plt.figure()
# for i in range(3):
#     plt.subplot(131+i)
#     map = Xd[:, i*30].reshape([n, n])
#     plt.imshow(map, cmap=plt.cm.gray)
#
# plt.figure()
# map = Xd[:,0].reshape([n,n])
# plt.subplot(131)
# plt.imshow(map, cmap=plt.cm.gray)
# plt.subplot(132)
# plt.imshow(np.abs(map), cmap=plt.cm.gray)
# plt.subplot(133)
# plt.imshow(map**5, cmap=plt.cm.gray)
# plt.show()
