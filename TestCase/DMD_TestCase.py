import numpy as np
from numpy import pi,multiply,power,exp,cosh,tanh
from src.dmd import dmd
import matplotlib.pyplot as plt


# Test case from: http://www.pyrunner.com/weblog/2016/07/25/dmd-python/
# define time and space domains
x = np.linspace(-10, 10, 100)
t = np.linspace(0, 6*pi, 80)
dt = t[2] - t[1]
Xm,Tm = np.meshgrid(x, t)

# create three spatiotemporal patterns
f1 = multiply(20-0.2*power(Xm, 2), exp((2.3j)*Tm))
f2 = multiply(Xm, exp(0.6j*Tm))
f3 = multiply(5*multiply(1/cosh(Xm/2), tanh(Xm/2)), 2*exp((0.1+2.8j)*Tm))

# combine signals and make data matrix
D = (f1 + f2 + f3).T

# create DMD input-output matrices
X = D[:,:-1]
Y = D[:,1:]

Phi,Psi = dmd(X,Y,3)

plt.figure()
for i in range(3):
    plt.plot(np.real(Phi[i]))
    # plt.plot(np.imag(Phi[i]))

plt.figure()
for i in range(3):
    plt.subplot(311+i)
    plt.plot(np.real(Psi[i]))
    plt.plot(np.imag(Psi[i]))
plt.show()