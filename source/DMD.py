import numpy as np
from numpy import dot, diag, multiply, power
from numpy.linalg import svd, inv, pinv
import dask.array as da


# Definition of DMD
# Dynamic Mode Decomposition

def snapshot_dmd(snapshots, snapshots_prime, rank=None, dask=False):
    try:
        t,m,n = snapshots.shape
    except:
        print('Error: wrong shape of the input snapshots')
        return 0

    X = snapshots.reshape([t, m * n]).T
    Y = snapshots_prime.reshape([t, m * n]).T

    modes, coefs = dmd(X,Y,rank,dask)

    return modes, coefs


def dmd(X, Y, rank=None, dask=False):

    if dask:
        X = da.from_array(X)
        U2, Sig2, Vh2 = da.linalg.svd(X)
        U2, Sig2, Vh2 = np.array(U2), np.array(Sig2), np.array(Vh2)
    else:
        U2, Sig2, Vh2 = svd(X, False)

    # rank truncation
    if not rank:
        rank = len(X[0])
    U = U2[:, :rank]
    S = diag(Sig2)[:rank, :rank]
    V = Vh2.conj().T[:, :rank]

    # definition of virtual time domain
    t = np.linspace(0, len(X[0]) + 1, len(X[0]) + 1)
    dt = t[1] - [0]

    # build A tilde
    A_tilde = dot(dot(dot(U.conj().T, Y), V), inv(S))
    mu, W = np.linalg.eig(A_tilde)

    # build DMD modes
    Phi = dot(dot(dot(Y, V), inv(S)), W)

    # compute time evolution
    b = dot(pinv(Phi), X[:, 0])
    Psi = np.zeros([rank, len(t)], dtype='complex')
    for i, _t in enumerate(t):
        Psi[:, i] = multiply(power(mu, _t / dt), b)

    return Phi, Psi