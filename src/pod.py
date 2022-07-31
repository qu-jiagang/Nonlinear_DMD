import numpy as np
from numpy.linalg import svd
import dask.array as da


# definition of POD
# Proper orthogonal  decomposition
# POD is often formulated by taking the SVD of the data matrix X

def snapshot_pod(snapshots, rank=None, subtract=True, dask=False):
    # snapshots: shape of t*m*n
    # t: number of snapshots
    # m,n: spatial dimensions of the snapshots
    try:
        t,m,n = snapshots.shape
    except:
        print('Error: wrong shape of the input snapshots')
        return 0

    X = snapshots.reshape([t,m*n]).T

    modes, coefs = pod(X, rank, subtract,dask)

    if not rank or rank>len(X):
        rank = len(X[0])

    modes = modes.reshape([rank,m,n])
    coefs = coefs.reshape([rank,t])
    return modes, coefs


def pod(X, rank=None, subtract=False, dask=False):
    # X, a 2D matrix, shape of Space*Time
    if subtract:
        X -= np.mean(X, axis=1).reshape([-1,1])

    if dask:
        X = da.from_array(X)
        U, Sig, Vh = da.linalg.svd(X)
        U, Sig, Vh = np.array(U), np.array(Sig), np.array(Vh)
    else:
        U, Sig, Vh = svd(X, False)
    S = np.diag(Sig)

    if not rank or rank>len(X):
        rank = len(X[0])

    Psi = np.dot(U,S**(1/2))
    Phi = np.dot(S**(1/2),Vh).conj().T
    modes = Psi[:,:rank].T
    coefs = Phi[:,:rank].T

    return modes, coefs

