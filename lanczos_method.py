import numpy as np


def _lanczos_m_upd(A, m, matrix_shape, nv=1, rademacher=False, SV=None):
    """
    Lanczos algorithm computes symmetric m x m tridiagonal matrix T and matrix V with orthogonal rows
        constituting the basis of the Krylov subspace K_m(A, x),
        where x is an arbitrary starting unit vector.
        This implementation parallelizes `nv` starting vectors.

    Arguments:
        m: number of Lanczos steps
        nv: number of random vectors
        rademacher: True to use Rademacher distribution,
                    False - standard normal for random vectors
        SV: specified starting vectors

    Returns:
        T: a nv x m x m tensor, T[i, :, :] is the ith symmetric tridiagonal matrix
        V: a n x m x nv tensor, V[:, :, i] is the ith matrix with orthogonal rows
    """

    orthtol = 1e-16

    if type(SV) != np.ndarray:
        if rademacher:
            # SV = np.sign(np.random.randn(A.shape[0], nv))
            SV = np.sign(np.random.randn(matrix_shape[0], nv))
        else:
            # SV = np.random.randn(A.shape[0], nv)  # init random vectors in columns: n x nv
            SV = np.random.randn(matrix_shape[0], nv)

    V = np.zeros((SV.shape[0], m, nv))
    T = np.zeros((nv, m, m))

    np.divide(SV, np.linalg.norm(SV, axis=0), out=SV)  # normalize each column
    V[:, 0, :] = SV

    w = A.matvec(SV.squeeze())
    w = w.reshape(-1, 1)
    alpha = np.einsum('ij,ij->j', w, SV)
    w -= alpha[None, :] * SV
    beta = np.einsum('ij,ij->j', w, w)
    np.sqrt(beta, beta)

    T[:, 0, 0] = alpha
    T[:, 0, 1] = beta
    T[:, 1, 0] = beta

    np.divide(w, beta[None, :], out=w)
    V[:, 1, :] = w
    t = np.zeros((m, nv))

    for i in range(1, m):
        SVold = V[:, i - 1, :]
        SV = V[:, i, :]

        w = A.dot(SV.squeeze())  # sparse @ dense
        w = w.reshape(-1, 1)
        w -= beta[None, :] * SVold  # n x nv
        np.einsum('ij,ij->j', w, SV, out=alpha)

        T[:, i, i] = alpha

        if i < m - 1:
            w -= alpha[None, :] * SV  # n x nv
            # reortho
            np.einsum('ijk,ik->jk', V, w, out=t)
            w -= np.einsum('ijk,jk->ik', V, t)
            np.einsum('ij,ij->j', w, w, out=beta)
            np.sqrt(beta, beta)
            np.divide(w, beta[None, :], out=w)

            T[:, i, i + 1] = beta
            T[:, i + 1, i] = beta

            # more reotho
            innerprod = np.einsum('ijk,ik->jk', V, w)
            reortho = False
            for _ in range(100):
                if (innerprod <= orthtol).sum():
                    reortho = True
                    break
                np.einsum('ijk,ik->jk', V, w, out=t)
                w -= np.einsum('ijk,jk->ik', V, t)
                np.divide(w, np.linalg.norm(w, axis=0)[None, :], out=w)
                innerprod = np.einsum('ijk,ik->jk', V, w)

            V[:, i + 1, :] = w

            if (np.abs(beta) > 1e-2).sum() == 0 or not reortho:
                break
    return T, V

