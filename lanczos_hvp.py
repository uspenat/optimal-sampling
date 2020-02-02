import torch
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from lanczos_method import _lanczos_m_upd


def lanczos(
            operator,
            size,
            num_lanczos_vectors,
            use_gpu=False,
            start_vector=None
    ):
    """
    Parameters
    -------------
    operator: ModelHessianOperator.
    num_lanczos_vectors : int
        number of lanczos vectors to compute.
    use_gpu: bool
        if true, use cuda tensors.

    Returns
    ----------------
    T: a nv x m x m tensor, T[i, :, :] is the ith symmetric tridiagonal matrix
    V: a n x m x nv tensor, V[:, :, i] is the ith matrix with orthogonal rows
    """

    shape = (size, size)

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        if use_gpu:
            x = x.cuda()
        return operator.apply(x.float()).cpu().numpy()

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    T, V = _lanczos_m_upd(A=scipy_op, m=num_lanczos_vectors, matrix_shape=shape, SV=start_vector)
    return T, V