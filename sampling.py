import torch
from lanczos_hvp import lanczos
from hvp_operator import ModelHessianOperator
from scipy.linalg import funm, inv, sqrtm
import numpy as np


def get_tensors(model, vec):
    tlist = torch.split(vec, [p.numel() for p in model.parameters()])
    tlist = tuple(t.view(*p.shape) for t, p in zip(tlist, model.parameters()))
    return tlist


def set_model_parameters(model, vec):
    print(f'norm(vec) = {np.linalg.norm(vec)}')
    vec = vec.real
    vec = torch.tensor(vec)
    tlist = get_tensors(model, vec)
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), tlist):
            # print(f'param.size() = {param.size()}')
            # print(f'new_param.size() = {new_param.size()}')
            param.copy_(new_param)


def generate_weights(model, criterion, data_input, data_target, weights_mle, arbitrary=False, k=2):
    n = weights_mle.shape[0]
    m = n // k
    print(f'n = {n}')
    print(f'm = {m}')

    z = np.random.randn(n, 1)

    op = ModelHessianOperator(model, criterion, data_input, data_target)

    T, V = lanczos(operator=op, num_lanczos_vectors=m, size=n, use_gpu=False, start_vector=z)
    print('Lanczos done')
    T = T.squeeze()
    V = V.squeeze()
    V = np.matrix(V)

    func = lambda x: x ** (-0.5)

    # make T positive definite
    # w, U = np.linalg.eigh(T)
    # c = 1e-10
    # w[w < c] = c
    # T = U @ np.diag(w) @ U.T
    # print('T found')

    e_1 = np.zeros((m, 1))
    e_1[0] = 1
    T_fun = funm(T, func)
    print('funm computed')
    #     T_fun = sqrtm(T) + inv(T)
    matvec = np.linalg.norm(z) * V @ T_fun @ e_1
    if arbitrary:
        nr = np.linalg.norm(matvec)
        matvec = np.random.randn(n, 1)
        matvec /= np.linalg.norm(matvec)
        matvec *= nr
    return weights_mle[:, None] + (1 / np.sqrt(len(data_input))) * matvec
