import torch


def polar_retraction(x, omega):
    """Based on the polar decomposition of a unitary matrix as.
    """
    n = omega.shape[0]
    w, s, v = torch.svd(torch.eye(n) + omega)

    return x.mm(w.mm(v))
