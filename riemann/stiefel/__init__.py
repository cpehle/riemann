import torch

# see http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf


def projective_retraction(x):
    # see https://hal.archives-ouvertes.fr/hal-00651608/document
    u, _, v = torch.svd(x)
    return u.mm(v.t())


def qr_retraction(x, zeta):
    """
    Reference:
        Optimization Algorithms on Matrix Manifolds, P.-A. Absil, R. Mahony, R. Sepulchre, Example 4.1.3
    """
    q, r = torch.qr(x + zeta)
    return q


def metric(a, b):
    return torch.trace(a.t().mm(b))
