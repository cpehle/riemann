import torch


def random(n):
    """Create a random orthogonal matrix
    """
    h = torch.random.randn(n, n)
    q, r = torch.qr(h)
    q_modified = q.mm(torch.diag(torch.exp(1j * np.pi * 2 * torch.random.randn(n))))
    return q_modified


def qr_retraction(x, omega):
    """
    Reference:
        Optimization Algorithms on Matrix Manifolds, P.-A. Absil, R. Mahony, R. Sepulchre, Example 4.1.2
    """
    n = omega.shape[0]
    q, r = torch.qr(torch.eye(n) + omega)
    return x.mm(q)
