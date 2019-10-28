import torch

"""

References:

- http://www.cis.upenn.edu/~cis515/Diffgeom-Grassmann.Absil.pdf
- https://hal.archives-ouvertes.fr/hal-00651608/document
"""


def projective_retraction(x):
    """
    Reference:
        https://hal.archives-ouvertes.fr/hal-00651608/document
    """
    u, s, v = torch.svd(x)
    return s.diag().mm(u.mm(v.t()))


def geodesic(p, dx, t):
    u, s, v = torch.svd(dx)
    return p.mm(v.mm(torch.cos(s * t).diag())) + u.mm(torch.sin(s * t).diag())


def cotangent_to_tangent_at(p, x):
    pass


def metric_at(p, a, b):
    return torch.trace((p.t().mm(p).inverse().mm(a.t().mm(b))))
