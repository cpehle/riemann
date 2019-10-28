import torch


def gram_schmidt(a):
    m, n = a.shape
    q = torch.zeros([m, n], dtype=a.dtype, device=a.device)
    r = torch.zeros([n, n], dtype=a.dtype, device=a.device)

    for j in range(n):
        v = a[:, j]
        for i in range(j):
            r[i, j] = torch.dot(q[:, i], a[:, j])
            v = v - (r[i, j] * q[:, i])
        r[j, j] = torch.linalg.norm(v)
        q[:, j] = v / r[j, j]

    return q, r


def modified_gram_schmidt(a):
    m, n = a.shape

    q = torch.zeros([m, n], dtype=a.dtype, device=a.device)
    r = torch.zeros([n, n], dtype=a.dtype, device=a.device)

    for i in range(n):
        r[i, i] = torch.linalg.norm(a[:, i])
        q[:, i] = a[:, i] / r[i, i]
        for j in range(i, n):
            r[i, j] = torch.dot(q[:, i], a[:, j])
            a[:, j] = a[:, j] - r[i, j] * q[:, i]
    return q, r
