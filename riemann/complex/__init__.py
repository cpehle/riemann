import torch


def average_retract(x, omega):
    n = x.shape[0]
    x_ = x + omega
    x_real = 0.5 * (x_[0:n, 0:n] + x_[n : 2 * n, n : 2 * n])
    x_imag = 0.5 * (x_[n : 2 * n, 0:n] - x_[0:n, n : 2 * n])
    return torch.cat((torch.cat((x_real, -x_imag)), torch.cat((x_imag, x_real))), dim=1)
