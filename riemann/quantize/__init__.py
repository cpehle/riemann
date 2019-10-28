import torch


def binarize(x):
    mean = torch.mean(x)
    return torch.sign(x - mean)


def binarize_esser(w_prev, w, h=0.1):
    return torch.where(
        w <= (-0.5 - h),
        -1 * torch.ones_like(w),
        torch.where(
            (w >= (-0.5 + h)) * (w <= (0.5 - h)),
            torch.zeros_like(w),
            torch.where(w >= (0.5 + h), torch.ones_like(w), w_prev),
        ),
    )
