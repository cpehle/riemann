import torch


def round_stochastic(x):
    x_low = torch.floor(x)
    return torch.where(
        torch.distributions.bernoulli.Bernoulli(probs=x - x_low).sample(),
        x_low + 1,
        x_low,
    )
