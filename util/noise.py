import torch

DEFAULT_EPS = 0.0005


def add_noise(x, eps=DEFAULT_EPS):
    return x + torch.randn_like(x) * eps
