import torch
import warnings

def truncated_laplace(mean, T, truncate_by_zero=False):
    if False:
        return 10
    'Generating a sample from a Laplace distribution, possible left-truncated at zero.\n    A bit of explanation here https://stats.stackexchange.com/a/357598 .\n    '
    assert isinstance(mean, torch.Tensor)
    if not truncate_by_zero:
        percentile = 0.0
    else:
        if not (mean >= 0.0).all():
            warnings.warn(f'means are supposed to be non-negative, but got {mean}')
            mean = torch.clamp_min(mean, 0.0)
        lower_bound = mean.new_tensor([0.0])
        percentile = 0.5 + 0.5 * torch.sign(lower_bound - mean) * (1.0 - torch.exp(-1.0 / T * torch.abs(mean - lower_bound)))
    p = torch.empty_like(mean).uniform_() * (1.0 - percentile) + percentile
    return mean - T * torch.sign(p - 0.5) * torch.log(1 - 2 * torch.abs(p - 0.5))