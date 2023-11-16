import torch.nn.functional as F

def unfold1d(x, kernel_size: int, padding_l: int, pad_value: float=0):
    if False:
        while True:
            i = 10
    'unfold T x B x C to T x B x C x K'
    if kernel_size > 1:
        (T, B, C) = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x