import torch

def enable_tf32() -> None:
    if False:
        return 10
    '\n    Overview:\n        Enable tf32 on matmul and cudnn for faster computation. This only works on Ampere GPU devices.         For detailed information, please refer to:         https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices.\n    '
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True