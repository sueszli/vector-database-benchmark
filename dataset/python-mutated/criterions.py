import torch.nn as nn

def cel(rank):
    if False:
        return 10
    'A function that creates a CrossEntropyLoss\n    criterion for training.\n    Args:\n        rank (int): worker rank\n    '
    return nn.CrossEntropyLoss().cuda(rank)