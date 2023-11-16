import torch

def accuracy(output, target):
    if False:
        return 10
    with torch.no_grad():
        (_, idx) = torch.max(output, 1, keepdim=True)
    return (sum(target == idx.squeeze()), len(target))