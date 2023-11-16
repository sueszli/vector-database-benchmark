import numpy as np
import torch.nn as nn

def normal_init(module, mean=0, std=1, bias=0):
    if False:
        print('Hello World!')
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def bias_init_with_prob(prior_prob):
    if False:
        print('Hello World!')
    'initialize conv/fc bias value according to a given probability value.'
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init