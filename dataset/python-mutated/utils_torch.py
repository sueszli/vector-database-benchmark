import inspect
import logging
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from captum.attr import Saliency
log = logging.getLogger('NP.utils_torch')

def init_parameter(dims):
    if False:
        while True:
            i = 10
    '\n    Create and initialize a new torch Parameter.\n\n    Parameters\n    ----------\n        dims : list or tuple\n            Desired dimensions of parameter\n\n    Returns\n    -------\n        nn.Parameter\n            initialized Parameter\n    '
    if len(dims) > 1:
        return nn.Parameter(nn.init.xavier_normal_(torch.randn(dims)), requires_grad=True)
    else:
        return nn.Parameter(torch.nn.init.xavier_normal_(torch.randn([1] + dims)).squeeze(0), requires_grad=True)

def penalize_nonzero(weights, eagerness=1.0, acceptance=1.0):
    if False:
        for i in range(10):
            print('nop')
    cliff = 1.0 / (np.e * eagerness)
    return torch.log(cliff + acceptance * torch.abs(weights)) - np.log(cliff)

def create_optimizer_from_config(optimizer_name, optimizer_args):
    if False:
        return 10
    '\n    Translate the optimizer name and arguments into a torch optimizer.\n    If an optimizer object is provided, it is returned as is.\n    The optimizer is not initialized yet since this is done by the trainer.\n\n    Parameters\n        ----------\n            optimizer_name : int\n                Object provided to NeuralProphet as optimizer.\n            optimizer_args : dict\n                Arguments for the optimizer.\n\n        Returns\n        -------\n            optimizer : torch.optim.Optimizer\n                The optimizer object.\n            optimizer_args : dict\n                The optimizer arguments.\n    '
    if isinstance(optimizer_name, str):
        if optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW
            optimizer_args['weight_decay'] = 0.001
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD
            optimizer_args['momentum'] = 0.9
            optimizer_args['weight_decay'] = 0.0001
        else:
            raise ValueError(f'The optimizer name {optimizer_name} is not supported.')
    elif inspect.isclass(optimizer_name) and issubclass(optimizer_name, torch.optim.Optimizer):
        optimizer = optimizer_name
    else:
        raise ValueError('The provided optimizer is not supported.')
    return (optimizer, optimizer_args)

def interprete_model(target_model: pl.LightningModule, net: str, forward_func: str, _input: torch.Tensor=None):
    if False:
        return 10
    '\n    Returns model input attributions for a given network and forward function.\n\n    Parameters\n    ----------\n        target_model : pl.LightningModule\n            The model for which input attributions are to be computed.\n\n        net : str\n            Name of the network for which input attributions are to be computed.\n\n        forward_func : str\n            Name of the forward function for which input attributions are to be computed.\n\n        _input : torch.Tensor\n            Input for which the attributions are to be computed.\n\n    Returns\n    -------\n        torch.Tensor\n            Input attributions for the given network and forward function.\n    '
    forward = getattr(target_model, forward_func)
    saliency = Saliency(forward_func=forward)
    num_quantiles = len(target_model.quantiles)
    num_in_features = getattr(target_model, net)[0].in_features
    num_out_features = getattr(target_model, net)[-1].out_features
    num_out_features_without_quantiles = int(num_out_features / num_quantiles)
    model_input = torch.ones(1, num_in_features, requires_grad=True) if _input is None else _input
    attributions = torch.empty((0, num_in_features))
    for output_feature in range(num_out_features_without_quantiles):
        for quantile in range(num_quantiles):
            target_attribution = saliency.attribute(model_input, target=[(output_feature, quantile)], abs=False)
            attributions = torch.cat((attributions, target_attribution), 0)
    return attributions