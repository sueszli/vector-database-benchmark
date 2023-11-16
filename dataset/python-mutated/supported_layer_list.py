import copy
import logging
import threading
import numpy as np
import paddle
from paddle.base.log_helper import get_logger
from paddle.incubate import asp
__all__ = []
_logger = get_logger(__name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

def _default_pruning(weight_nparray, m, n, func_name, param_name):
    if False:
        for i in range(10):
            print('nop')
    shape = weight_nparray.shape
    weight_pruned_nparray = copy.deepcopy(weight_nparray)
    weight_sparse_mask = np.ones_like(weight_pruned_nparray)
    exlude_cond_shape2 = len(shape) == 2 and shape[0] < m
    exlude_cond_shape4 = len(shape) == 4 and shape[1] < m
    if exlude_cond_shape2:
        _logger.warning('{} is not pruned because the first dimension of {} is smaller than {}'.format(param_name, shape, m))
        return (weight_pruned_nparray, weight_sparse_mask)
    if exlude_cond_shape4:
        _logger.warning('{} is not pruned because the second dimension of {} is smaller than {}'.format(param_name, shape, m))
        return (weight_pruned_nparray, weight_sparse_mask)
    checked_func_name = asp.CheckMethod.get_checking_method(func_name)
    weight_sparse_mask = asp.create_mask(weight_nparray.T, func_name=func_name, n=n, m=m).T
    weight_pruned_nparray = np.multiply(weight_nparray, weight_sparse_mask)
    assert asp.check_sparsity(weight_pruned_nparray.T, n=n, m=m, func_name=checked_func_name), f'Pruning {param_name} weight matrix failure!!!'
    return (weight_pruned_nparray, weight_sparse_mask)
_supported_layers_and_prune_func_map_lock = threading.Lock()
supported_layers_and_prune_func_map = {}

def add_supported_layer(layer, pruning_func=None):
    if False:
        return 10
    "\n\n    Add supported layers and its corresponding pruning function.\n\n    Args:\n        name (string|Layer): The name or type of layer, needed to support. If layer is `Layer` then\n                             it would be turn to string internally. ASP would use this name to match parameter's name and call\n                             its the corresponding pruning function.\n        pruning_func (function, optional): a function type which receives five argument (weight_nparray,\n                                           m, n, func_name, param_name), weight_nparray is a nparray of weight, param_name is the name of weight,\n                                           m, n, and func_name, please see `prune_model` for details.\n\n    "
    name = None
    if isinstance(layer, str):
        name = layer
    elif isinstance(layer, paddle.nn.Layer):
        name = paddle.nn.layer.layers._convert_camel_to_snake(type(layer).__name__)
    elif issubclass(layer, paddle.nn.Layer):
        name = paddle.nn.layer.layers._convert_camel_to_snake(layer.__name__)
    else:
        assert f'The type of layer should be string of Layer, but got {type(layer)}!'
    if pruning_func is None:
        pruning_func = _default_pruning
    _supported_layers_and_prune_func_map_lock.acquire()
    supported_layers_and_prune_func_map.update({name: pruning_func})
    _supported_layers_and_prune_func_map_lock.release()
add_supported_layer('fc')
add_supported_layer('linear')
add_supported_layer('conv2d')