"""A collection of useful utility functions.

This module contains various tensor sparsity/density measurement functions, together
with some random helper functions.
"""
import argparse
from collections import OrderedDict
from copy import deepcopy
import logging
import operator
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import inspect
msglogger = logging.getLogger()

def model_device(model):
    if False:
        i = 10
        return i + 15
    'Determine the device the model is allocated on.'
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        pass
    return 'cpu'

def optimizer_device_name(opt):
    if False:
        i = 10
        return i + 15
    return str(list(list(opt.state)[0])[0].device)

def to_np(var):
    if False:
        while True:
            i = 10
    return var.data.cpu().numpy()

def size2str(torch_size):
    if False:
        return 10
    if isinstance(torch_size, torch.Size):
        return size_to_str(torch_size)
    if isinstance(torch_size, (torch.FloatTensor, torch.cuda.FloatTensor)):
        return size_to_str(torch_size.size())
    if isinstance(torch_size, torch.autograd.Variable):
        return size_to_str(torch_size.data.size())
    if isinstance(torch_size, tuple) or isinstance(torch_size, list):
        return size_to_str(torch_size)
    raise TypeError

def size_to_str(torch_size):
    if False:
        for i in range(10):
            print('nop')
    'Convert a pytorch Size object to a string'
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '(' + ', '.join(['%d' % v for v in torch_size]) + ')'

def pretty_int(i):
    if False:
        i = 10
        return i + 15
    return '{:,}'.format(i)

class MutableNamedTuple(dict):

    def __init__(self, init_dict):
        if False:
            return 10
        for (k, v) in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        if False:
            while True:
                i = 10
        return self[key]

    def __setattr__(self, key, val):
        if False:
            return 10
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

def assign_layer_fq_names(container, name=None):
    if False:
        i = 10
        return i + 15
    "Assign human-readable names to the modules (layers).\n\n    Sometimes we need to access modules by their names, and we'd like to use\n    fully-qualified names for convenience.\n    "
    for (name, module) in container.named_modules():
        module.distiller_name = name

def find_module_by_fq_name(model, fq_mod_name):
    if False:
        for i in range(10):
            print('nop')
    "Given a module's fully-qualified name, find the module in the provided model.\n\n    A fully-qualified name is assigned to modules in function assign_layer_fq_names.\n\n    Arguments:\n        model: the model to search\n        fq_mod_name: the module whose name we want to look up\n\n    Returns:\n        The module or None, if the module was not found.\n    "
    for module in model.modules():
        if hasattr(module, 'distiller_name') and fq_mod_name == module.distiller_name:
            return module
    return None

def normalize_module_name(layer_name):
    if False:
        while True:
            i = 10
    "Normalize a module's name.\n\n    PyTorch let's you parallelize the computation of a model, by wrapping a model with a\n    DataParallel module.  Unfortunately, this changs the fully-qualified name of a module,\n    even though the actual functionality of the module doesn't change.\n    Many time, when we search for modules by name, we are indifferent to the DataParallel\n    module and want to use the same module name whether the module is parallel or not.\n    We call this module name normalization, and this is implemented here.\n    "
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)

def denormalize_module_name(parallel_model, normalized_name):
    if False:
        return 10
    'Convert back from the normalized form of the layer name, to PyTorch\'s name\n    which contains "artifacts" if DataParallel is used.\n    '
    fully_qualified_name = [mod_name for (mod_name, _) in parallel_model.named_modules() if normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name

def volume(tensor):
    if False:
        while True:
            i = 10
    'return the volume of a pytorch tensor'
    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError

def density(tensor):
    if False:
        while True:
            i = 10
    'Computes the density of a tensor.\n\n    Density is the fraction of non-zero elements in a tensor.\n    If a tensor has a density of 1.0, then it has no zero elements.\n\n    Args:\n        tensor: the tensor for which we compute the density.\n\n    Returns:\n        density (float)\n    '
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)

def sparsity(tensor):
    if False:
        return 10
    'Computes the sparsity of a tensor.\n\n    Sparsity is the fraction of zero elements in a tensor.\n    If a tensor has a density of 0.0, then it has all zero elements.\n    Sparsity and density are complementary.\n\n    Args:\n        tensor: the tensor for which we compute the density.\n\n    Returns:\n        sparsity (float)\n    '
    return 1.0 - density(tensor)

def sparsity_3D(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Filter-wise sparsity for 4D tensors'
    if tensor.dim() != 4:
        return 0
    l1_norms = distiller.norms.filters_lp_norm(tensor, p=1, length_normalized=False)
    num_nonzero_filters = len(torch.nonzero(l1_norms))
    num_filters = tensor.size(0)
    return 1 - num_nonzero_filters / num_filters

def density_3D(tensor):
    if False:
        print('Hello World!')
    'Filter-wise density for 4D tensors'
    return 1 - sparsity_3D(tensor)

def sparsity_2D(tensor):
    if False:
        print('Hello World!')
    "Create a list of sparsity levels for each channel in the tensor 't'\n\n    For 4D weight tensors (convolution weights), we flatten each kernel (channel)\n    so it becomes a row in a 3D tensor in which each channel is a filter.\n    So if the original 4D weights tensor is:\n        #OFMs x #IFMs x K x K\n    The flattened tensor is:\n        #OFMS x #IFMs x K^2\n\n    For 2D weight tensors (fully-connected weights), the tensors is shaped as\n        #IFMs x #OFMs\n    so we don't need to flatten anything.\n\n    To measure 2D sparsity, we sum the absolute values of the elements in each row,\n    and then count the number of rows having sum(abs(row values)) == 0.\n    "
    if tensor.dim() == 4:
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        view_2d = tensor
    else:
        return 0
    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs / num_structs

def density_2D(tensor):
    if False:
        return 10
    'Kernel-wise sparsity for 4D tensors'
    return 1 - sparsity_2D(tensor)

def non_zero_channels(tensor):
    if False:
        print('Hello World!')
    'Returns the indices of non-zero channels.\n\n    Non-zero channels are channels that have at least one coefficient that\n    is not zero.  Counting non-zero channels involves some tensor acrobatics.\n    '
    if tensor.dim() != 4:
        raise ValueError('Expecting a 4D tensor')
    norms = distiller.norms.channels_lp_norm(tensor, p=1)
    nonzero_channels = torch.nonzero(norms)
    return nonzero_channels

def sparsity_ch(tensor):
    if False:
        i = 10
        return i + 15
    'Channel-wise sparsity for 4D tensors'
    if tensor.dim() != 4:
        return 0
    nonzero_channels = len(non_zero_channels(tensor))
    n_channels = tensor.size(1)
    return 1 - nonzero_channels / n_channels

def density_ch(tensor):
    if False:
        return 10
    'Channel-wise density for 4D tensors'
    return 1 - sparsity_ch(tensor)

def sparsity_blocks(tensor, block_shape):
    if False:
        while True:
            i = 10
    'Block-wise sparsity for 4D tensors\n\n    Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1\n    '
    if tensor.dim() != 4:
        raise ValueError('sparsity_blocks is only supported for 4-D tensors')
    if len(block_shape) != 4:
        raise ValueError('Block shape must be specified as a 4-element tuple')
    (block_repetitions, block_depth, block_height, block_width) = block_shape
    if not block_width == block_height == 1:
        raise ValueError('Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1')
    super_block_volume = volume(block_shape)
    num_super_blocks = volume(tensor) / super_block_volume
    (num_filters, num_channels) = (tensor.size(0), tensor.size(1))
    kernel_size = tensor.size(2) * tensor.size(3)
    if block_depth > 1:
        view_dims = (num_filters * num_channels // (block_repetitions * block_depth), block_repetitions * block_depth, kernel_size)
    else:
        view_dims = (num_filters // block_repetitions, block_repetitions, -1)
    view1 = tensor.view(*view_dims)
    block_sums = view1.abs().sum(dim=1)
    nonzero_blocks = len(torch.nonzero(block_sums))
    return 1 - nonzero_blocks / num_super_blocks

def sparsity_matrix(tensor, dim):
    if False:
        i = 10
        return i + 15
    'Generic sparsity computation for 2D matrices'
    if tensor.dim() != 2:
        return 0
    num_structs = tensor.size()[dim]
    nonzero_structs = len(torch.nonzero(tensor.abs().sum(dim=1 - dim)))
    return 1 - nonzero_structs / num_structs

def sparsity_cols(tensor, transposed=True):
    if False:
        print('Hello World!')
    'Column-wise sparsity for 2D tensors\n\n    PyTorch GEMM matrices are transposed before they are used in the GEMM operation.\n    In other words the matrices are stored in memory transposed.  So by default we compute\n    the sparsity of the transposed dimension.\n    '
    if transposed:
        return sparsity_matrix(tensor, 0)
    return sparsity_matrix(tensor, 1)

def density_cols(tensor, transposed=True):
    if False:
        return 10
    'Column-wise density for 2D tensors'
    return 1 - sparsity_cols(tensor, transposed)

def sparsity_rows(tensor, transposed=True):
    if False:
        while True:
            i = 10
    'Row-wise sparsity for 2D matrices\n\n    PyTorch GEMM matrices are transposed before they are used in the GEMM operation.\n    In other words the matrices are stored in memory transposed.  So by default we compute\n    the sparsity of the transposed dimension.\n    '
    if transposed:
        return sparsity_matrix(tensor, 1)
    return sparsity_matrix(tensor, 0)

def density_rows(tensor, transposed=True):
    if False:
        print('Hello World!')
    'Row-wise density for 2D tensors'
    return 1 - sparsity_rows(tensor, transposed)

def model_sparsity(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    if False:
        i = 10
        return i + 15
    'Returns the model sparsity as a fraction in [0..1]'
    (sparsity, _, _) = model_params_stats(model, param_dims, param_types)
    return sparsity

def model_params_size(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    if False:
        for i in range(10):
            print('nop')
    'Returns the size of the model parameters, w/o counting zero coefficients'
    (_, _, sparse_params_cnt) = model_params_stats(model, param_dims, param_types)
    return sparse_params_cnt

def model_params_stats(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    if False:
        print('Hello World!')
    'Returns the model sparsity, weights count, and the count of weights in the sparse model.\n\n    Returns:\n        model_sparsity - the model weights sparsity (in percent)\n        params_cnt - the number of weights in the entire model (incl. zeros)\n        params_nnz_cnt - the number of weights in the entire model, excluding zeros.\n                         nnz stands for non-zeros.\n    '
    params_cnt = 0
    params_nnz_cnt = 0
    for (name, param) in model.state_dict().items():
        if param.dim() in param_dims and any((type in name for type in param_types)):
            _density = density(param)
            params_cnt += torch.numel(param)
            params_nnz_cnt += param.numel() * _density
    model_sparsity = (1 - params_nnz_cnt / params_cnt) * 100
    return (model_sparsity, params_cnt, params_nnz_cnt)

def norm_filters(weights, p=1):
    if False:
        while True:
            i = 10
    return distiller.norms.filters_lp_norm(weights, p)

def model_numel(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    if False:
        return 10
    "Count the number elements in a model's parameter tensors"
    total_numel = 0
    for (name, param) in model.state_dict().items():
        if param.dim() in param_dims and any((type in name for type in param_types)):
            total_numel += torch.numel(param)
    return total_numel

def activation_channels_l1(activation):
    if False:
        while True:
            i = 10
    "Calculate the L1-norms of an activation's channels.\n\n    The activation usually has the shape: (batch_size, num_channels, h, w).\n\n    When the activations are computed on a distributed GPU system, different parts of the\n    activation tensor might be computed by a differnt GPU. If this function is called from\n    the forward-callback of some activation module in the graph, we will only witness part\n    of the batch.  For example, if the batch_size is 256, and we are using 4 GPUS, instead\n    of seeing activations with shape = (256, num_channels, h, w), we may see 4 calls with\n    shape = (64, num_channels, h, w).\n\n    Since we want to calculate the average of the L1-norm of each of the channels of the\n    activation, we need to move the partial sums results to the CPU, where they will be\n    added together.\n\n    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the\n    activations in the mini-batch, compute the mean of the L! magnitude of each channel).\n    "
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))
        featuremap_norms = view_2d.norm(p=1, dim=1)
        featuremap_norms_mat = featuremap_norms.view(activation.size(0), activation.size(1))
    elif activation.dim() == 2:
        featuremap_norms_mat = activation.norm(p=1, dim=1)
    else:
        raise ValueError('activation_channels_l1: Unsupported shape: '.format(activation.shape))
    return featuremap_norms_mat.mean(dim=0).cpu()

def activation_channels_means(activation):
    if False:
        while True:
            i = 10
    'Calculate the mean of each of an activation\'s channels.\n\n    The activation usually has the shape: (batch_size, num_channels, h, w).\n\n    "We first use global average pooling to convert the output of layer i, which is a\n    c x h x w tensor, into a 1 x c vector."\n\n    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the\n    activations in the mini-batch, compute the mean of the L1 magnitude of each channel).\n    '
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))
        featuremap_means = view_2d.mean(dim=1)
        featuremap_means_mat = featuremap_means.view(activation.size(0), activation.size(1))
    elif activation.dim() == 2:
        featuremap_means_mat = activation.mean(dim=1)
    else:
        raise ValueError('activation_channels_means: Unsupported shape: '.format(activation.shape))
    return featuremap_means_mat.mean(dim=0).cpu()

def activation_channels_apoz(activation):
    if False:
        print('Hello World!')
    'Calculate the APoZ of each of an activation\'s channels.\n\n    APoZ is the Average Percentage of Zeros (or simply: average sparsity) and is defined in:\n    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures".\n\n    The activation usually has the shape: (batch_size, num_channels, h, w).\n\n    "We first use global average pooling to convert the output of layer i, which is a\n    c x h x w tensor, into a 1 x c vector."\n\n    Returns - for each channel: the batch-mean of its sparsity.\n    '
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))
        featuremap_apoz = view_2d.abs().gt(0).sum(dim=1).float() / (activation.size(2) * activation.size(3))
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0).sum(dim=1).float() / activation.size(1)
    else:
        raise ValueError('activation_channels_apoz: Unsupported shape: '.format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()

def log_training_progress(stats_dict, params_dict, epoch, steps_completed, total_steps, log_freq, loggers):
    if False:
        return 10
    "Log information about the training progress, and the distribution of the weight tensors.\n\n    Args:\n        stats_dict: A tuple of (group_name, dict(var_to_log)).  Grouping statistics variables is useful for logger\n          backends such as TensorBoard.  The dictionary of var_to_log has string key, and float values.\n          For example:\n              stats = ('Peformance/Validation/',\n                       OrderedDict([('Loss', vloss),\n                                    ('Top1', top1),\n                                    ('Top5', top5)]))\n        params_dict: A parameter dictionary, such as the one returned by model.named_parameters()\n        epoch: The current epoch\n        steps_completed: The current step in the epoch\n        total_steps: The total number of training steps taken so far\n        log_freq: The number of steps between logging records\n        loggers: A list of loggers to send the log info to\n    "
    if loggers is None:
        return
    if not isinstance(loggers, list):
        loggers = [loggers]
    for logger in loggers:
        logger.log_training_progress(stats_dict, epoch, steps_completed, total_steps, freq=log_freq)
        logger.log_weights_distribution(params_dict, steps_completed)

def log_activation_statistics(epoch, phase, loggers, collector):
    if False:
        i = 10
        return i + 15
    'Log information about the sparsity of the activations'
    if collector is None:
        return
    for logger in loggers:
        logger.log_activation_statistic(phase, collector.stat_name, collector.value(), epoch)

def log_weights_sparsity(model, epoch, loggers):
    if False:
        for i in range(10):
            print('nop')
    'Log information about the weights sparsity'
    for logger in loggers:
        logger.log_weights_sparsity(model, epoch)

def log_model_buffers(model, buffer_names, tag_prefix, epoch, steps_completed, total_steps, log_freq, loggers=()):
    if False:
        return 10
    "\n    Log values of model buffers. 'buffer_names' is a list of buffers to be logged (which not necessarily exist\n    in all layers in the model).\n\n    USE WITH CARE:\n        * This logger logs each value within the buffers. As such, while any buffer can be passed\n          it is not really intended for big buffers such as model weights.\n        * Special attention is needed when using this using this functionality in TensorBoardLogger, as it could\n          significantly slow down the load time of TensorBard. Please see the documentation of 'log_model_buffers'\n          in that class.\n\n    Args:\n        model: Model containing buffers to be logged\n        buffer_names: Names of buffers to be logged. Expected to be\n        tag_prefix: Prefix to be used before buffer name by logger\n        epoch: The current epoch\n        steps_completed: The current step in the epoch\n        total_steps: The total number of training steps taken so far\n        log_freq: The number of steps between logging records\n        loggers: An iterable of loggers to send the log info to\n    "
    for logger in loggers:
        logger.log_model_buffers(model, buffer_names, tag_prefix, epoch, steps_completed, total_steps, log_freq)

def has_children(module):
    if False:
        return 10
    try:
        next(module.children())
        return True
    except StopIteration:
        return False

def _validate_input_shape(dataset, input_shape):
    if False:
        return 10
    if dataset:
        try:
            return tuple(distiller.apputils.classification_get_input_shape(dataset))
        except ValueError:
            raise ValueError("Can't infer input shape for dataset {}, please pass shape directly".format(dataset))
    else:
        if input_shape is None:
            raise ValueError('Must provide either dataset name or input shape')
        if not isinstance(input_shape, tuple):
            raise TypeError('Shape should be a tuple of integers, or a tuple of tuples of integers')

        def val_recurse(in_shape):
            if False:
                print('Hello World!')
            if all((isinstance(x, int) for x in in_shape)):
                if any((x < 0 for x in in_shape)):
                    raise ValueError("Shape can't contain negative dimensions: {}".format(in_shape))
                return in_shape
            if all((isinstance(x, tuple) for x in in_shape)):
                return tuple((val_recurse(x) for x in in_shape))
            raise TypeError('Shape should be a tuple of integers, or a tuple of tuples of integers')
        return val_recurse(input_shape)

def get_dummy_input(dataset=None, device=None, input_shape=None):
    if False:
        i = 10
        return i + 15
    "Generate a representative dummy (random) input.\n\n    If a device is specified, then the dummy_input is moved to that device.\n\n    Args:\n        dataset (str): Name of dataset from which to infer the shape\n        device (str or torch.device): Device on which to create the input\n        input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing\n          arbitrarily complex collections of tensors. Used only if 'dataset' is None\n    "

    def create_single(shape):
        if False:
            i = 10
            return i + 15
        t = torch.randn(shape)
        if device:
            t = t.to(device)
        return t

    def create_recurse(shape):
        if False:
            i = 10
            return i + 15
        if all((isinstance(x, int) for x in shape)):
            return create_single(shape)
        return tuple((create_recurse(s) for s in shape))
    input_shape = input_shape or dataset.shape
    return create_recurse(input_shape)

def set_model_input_shape_attr(model, dataset=None, input_shape=None):
    if False:
        print('Hello World!')
    "Sets an attribute named 'input_shape' within the model instance, specifying the expected input shape\n\n    Args:\n          model (nn.Module): Model instance\n          dataset (str): Name of dataset from which to infer input shape\n          input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing\n            arbitrarily complex collections of tensors. Used only if 'dataset' is None\n    "
    if not hasattr(model, 'input_shape'):
        model.input_shape = _validate_input_shape(dataset, input_shape)

def make_non_parallel_copy(model):
    if False:
        print('Hello World!')
    'Make a non-data-parallel copy of the provided model.\n\n    torch.nn.DataParallel instances are removed.\n    '

    def replace_data_parallel(container):
        if False:
            while True:
                i = 10
        for (name, module) in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)
    return new_model

def set_seed(seed):
    if False:
        i = 10
        return i + 15
    'Seed the PRNG for the CPU, Cuda, numpy and Python'
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_deterministic(seed=0):
    if False:
        while True:
            i = 10
    'Try to configure the system for reproducible results.\n\n    Experiment reproducibility is sometimes important.  Pete Warden expounded about this\n    in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/\n    For Pytorch specifics see: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility\n    '
    msglogger.debug('set_deterministic was invoked')
    if seed is None:
        seed = 0
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    if False:
        while True:
            i = 10
    'Function to load YAML file using an OrderedDict\n\n    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts\n    '

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        if False:
            print('Hello World!')
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, OrderedLoader)

def yaml_ordered_save(fname, ordered_dict):
    if False:
        print('Hello World!')

    def ordered_dict_representer(self, value):
        if False:
            while True:
                i = 10
        return self.represent_mapping('tag:yaml.org,2002:map', value.items())
    yaml.add_representer(OrderedDict, ordered_dict_representer)
    with open(fname, 'w') as f:
        yaml.dump(ordered_dict, f, default_flow_style=False)

def float_range_argparse_checker(min_val=0.0, max_val=1.0, exc_min=False, exc_max=False):
    if False:
        for i in range(10):
            print('nop')

    def checker(val_str):
        if False:
            for i in range(10):
                print('nop')
        val = float(val_str)
        (min_op, min_op_str) = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        (max_op, max_op_str) = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        raise argparse.ArgumentTypeError('Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker

def filter_kwargs(dict_to_filter, function_to_call):
    if False:
        return 10
    "Utility to check which arguments in the passed dictionary exist in a function's signature\n\n    The function returns two dicts, one with just the valid args from the input and one with the invalid args.\n    The caller can then decide to ignore the existence of invalid args, depending on context.\n    "
    sig = inspect.signature(function_to_call)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    valid_args = {}
    invalid_args = {}
    for key in dict_to_filter:
        if key in filter_keys:
            valid_args[key] = dict_to_filter[key]
        else:
            invalid_args[key] = dict_to_filter[key]
    return (valid_args, invalid_args)

def convert_tensors_recursively_to(val, *args, **kwargs):
    if False:
        print('Hello World!')
    ' Applies `.to(*args, **kwargs)` to each tensor inside val tree. Other values remain the same.'
    if isinstance(val, torch.Tensor):
        return val.to(*args, **kwargs)
    if isinstance(val, (tuple, list)):
        return type(val)((convert_tensors_recursively_to(item, *args, **kwargs) for item in val))
    return val

def param_name_2_module_name(param_name):
    if False:
        i = 10
        return i + 15
    return '.'.join(param_name.split('.')[:-1])