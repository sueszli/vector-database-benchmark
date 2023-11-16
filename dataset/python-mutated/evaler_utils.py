import math
import numpy as np
import random
import heapq
from . import utils
from .lv_types import ImageData, PointData
from collections import OrderedDict
from itertools import groupby, islice
import operator
from typing import Callable, List, Iterable, Any, Sized, Tuple
from . import utils
from . import tensor_utils

def skip_k(l: Iterable[Any], k: int) -> Iterable[Any]:
    if False:
        for i in range(10):
            print('nop')
    'For given iterable, return only k-th items, strating from 0\n    '
    for (index, item) in enumerate(g):
        if index % k == 0:
            yield item

def to_tuples(l: Iterable[Any], key_f=lambda x: x, val_f=lambda x: x) -> Iterable[tuple]:
    if False:
        return 10
    'Apply functions on each item to generate tuples of key and value pairs\n    '
    return ((key_f(i), val_f(i)) for i in l)

def group_reduce(l: Iterable[Any], key_f=lambda x: x, val_f=lambda x: x, reducer: Callable[[List[Any]], Any]=None) -> Iterable[tuple]:
    if False:
        return 10
    'Group values by key and then apply reducer on each group\n    '
    tuples = to_tuples(l, key_f, val_f)
    tuples = sorted(tuples, key=operator.itemgetter(0))
    tuples = list(tuples)
    groups = groupby(tuples, key=operator.itemgetter(0))
    groups = ((key, (t1 for (t0, t1) in group)) for (key, group) in groups)
    if reducer:
        groups = ((k, reducer(items)) for (k, items) in groups)
    return groups

def combine_groups(existing_groups: dict, new_groups: Iterable[tuple], sort_key, reverse=False, k=1) -> None:
    if False:
        while True:
            i = 10
    'concate items in new groups with existing groups, sort items and take k items in each group\n    '
    for new_group in new_groups:
        exisiting_items = existing_groups.get(new_group[0], None)
        if exisiting_items is None:
            merged = new_group[1]
        else:
            exisiting_items = list(exisiting_items)
            new_group = list(new_group)
            merged = heapq.merge(exisiting_items, new_group[1], key=sort_key, reverse=reverse)
        merged = list(merged)
        existing_groups[new_group[0]] = list(islice(merged, k))

def topk(labels: Sized, metric: Sized=None, items: Sized=None, k: int=1, order='rnd', sort_groups=False, out_f: callable=None) -> Iterable[Any]:
    if False:
        print('Hello World!')
    'Returns groups of k items for each label sorted by metric\n\n    This function accepts batch values, for example, for image classification with batch of 100,\n    we may have 100 rows and columns for input, net_output, label, loss. We want to group by label\n    then for each group sort by loss and take first two value from each group. This would allow us \n    to display best two predictions for each class in a batch. If we sort by loss in reverse then\n    we can display worse two predictions in a batch. The parameter of this function is columns for the batch\n    i.e. in this example labels would be list of 100 values, metric would be list of 100 floats for loss per item\n    and items parameter could be list of 100 tuples of (input, output)\n    '
    if labels is None:
        if metric is not None:
            labels = [0] * len(metric)
        else:
            raise ValueError('Both labels and metric parameters cannot be None')
    labels = tensor_utils.to_scaler_list(labels)
    if metric is None or len(metric) == 0:
        metric = [0] * len(labels)
    else:
        metric = tensor_utils.to_mean_list(metric)
    if items is None or len(items) == 0:
        items = [None] * len(labels)
    else:
        items = [tensor_utils.to_np_list(item) for item in items]
    batch = list(((*i[:2], i[2:]) for i in zip(labels, metric, *items)))
    reverse = True if order == 'dsc' else False
    key_f = (lambda i: i[1]) if order != 'rnd' else lambda i: random.random()
    groups = group_reduce(batch, key_f=lambda b: b[0], reducer=lambda bi: islice(sorted(bi, key=key_f, reverse=reverse), k))
    if sort_groups:
        groups = sorted(groups.items(), key=lambda g: g[0])
    if out_f:
        return (out_val for group in groups for out_val in out_f(group))
    else:
        return groups

def topk_all(batches: Iterable[Any], batch_vals: Callable[[Any], Tuple[Sized, Sized, Sized]], out_f: callable, k: int=1, order='rnd', sort_groups=True) -> Iterable[Any]:
    if False:
        for i in range(10):
            print('nop')
    'Same as k but here we maintain top items across entire run\n    '
    merged_groups = {}
    for batch in batches:
        unpacker = lambda a0, a1, a2=None: (a0, a1, a2)
        (metric, items, labels) = unpacker(*batch_vals(batch))
        groups = topk(labels, metric, items, k=k, order=order, sort_groups=False)
        reverse = True if order == 'dsc' else False
        sort_key = (lambda g: g[1]) if order != 'rnd' else lambda g: random.random()
        combine_groups(merged_groups, groups, sort_key=sort_key, reverse=reverse, k=k)
        sorted_groups = sorted(merged_groups.items(), key=lambda g: g[0]) if sort_groups else merged_groups
        sorted_groups = list(sorted_groups)
        if out_f:
            yield (out_f(*val) for (key, vals) in sorted_groups for val in vals)
        else:
            yield sorted_groups

def reduce_params(model, param_reducer: callable, include_weights=True, include_bias=False):
    if False:
        i = 10
        return i + 15
    'aggregate weights or biases, use param_reducer to transform tensor to scaler\n    '
    for (i, (param_group_name, param_group)) in enumerate(model.named_parameters()):
        if param_group.requires_grad:
            is_bias = 'bias' in param_group_name
            if include_weights and (not is_bias) or (include_bias and is_bias):
                yield PointData(x=i, y=param_reducer(param_group), annotation=param_group_name)

def image_class_outf(label, metric, item):
    if False:
        return 10
    'item is assumed to be (input_image, logits, ....)\n    '
    net_input = tensor_utils.tensor2np(item[0]) if len(item) > 0 else None
    title = 'Label:{},Loss:{:.2f}'.format(label, metric)
    if len(item) > 1:
        net_output = tensor_utils.tensor2np(item[1])
        net_output_i = np.argmax(net_output)
        net_output_p = net_output[net_output_i]
        title += ',Prob:{:.2f},Pred:{:.2f}'.format(math.exp(net_output_p), net_output_i)
    return ImageData((net_input,), title=title)

def image_image_outf(label, metric, item):
    if False:
        print('Hello World!')
    'item is assumed to be (Image1, Image2, ....)\n    '
    return ImageData(tuple((tensor_utils.tensor2np(i) for i in item)), title='loss:{:.2f}'.format(metric))

def grads_abs_mean(model):
    if False:
        for i in range(10):
            print('nop')
    return reduce_params(model, lambda p: p.grad.abs().mean().item())

def grads_abs_sum(model):
    if False:
        for i in range(10):
            print('nop')
    return reduce_params(model, lambda p: p.grad.abs().sum().item())

def weights_abs_mean(model):
    if False:
        while True:
            i = 10
    return reduce_params(model, lambda p: p.abs().mean().item())

def weights_abs_sum(model):
    if False:
        while True:
            i = 10
    return reduce_params(model, lambda p: p.abs().sum().item())