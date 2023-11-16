import torch
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.func import functional_call
from ..pattern_matcher import CallModuleVarArgs, Match, register_graph_pattern
from .pre_grad import efficient_conv_bn_eval_pass

def efficient_conv_bn_eval(bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor):
    if False:
        print('Hello World!')
    '\n    Implementation based on https://arxiv.org/abs/2305.11624\n    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"\n    It leverages the associative law between convolution and affine transform,\n    i.e., normalize (weight conv feature) = (normalize weight) conv feature.\n    It works for Eval mode of ConvBN blocks during validation, and can be used\n    for **training** as well, but only if one sets `bn.training=False`. It\n     reduces memory footprint and computation cost, at the cost of slightly\n     reduced numerical stability.\n    Args:\n        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.\n        conv (nn.modules.conv._ConvNd): a conv module\n        x (torch.Tensor): Input feature map.\n    '
    assert bn.running_var is not None
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)
    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)
    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)
    target_shape = [-1] + [1] * (conv.weight.ndim - 1)
    if isinstance(conv, nn.modules.conv._ConvTransposeNd):
        target_shape[:2] = [target_shape[1], target_shape[0]]
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(target_shape)
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (bias_on_the_fly - bn.running_mean)
    input = x
    params = {'weight': weight_on_the_fly, 'bias': bias_on_the_fly}
    output = functional_call(conv, params, input)
    return output

@register_graph_pattern(CallModuleVarArgs([nn.modules.batchnorm._BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]), pass_dict=efficient_conv_bn_eval_pass, extra_check=lambda match: not inductor_config.freezing and inductor_config.efficient_conv_bn_eval_fx_passes)
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs):
    if False:
        return 10
    bn_node = match.nodes[0]
    graph = match.graph
    gm = graph.owning_module
    bn_mod = getattr(gm, bn_node.target)
    if not bn_mod.track_running_stats or bn_mod.training:
        return
    if bn_node.args:
        input_node = bn_node.args[0]
    else:
        input_node = bn_node.kwargs['input']
    if input_node.op != 'call_module':
        return
    if not hasattr(gm, input_node.target):
        return
    input_mod = getattr(gm, input_node.target)
    supported_convs = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if not any((isinstance(input_mod, cls) for cls in supported_convs)):
        return
    conv_node = input_node
    if len(conv_node.users) > 1:
        return
    counters['inductor']['efficient_conv_bn_eval'] += 1
    with graph.inserting_before(conv_node):
        conv_get_node = graph.create_node(op='get_attr', target=conv_node.target, name='get_conv')
        bn_get_node = graph.create_node(op='get_attr', target=bn_node.target, name='get_bn')
        if conv_node.args:
            conv_input = conv_node.args[0]
        else:
            conv_input = conv_node.kwargs['input']
        args = (bn_get_node, conv_get_node, conv_input)
        new_node = graph.create_node(op='call_function', target=efficient_conv_bn_eval, args=args, name='efficient_conv_bn_eval')
    bn_node.replace_all_uses_with(new_node)
    graph.erase_node(bn_node)
    graph.erase_node(conv_node)