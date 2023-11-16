from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from .container import NodeInfo
    from .model_speedup import ModelSpeedup
import operator
import torch
from torch.nn import functional as F
from torch.fx.node import Node
from torch.utils._pytree import tree_flatten, tree_unflatten
from .utils import randomize_tensor_inplace, randomize_if_tensor, tree_map_zip, torch_float_dtype, poss_deepcopy

class MaskUpdater:

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        if False:
            i = 10
            return i + 15
        raise RuntimeError('detect method should be overrided!')

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        '\n        preprocesses before direct update sparsity\n        default action:\n            do randomize to node_info.output_origin and store to node_info.output_randomize\n            for submodules, randomize and apply masks to module.named_parameters\n        '
        raise RuntimeError('direct_update_preprocess method should be overrided!')

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            for i in range(10):
                print('nop')
        '\n        main processes to direct update sparsity\n        default action:\n            get all input from node_info.output_randomize and apply the node_info.output_masks;\n            execute the node and get the output;\n            calc the out_mask from the output and store to node_info.output_masks.\n        '
        raise RuntimeError('direct_update_process method should be overrided!')

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            for i in range(10):
                print('nop')
        '\n        post processes after direct update sparsity\n        default action:\n            no action\n        '
        raise RuntimeError('direct_update_postprocess method should be overrided!')

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            i = 10
            return i + 15
        '\n        preprocesses before indirect update sparsity\n        default action:\n            remove all units but maintain struct of node_info.output_origin and store to node_info.output_grad\n            for submodules, do tensor_requires_grad to module.named_parameters\n        '
        raise RuntimeError('indirect_update_preprocess method should be overrided!')

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        '\n        main processes to direct update sparsity\n        default action:\n            calc the out_mask from the node_info.output_grad and store to node_info.output_masks.\n            get all input from node_info.output_origin, randomize it, apply the node_info.output_masks, and do tensor_requires_grad;\n            execute the node and get the output;\n            do backward to output, and for each input, store the grad to node_info.output_grad;\n            for each named_parameters in submodules, update param_masks_1 from grad.\n        '
        raise RuntimeError('indirect_update_process method should be overrided!')

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            return 10
        '\n        post processes after indirect update sparsity\n        default action:\n            no action\n        '
        raise RuntimeError('indirect_update_postprocess method should be overrided!')

class DefaultMaskUpdater(MaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        if False:
            return 10
        '\n        Return true to every node.\n        '
        return True

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            for i in range(10):
                print('nop')
        '\n        Do randomize to node_info.output_origin and store to node_info.output_randomize\n        '
        node_info = model_speedup.node_infos[node]
        (batch_dim, batch_size) = (model_speedup.batch_dim, model_speedup.batch_size)
        node_info.output_randomize = tree_map_zip(lambda t: randomize_if_tensor(t, batch_dim, batch_size), node_info.output_origin)

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        '\n        Get all input from node_info.output_randomize and execute the node,\n        calc the output_masks and store to node_info.output_masks\n        '
        node_info = model_speedup.node_infos[node]
        with torch.no_grad():
            args = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_randomize if isinstance(nd, Node) else nd, node.args)
            args_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node.args)
            args = tree_map_zip(lambda t, m: (t * m).type_as(t) if m is not None else t, args, args_masks)
            kwargs = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_randomize if isinstance(nd, Node) else nd, node.kwargs)
            kwargs_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node.kwargs)
            kwargs = tree_map_zip(lambda t, m: (t * m).type_as(t) if m is not None else t, kwargs, kwargs_masks)
            output = getattr(model_speedup, node.op)(node.target, args, kwargs)
            if node_info.output_masks is not None:
                calc_masks = tree_map_zip(model_speedup.direct_calc_mask, output, node_info.output_masks)
            else:
                calc_masks = tree_map_zip(model_speedup.direct_calc_mask, output)
            node_info.output_masks = calc_masks
        if model_speedup.garbage_collect_values:
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete].output_randomize

    def direct_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            i = 10
            return i + 15
        pass

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        node_info = model_speedup.node_infos[node]
        node_info.output_grad = tree_map_zip(lambda x: None, node_info.output_origin)

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            i = 10
            return i + 15
        node_info = model_speedup.node_infos[node]
        (batch_dim, batch_size) = (model_speedup.batch_dim, model_speedup.batch_size)
        node_info.output_masks = tree_map_zip(model_speedup.indirect_calc_mask, node_info.output_grad, node_info.output_masks)

        def randomize_inputs(node_args):
            if False:
                i = 10
                return i + 15
            args = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_origin if isinstance(nd, Node) else nd, node_args)
            args = tree_map_zip(lambda t: randomize_if_tensor(t, batch_dim, batch_size), args)
            args_masks = tree_map_zip(lambda nd: model_speedup.node_infos[nd].output_masks if isinstance(nd, Node) else None, node_args)
            args = tree_map_zip(lambda t, m: (t * m).type_as(t) if m is not None else t, args, args_masks)

            def require_grad_(obj):
                if False:
                    print('Hello World!')
                if isinstance(obj, torch.Tensor) and model_speedup.tensor_propagate_check(obj) and (obj.dtype in torch_float_dtype):
                    obj.requires_grad_(True)
                return obj
            return tree_map_zip(lambda t: require_grad_(t), args)
        args = randomize_inputs(node.args)
        kwargs = randomize_inputs(node.kwargs)
        args_cloned = tree_map_zip(lambda t: t.clone() if isinstance(t, torch.Tensor) else poss_deepcopy(t), args)
        kwargs_cloned = tree_map_zip(lambda t: t.clone() if isinstance(t, torch.Tensor) else poss_deepcopy(t), kwargs)
        output = getattr(model_speedup, node.op)(node.target, args_cloned, kwargs_cloned)
        tree_map_zip(model_speedup.indirect_backward, output, node_info.output_masks)

        def indirect_pass_grad(nodes, args):
            if False:
                return 10
            if nodes is None:
                return
            elif isinstance(nodes, (list, tuple)):
                assert isinstance(args, (list, tuple))
                for (x, y) in zip(nodes, args):
                    indirect_pass_grad(x, y)
            elif isinstance(nodes, dict):
                assert isinstance(args, dict)
                for (x, y) in zip(nodes.values(), args.values()):
                    indirect_pass_grad(x, y)
            elif isinstance(nodes, Node):
                model_speedup.indirect_pass_grad(nodes, args)
            else:
                assert not isinstance(args, torch.Tensor)
        indirect_pass_grad(node.args, args)
        indirect_pass_grad(node.kwargs, kwargs)

    def indirect_update_postprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        pass

class LeafModuleMaskUpdater(DefaultMaskUpdater):

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        the default MaskUpdater for leaf module, so return true if the node is a module calling\n        '
        if node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            param_masks = model_speedup.masks.get(node.target, {})
            for (k, v) in module.named_parameters():
                if k not in param_masks:
                    param_masks[k] = torch.ones_like(v)
            model_speedup.node_infos[node].module = module
            model_speedup.node_infos[node].param_masks = param_masks
            return True
        else:
            return False

    def direct_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            return 10
        super().direct_update_preprocess(model_speedup, node)
        with torch.no_grad():
            node_info: 'NodeInfo' = model_speedup.node_infos[node]
            for (k, v) in node_info.module.named_parameters():
                randomize_tensor_inplace(v)
                v *= node_info.param_masks[k]

    def indirect_update_preprocess(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            for i in range(10):
                print('nop')
        super().indirect_update_preprocess(model_speedup, node)
        node_info: 'NodeInfo' = model_speedup.node_infos[node]
        for (_, v) in node_info.module.named_parameters():
            if isinstance(v, torch.Tensor) and model_speedup.tensor_propagate_check(v) and (v.dtype in torch_float_dtype):
                v.requires_grad_(True)

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        super().indirect_update_process(model_speedup, node)
        node_info: 'NodeInfo' = model_speedup.node_infos[node]
        for (k, v) in node_info.module.named_parameters():
            if isinstance(v, torch.Tensor) and model_speedup.tensor_propagate_check(v) and (v.dtype in torch_float_dtype):
                grad_zero = v.grad.data == 0
                node_info.param_masks[k][grad_zero] = 0

class NoMaskUpdater(DefaultMaskUpdater):
    """
    For some ops that will not produce masks.
    """

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        if False:
            return 10
        '\n        the default MaskUpdater for operators that will not change mask value\n        '
        if node.op == 'call_function':
            if node.target in (len, operator.is_, operator.is_not, operator.contains):
                return True
        elif node.op == 'call_method':
            if isinstance(node.args[0], Node) and isinstance(model_speedup.node_infos[node.args[0]].output_origin, torch.Tensor):
                if node.target in ('dim', 'size'):
                    return True
        return False

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        '\n        get all input from node_info.output_randomize and execute the node\n        calc the out_mask and store to node_info.output_masks\n        '
        with torch.no_grad():
            model_speedup.node_infos[node].output_masks = tree_map_zip(lambda t: None, model_speedup.node_infos[node].output_origin)
        if model_speedup.garbage_collect_values:
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete]._output_randomize
no_change_act_func = (F.relu, F.relu_, F.hardtanh, F.hardtanh_, F.hardswish, F.relu6, F.elu, F.elu_, F.selu, F.celu, F.leaky_relu, F.leaky_relu_, F.rrelu, F.rrelu_, F.gelu, F.logsigmoid, F.hardshrink, F.tanhshrink, F.softsign, F.softplus, F.softmin, F.softmax, F.softshrink, F.gumbel_softmax, F.log_softmax, F.tanh, F.sigmoid, F.hardsigmoid, F.silu, F.mish)
no_change_act_module = (torch.nn.Softmin, torch.nn.Softmax, torch.nn.Softmax2d, torch.nn.LogSoftmax, torch.nn.ELU, torch.nn.Hardshrink, torch.nn.Hardsigmoid, torch.nn.Hardtanh, torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.LogSigmoid, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.RReLU, torch.nn.SELU, torch.nn.CELU, torch.nn.GELU, torch.nn.Sigmoid, torch.nn.SiLU, torch.nn.Mish, torch.nn.Softplus, torch.nn.Softshrink, torch.nn.Softsign, torch.nn.Tanh, torch.nn.Tanhshrink, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LazyBatchNorm1d, torch.nn.LazyBatchNorm2d, torch.nn.LazyBatchNorm3d, torch.nn.GroupNorm, torch.nn.SyncBatchNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LazyInstanceNorm1d, torch.nn.LazyInstanceNorm2d, torch.nn.LazyInstanceNorm3d, torch.nn.LayerNorm)

class NoChangeMaskUpdater(DefaultMaskUpdater):
    """
    for some special op that masks will not change when execute
    1. for getitem op, it's no need to calc masks. do in fast path to run the algorithm faster.
    2. for (softmax, log_softmax) ops, the default process will get a wrong mask. actually we should just copy the mask from input to
        output.
    """

    def __init__(self, customized_no_change_act_module: Tuple | None=None, customized_no_change_act_func: Tuple | None=None):
        if False:
            print('Hello World!')
        self.no_change_act_module = no_change_act_module if not customized_no_change_act_module else no_change_act_module + customized_no_change_act_module
        self.no_change_act_func = no_change_act_func if not customized_no_change_act_func else no_change_act_func + customized_no_change_act_func

    def direct_activation(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            print('Hello World!')
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']
        input_mask = model_speedup.node_infos[input_node].output_masks
        model_speedup.node_infos[node].output_masks = tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else poss_deepcopy(t), input_mask)

    def indirect_activation(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            i = 10
            return i + 15
        if len(node.args) != 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs['input']
        input_grad = tree_map_zip(lambda t, m: (t * m).type_as(t) if isinstance(m, torch.Tensor) else t, model_speedup.node_infos[node].output_grad, model_speedup.node_infos[node].output_masks)
        dummy_input = torch.rand_like(input_grad)
        dummy_input.grad = input_grad
        model_speedup.indirect_pass_grad(input_node, dummy_input)

    def direct_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        assert len(node.args) == 2
        arg_0_masks = model_speedup.node_infos[node.args[0]].output_masks
        arg_1_val = model_speedup.node_infos[node.args[1]].output_randomize if isinstance(node.args[1], Node) else node.args[1]
        sub_mask = operator.getitem(arg_0_masks, arg_1_val)
        model_speedup.node_infos[node].output_masks = tree_map_zip(lambda t: t.clone().detach() if isinstance(t, torch.Tensor) else poss_deepcopy(t), sub_mask)

    def indirect_getitem(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            print('Hello World!')
        assert len(node.args) == 2
        input_grad = tree_map_zip(lambda t, m: (t * m).type_as(t) if isinstance(m, torch.Tensor) else t, model_speedup.node_infos[node].output_grad, model_speedup.node_infos[node].output_masks)
        arg_1_val = model_speedup.node_infos[node.args[1]].output_randomize if isinstance(node.args[1], Node) else node.args[1]
        input_node_info = model_speedup.node_infos[node.args[0]]
        (flat_args, spec) = tree_flatten(input_node_info.output_grad)
        flat_grads = [None for _ in range(len(flat_args))]
        flat_grads[arg_1_val] = input_grad
        input_grads = tree_unflatten(flat_grads, spec)

        def add_grad(grad, input_grad):
            if False:
                print('Hello World!')
            if isinstance(input_grad, torch.Tensor):
                if grad is not None and input_grad is not None:
                    return grad + input_grad
                elif grad is None:
                    return input_grad
                else:
                    return grad
            else:
                return grad
        model_speedup.node_infos[node].output_grad = tree_map_zip(add_grad, model_speedup.node_infos[node.args[0]].output_grad, input_grads)

    def detect(self, model_speedup: 'ModelSpeedup', node: Node) -> bool:
        if False:
            i = 10
            return i + 15
        return self.detect_helper(model_speedup, node) is not None

    def detect_helper(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        if node.op == 'call_function':
            if node.target in self.no_change_act_func:
                return (self.direct_activation, self.indirect_activation)
            elif node.target == operator.getitem:
                if isinstance(node.args[0], Node) and type(model_speedup.node_infos[node.args[0]].output_origin) in (tuple, list, dict):
                    return (self.direct_getitem, self.indirect_getitem)
        elif node.op == 'call_module':
            module: torch.nn.Module = model_speedup.fetch_attr(node.target)
            if isinstance(module, self.no_change_act_module):
                return (self.direct_activation, self.indirect_activation)
        elif node.op == 'call_method':
            if isinstance(node.args[0], Node) and isinstance(model_speedup.node_infos[node.args[0]].output_origin, torch.Tensor):
                if node.target in ('clone', 'detach'):
                    return (self.direct_activation, self.indirect_activation)
        return None

    def direct_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            for i in range(10):
                print('nop')
        '\n        get all input from node_info.output_randomize and execute the node\n        calc the out_mask and store to node_info.output_masks\n        '
        (direct_fn, _) = self.detect_helper(model_speedup, node)
        with torch.no_grad():
            direct_fn(model_speedup, node)
        if model_speedup.garbage_collect_values:
            for to_delete in model_speedup.user_to_last_uses.get(node, []):
                del model_speedup.node_infos[to_delete].output_randomize

    def indirect_update_process(self, model_speedup: 'ModelSpeedup', node: Node):
        if False:
            while True:
                i = 10
        node_info = model_speedup.node_infos[node]
        node_info.output_masks = tree_map_zip(model_speedup.indirect_calc_mask, node_info.output_grad, node_info.output_masks)
        (_, indirect_fn) = self.detect_helper(model_speedup, node)
        indirect_fn(model_speedup, node)