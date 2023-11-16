import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum

def _parent_name(target: str) -> Tuple[str, str]:
    if False:
        print('Hello World!')
    '\n    Splits a qualname into parent path and last atom.\n    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)\n    '
    (*parent, name) = target.rsplit('.', 1)
    return (parent[0] if parent else '', name)

def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if False:
        i = 10
        return i + 15
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for (expected_type, current_node) in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    if False:
        i = 10
        return i + 15
    assert isinstance(node.target, str)
    (parent_name, name) = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)

def fuse(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    if False:
        i = 10
        return i + 15
    '\n    Fuses convolution/BN layers for inference purposes. Will deepcopy your\n    model by default, but can modify the model inplace as well.\n    '
    patterns = [(nn.Conv1d, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d), (nn.Conv3d, nn.BatchNorm3d)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)
    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)

def remove_dropout(model: nn.Module) -> nn.Module:
    if False:
        print('Hello World!')
    '\n    Removes all dropout layers from the module.\n    '
    fx_model = fx.symbolic_trace(model)

    class DropoutRemover(torch.fx.Transformer):

        def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
            if False:
                i = 10
                return i + 15
            if isinstance(self.submodules[target], nn.Dropout):
                assert len(args) == 1
                return args[0]
            else:
                return super().call_module(target, args, kwargs)
    return DropoutRemover(fx_model).transform()

def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], inputs: List[fx.Node], outputs: List[fx.Node]):
    if False:
        return 10
    '\n    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.\n    '
    new_graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}
    for input in inputs:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    new_graph.output([env[output] for output in outputs])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)
mkldnn_supported = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, torch.relu, torch.transpose, torch.sigmoid, F.relu, F.avg_pool2d, F.adaptive_avg_pool2d]
mkldnn_supported_unknown = [operator.add, operator.mul]
mkldnn_map = {nn.Conv2d: th_mkldnn.MkldnnConv2d, nn.Linear: th_mkldnn.MkldnnLinear, nn.BatchNorm2d: lambda a, _: th_mkldnn.MkldnnBatchNorm(a)}

def modules_to_mkldnn(nodes: List[fx.Node], modules: Dict[str, nn.Module]):
    if False:
        print('Hello World!')
    "\n    For each node, if it's a module that can be preconverted into MKLDNN,\n    then we do so and create a mapping to allow us to convert from the MKLDNN\n    version of the module to the original.\n    "
    old_modules: Dict[nn.Module, nn.Module] = {}
    for node in nodes:
        if node.op == 'call_module':
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in mkldnn_map:
                new_module = mkldnn_map[type(cur_module)](cur_module, torch.float)
                assert isinstance(new_module, nn.Module)
                old_modules[new_module] = copy.deepcopy(cur_module)
                replace_node_module(node, modules, new_module)
    return old_modules

def reset_modules(nodes: List[fx.Node], modules: Dict[str, nn.Module], old_modules: Dict[nn.Module, nn.Module]):
    if False:
        return 10
    "\n    Maps each module that's been changed with `modules_to_mkldnn` back to its\n    original.\n    "
    for node in nodes:
        if node.op == 'call_module':
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if cur_module in old_modules:
                replace_node_module(node, modules, old_modules[cur_module])

class MklSubgraph:

    def __init__(self, fx_graph: fx.Graph):
        if False:
            for i in range(10):
                print('nop')
        self.fx_graph = fx_graph
        self.nodes: List[fx.Node] = []
        self.start_nodes: List[fx.Node] = []
        self.end_nodes: List[fx.Node] = []

def gen_mkl_autotuner(example_inputs, iters=10, warmup=1):
    if False:
        return 10
    '\n    This generates a heuristic that can be passed into `optimize_for_inference` that\n    determines whether a subgraph should be run in MKL by running it with the example_inputs.\n\n    Example usage:\n        heuristic = gen_mkl_autotuner(example_inputs, iters=10)\n        fast_model = optimization.optimize_for_inference(model, heuristic)\n    '
    fx_model = None
    old_modules = None

    def use_mkl_heuristic(graph: MklSubgraph) -> bool:
        if False:
            return 10
        nonlocal fx_model, old_modules
        input_nodes = graph.start_nodes
        if fx_model is None:
            fx_model = graph.fx_graph.owning_module
            old_modules = graph.fx_graph.old_modules
            ShapeProp(fx_model).propagate(example_inputs)
        sample_inputs = [torch.randn(node.shape) for node in input_nodes]
        output_args = cast(List[fx.Node], [node.args[0] for node in graph.end_nodes])
        submodule = extract_subgraph(fx_model, graph.nodes, input_nodes, output_args)

        def benchmark(f):
            if False:
                while True:
                    i = 10
            for _ in range(warmup):
                f()
            begin = time.time()
            for _ in range(iters):
                out = f()
            return time.time() - begin
        mkl_time = benchmark(lambda : [i.to_dense() for i in submodule(*[i.to_mkldnn() for i in sample_inputs])])
        reset_modules(submodule.graph.nodes, dict(submodule.named_modules()), old_modules)
        no_mkl_time = benchmark(lambda : submodule(*sample_inputs))
        return mkl_time < no_mkl_time
    return use_mkl_heuristic

def use_mkl_length(graph: MklSubgraph) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    This is a heuristic that can be passed into `optimize_for_inference` that\n    determines whether a subgraph should be run in MKL by checking if there\n    are more than 2 nodes in it\n    '
    return len(graph.nodes) > 2

class UnionFind:

    def __init__(self, n):
        if False:
            return 10
        self.parent: List[Optional[int]] = [None] * n
        self.size: List[int] = [0] * n

    def make_set(self, v: int):
        if False:
            for i in range(10):
                print('nop')
        self.parent[v] = v
        self.size[v] = 1

    def find(self, v: int) -> int:
        if False:
            return 10
        par = self.parent[v]
        if v == par:
            return v
        assert par is not None
        self.parent[v] = self.find(par)
        return cast(int, self.parent[v])

    def join(self, a: int, b: int):
        if False:
            while True:
                i = 10
        (a, b) = (self.find(a), self.find(b))
        if a == b:
            return a
        if self.size[a] < self.size[b]:
            (a, b) = (b, a)
        self.parent[b] = a
        self.size[a] += self.size[b]

def optimize_for_inference(model: torch.nn.Module, pass_config: Optional[Dict[str, Any]]=None, tracer: Type[fx.Tracer]=fx.Tracer) -> torch.nn.Module:
    if False:
        print('Hello World!')
    "\n    Performs a set of optimization passes to optimize a model for the\n    purposes of inference. Specifically, the passes that are run are:\n    1. Conv/BN fusion\n    2. Dropout removal\n    3. MKL layout optimizations\n\n    The third optimization takes a function `use_mkl_heuristic` that's used\n    to determine whether a subgraph should be explicitly run in MKL layout.\n\n    Note: As FX does not currently handle aliasing, this pass currently\n    assumes nothing aliases. If that isn't true, use at your own risk.\n    "
    default_pass_config = {'conv_bn_fuse': True, 'remove_dropout': True, 'mkldnn_layout_optimize': {'heuristic': use_mkl_length}}
    if pass_config is None:
        pass_config = {}
    default_pass_config.update(pass_config)
    if default_pass_config['conv_bn_fuse']:
        model = fuse(model)
    if default_pass_config['remove_dropout']:
        model = remove_dropout(model)
    if default_pass_config['mkldnn_layout_optimize'] is False:
        return model
    if not isinstance(default_pass_config['mkldnn_layout_optimize'], dict):
        raise RuntimeError('mkldnn_layout_optimize config is not a dict')
    if 'heuristic' not in default_pass_config['mkldnn_layout_optimize']:
        raise RuntimeError('Heuristic not found in mkldnn_layout_optimize config')
    use_mkl_heuristic = default_pass_config['mkldnn_layout_optimize']['heuristic']
    cur_tracer = tracer()
    fx_graph = cur_tracer.trace(copy.deepcopy(model))
    fx_model = fx.GraphModule(cur_tracer.root, fx_graph)
    modules: Dict[str, nn.Module] = dict(model.named_modules())

    class MklSupport(Enum):
        NO = 1
        YES = 2
        UNKNOWN = 3
    for node in list(fx_graph.nodes):
        supports_mkldnn = MklSupport.NO
        if node.op == 'call_module':
            cur_module = modules[node.target]
            if type(cur_module) in mkldnn_supported:
                supports_mkldnn = MklSupport.YES
                sample_parameter = next(cur_module.parameters(), None)
                if sample_parameter is not None:
                    assert sample_parameter.dtype == torch.float, 'this pass is only for torch.float modules'
                    assert sample_parameter.device == torch.device('cpu'), 'this pass is only for CPU modules'
        elif node.op == 'call_function':
            if node.target in mkldnn_supported:
                supports_mkldnn = MklSupport.YES
            elif node.target in mkldnn_supported_unknown:
                supports_mkldnn = MklSupport.UNKNOWN
        if supports_mkldnn != MklSupport.NO:
            if supports_mkldnn == MklSupport.UNKNOWN:
                if not any((arg.target == 'to_dense' for arg in node.args)):
                    continue
            with fx_graph.inserting_before(node):
                mkldnn_args = fx.map_arg(node.args, lambda n: fx_graph.call_method('to_mkldnn', (n,)))
            node.args = cast(Tuple[fx.node.Argument], mkldnn_args)
            with fx_graph.inserting_after(node):
                dense_x = fx_graph.create_node('call_method', 'to_dense', (node,))
                node.replace_all_uses_with(dense_x)
                dense_x.args = (node,)
    old_modules = modules_to_mkldnn(list(fx_graph.nodes), modules)
    fx_graph.old_modules = old_modules
    for node in fx_graph.nodes:
        if node.op == 'call_method' and node.target == 'to_dense':
            prv_node = node.args[0]
            users = list(node.users)
            for user in users:
                if user.op == 'call_method' and user.target == 'to_mkldnn':
                    user.replace_all_uses_with(prv_node)
                    fx_graph.erase_node(user)
            if len(node.users) == 0:
                fx_graph.erase_node(node)
    num_nodes = len(fx_graph.nodes)
    uf = UnionFind(num_nodes)

    def get_color(n):
        if False:
            while True:
                i = 10
        if hasattr(n, 'color'):
            return uf.find(n.color)
        if hasattr(n, 'start_color'):
            return uf.find(n.start_color)
        return None
    for (cur_idx, node) in enumerate(fx_graph.nodes):
        if node.op == 'call_method' and node.target == 'to_mkldnn':
            node.start_color = cur_idx
            uf.make_set(cur_idx)
        elif node.op == 'call_method' and node.target == 'to_dense':
            assert get_color(node.args[0]) is not None
            node.end_color = get_color(node.args[0])
        else:
            cur_colors = [get_color(i) for i in node.all_input_nodes if isinstance(i, fx.Node) if get_color(i) is not None]
            if len(cur_colors) == 0:
                continue
            assert not any((i is None for i in cur_colors))
            cur_colors = sorted(cur_colors)
            node.color = cur_colors[0]
            for other_color in cur_colors[1:]:
                uf.join(cur_colors[0], other_color)
    mkldnn_graphs: Dict[int, MklSubgraph] = defaultdict(lambda : MklSubgraph(fx_graph))
    for node in fx_graph.nodes:
        if hasattr(node, 'color'):
            mkldnn_graphs[uf.find(node.color)].nodes.append(node)
        if hasattr(node, 'start_color'):
            mkldnn_graphs[uf.find(node.start_color)].start_nodes.append(node)
        if hasattr(node, 'end_color'):
            mkldnn_graphs[uf.find(node.end_color)].end_nodes.append(node)
    for graph in mkldnn_graphs.values():
        if not use_mkl_heuristic(graph):
            for node in graph.start_nodes + graph.end_nodes:
                prv = node.args[0]
                node.replace_all_uses_with(prv)
                fx_graph.erase_node(node)
            reset_modules(graph.nodes, modules, old_modules)
    mkldnn_conversions = 0
    for node in fx_graph.nodes:
        if node.target == 'to_mkldnn' or node.target == 'to_dense':
            mkldnn_conversions += 1
    logging.getLogger(__name__).info(f'mkldnn conversions: {mkldnn_conversions}')
    fx_graph.lint()
    result = fx.GraphModule(model, fx_graph)
    return result