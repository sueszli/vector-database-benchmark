import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten
from torch.utils import _pytree as pytree
aten = torch.ops.aten

def get_aten_target(node):
    if False:
        print('Hello World!')
    if hasattr(node.target, 'overloadpacket'):
        return node.target.overloadpacket
    return node.target
rand_ops = [aten.dropout, aten._fused_dropout, aten._standard_gamma, aten.bernoulli, aten.multinomial, aten.native_dropout, aten.normal, aten.poisson, aten.binomial, aten.rrelu, aten.rand_like, aten.rand, aten.randint, aten.randn, aten.randperm]

def fx_graph_cse(fx_g: torch.fx.graph.Graph):
    if False:
        while True:
            i = 10
    new_graph = fx.Graph()
    env = {}
    hash_env = {}
    token_map = {}
    for n in fx_g.nodes:
        if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr' or (get_aten_target(n) in rand_ops):
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:

            def substitute(arg_list):
                if False:
                    for i in range(10):
                        print('nop')
                (arg_list, spec) = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                return (tuple(arg_list), spec)
            (args, args_spec) = substitute(n.args)
            (kwargs, kwargs_spec) = substitute(n.kwargs)
            token = {'target': n.target, 'args': args, 'args_spec': args_spec, 'kwargs': kwargs, 'kwargs_spec': kwargs_spec}
            hash_arg = hash((args, kwargs))
            hash_val = (n.target, hash_arg)
            hash_val_in_hash_env = hash_val in hash_env
            if hash_val_in_hash_env and token_map[hash_val] == token:
                env[n] = hash_env[hash_val]
                continue
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token
    return new_graph

def strip_overloads(gm):
    if False:
        while True:
            i = 10
    '\n    Modifies the target of graph nodes in :attr:`gm` to strip overloads.\n\n    Args:\n        gm(fx.GraphModule): The input Fx graph module to be modified\n    '
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()

def get_placeholders(graph):
    if False:
        return 10
    return list(filter(lambda x: x.op == 'placeholder', graph.nodes))

def get_outputs(graph):
    if False:
        i = 10
        return i + 15
    for node in graph.nodes:
        if node.op == 'output':
            return pytree.tree_leaves(node.args[0])
    raise AssertionError('No output node found')