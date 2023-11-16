import logging
import operator
from collections import defaultdict
from typing import Set
import torch
from torch.fx import GraphModule
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn import Module
from torch.utils._pytree import tree_map
from .common import aot_autograd
from .registry import register_backend
log = logging.getLogger(__name__)

def cloner(t):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(t, torch.Tensor):
        return t.clone()
    else:
        return t

class CudaGraphModule(Module):
    gm: GraphModule
    mutated_inputs: Set[int]

    def __init__(self, gm, mutated_inputs):
        if False:
            print('Hello World!')
        super().__init__()
        self.gm = gm
        self.mutated_inputs = mutated_inputs
    warmed_up = False
    graph = None
    static_inputs = None
    static_outputs = None

    def __call__(self, *args):
        if False:
            return 10
        if self.graph is not None:
            assert len(args) == len(self.static_inputs)
            for (dst, src) in zip(self.static_inputs, args):
                dst.copy_(src)
            self.graph.replay()
            for i in self.mutated_inputs:
                args[i].copy_(self.static_inputs[i])
            return tree_map(cloner, self.static_outputs)
        elif self.warmed_up:
            self.static_inputs = [x.clone() for x in args]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = self.gm(*self.static_inputs)
            self.graph.replay()
            for i in self.mutated_inputs:
                args[i].copy_(self.static_inputs[i])
            return tree_map(cloner, self.static_outputs)
        else:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                r = self.gm(*args)
            torch.cuda.current_stream().wait_stream(stream)
            self.warmed_up = True
            return r

def find_input_mutations(g):
    if False:
        i = 10
        return i + 15

    def meta_fk(meta):
        if False:
            return 10
        return meta['val'] if 'val' in meta else meta['fake_result']
    inputs = defaultdict(set)
    input_idx = 0
    mutated_inputs = set()
    for n in g.nodes:
        if n.op == 'placeholder':
            inputs[StorageWeakRef(meta_fk(n.meta)._typed_storage())].add(input_idx)
            input_idx += 1
        elif n.op == 'call_function':
            if n.target is operator.getitem:
                continue
            schema = n.target._schema
            for (i, arg) in enumerate(schema.arguments):
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    if arg.name not in n.kwargs:
                        continue
                    argument = n.kwargs[arg.name]
                mut_arg = False
                if arg.alias_info:
                    if arg.alias_info.is_write:
                        mut_arg = True
                if mut_arg:
                    mutated_inputs |= inputs[StorageWeakRef(meta_fk(argument.meta)._typed_storage())]
    return mutated_inputs

def apply_cuda_graphs(gm):
    if False:
        for i in range(10):
            print('nop')
    for n in gm.graph.nodes:
        if n.op == 'call_module':
            assert not n.kwargs
            submod = gm.get_submodule(n.target)
            gm.delete_submodule(n.target)
            mutated_inputs = find_input_mutations(submod.graph)
            gm.add_submodule(n.target, CudaGraphModule(submod, mutated_inputs))

def cudagraphs(model, inputs):
    if False:
        i = 10
        return i + 15
    model = partition_cudagraphs(model, inputs)
    apply_cuda_graphs(model)
    return model
aot_cudagraphs = aot_autograd(fw_compiler=cudagraphs, bw_compiler=cudagraphs)
register_backend(name='cudagraphs', compiler_fn=aot_cudagraphs)

def cudagraphs_inner(model, inputs, copy_outputs=True, copy_inputs=True):
    if False:
        return 10
    "This isn't registered as a backend, but is used in some benchmarks"
    assert isinstance(inputs, (list, tuple))
    if copy_inputs:
        static_inputs = [torch.zeros_like(x) for x in inputs]
    else:
        static_inputs = list(inputs)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        if False:
            return 10
        assert len(static_inputs) == len(new_inputs)
        if copy_inputs:
            for (dst, src) in zip(static_inputs, new_inputs):
                dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs
    return run