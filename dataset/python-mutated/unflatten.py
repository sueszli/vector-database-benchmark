import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import ConstantArgument, ModuleCallSignature, SymIntArgument, TensorArgument
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook

def _assign_attr(from_obj: torch.Tensor, to_module: torch.nn.Module, target: str, is_parameter: bool):
    if False:
        print('Hello World!')
    (*prefix, field) = target.split('.')
    for item in prefix:
        t = getattr(to_module, item, None)
        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t
    if not isinstance(from_obj, torch.Tensor):
        raise ValueError('Expected only parameters or buffers, got:', type(from_obj))
    if is_parameter:
        to_module.register_parameter(field, torch.nn.Parameter(from_obj))
    else:
        to_module.register_buffer(field, from_obj)

class _UnflattenedModule(torch.fx.GraphModule):

    def __init__(self, export_module: ExportedProgram):
        if False:
            print('Hello World!')
        if export_module.graph_signature.backward_signature is not None:
            raise ValueError('Unflattening on JointExportModule NYI')
        super().__init__({}, torch.fx.Graph(), '_UnflattenedModule')
        export_graph = deepcopy(export_module.graph)
        self.graph_signature = deepcopy(export_module.graph_signature)
        self.module_call_graph = deepcopy(export_module.module_call_graph)
        _inplace_buffer_mutations(export_graph, self.graph_signature)
        _outline_submodules(export_graph, self)
        self.range_constraints = export_module.range_constraints
        self.equality_constraints = export_module.equality_constraints
        state_dict = export_module.state_dict
        for name in self.graph_signature.parameters:
            cloned = state_dict[name].clone()
            _assign_attr(cloned, self, name, is_parameter=True)
        for name in self.graph_signature.buffers:
            cloned = state_dict[name].clone()
            _assign_attr(cloned, self, name, is_parameter=False)
        inputs_to_state: Dict[str, str] = {**self.graph_signature.inputs_to_parameters, **self.graph_signature.inputs_to_buffers}
        _sink_params(self, inputs_to_state, [])
        for module in self.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != 'placeholder':
                    continue
                assert node.name not in inputs_to_state

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (flat_args, in_spec) = pytree.tree_flatten((args, kwargs))
        assert self.module_call_graph[0].fqn == ''
        signature = self.module_call_graph[0].signature
        if in_spec != signature.in_spec:
            raise TypeError(f"Input treespec does not match with exported module's. Are you sure you are calling this with the right arguments? Input treespec: {in_spec}. ", f'Exported module treespec: {signature.in_spec}')
        tree_out = super().__call__(*flat_args)
        return pytree.tree_unflatten(tree_out, signature.out_spec)

def unflatten(module: ExportedProgram) -> _UnflattenedModule:
    if False:
        for i in range(10):
            print('nop')
    'Unflatten an ExportedProgram, producing a module with the same module\n    hierarchy as the original eager module.\n    '
    module = _UnflattenedModule(module)
    module.register_forward_pre_hook(_check_input_constraints_pre_hook)
    return module

def _inplace_buffer_mutations(graph: torch.fx.Graph, graph_signature) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Transform buffer mutations from their functionalized form into a copy_\n    node in the graph.\n\n    Functionalization represents buffer mutation by passing the buffer as an input and output. So for example, the eager code:\n        def forward(self, x):\n            self.buffer += x\n            return x * x\n\n    Will become a graph that looks like:\n        def forward(self, buffer, x):\n            mutated_buffer = aten.add(buffer, x)\n            mul = aten.mul(x, x)\n            return (mutated_buffer, mul)\n\n    We want to inplace this into something that looks like the original eager code:\n        def forward(self, buffer, x):\n            mutated_buffer = aten.add(buffer, x)\n            buffer.copy_(mutated_buffer)\n            mul = aten.mul(x, x)\n            return (mul,)\n    '
    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == 'output' and len(output_node.args) == 1
    return_args = output_node.args[0]
    mutation_node_to_buffer = graph_signature.buffers_to_mutate
    mutations = return_args[:len(mutation_node_to_buffer)]
    buffers_to_inputs = {v: k for (k, v) in graph_signature.inputs_to_buffers.items()}
    input_name_to_node = {node.name: node for node in graph.nodes if node.op == 'placeholder'}
    for mutation in mutations:
        buffer_name = mutation_node_to_buffer[mutation.name]
        input_name = buffers_to_inputs[buffer_name]
        input_node = input_name_to_node[input_name]
        with graph.inserting_after(mutation):
            new_node = graph.create_node('call_function', torch.ops.aten.copy_, (input_node, mutation))
            for (k, v) in mutation.meta.items():
                new_node.meta[k] = v
        mutation.replace_all_uses_with(new_node, lambda x: x is not new_node)
    user_outputs = tuple(return_args[len(mutation_node_to_buffer):])
    output_node.args = (user_outputs,)

def is_prefix(candidate, target):
    if False:
        i = 10
        return i + 15
    'Check whether `candidate` is a prefix of `target`.'
    return len(candidate) < len(target) and target[:len(candidate)] == candidate

def compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    if False:
        print('Hello World!')
    if parent_fqn == '':
        return child_fqn
    parent_split = parent_fqn.split('.')
    child_split = child_fqn.split('.')
    assert child_split[:len(parent_split)] == parent_split, f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'"
    return '.'.join(child_split[len(parent_split):])

def _verify_graph_equivalence(x: torch.fx.GraphModule, y: torch.fx.GraphModule):
    if False:
        for i in range(10):
            print('nop')

    def graph_dump(graph: torch.fx.Graph) -> str:
        if False:
            i = 10
            return i + 15
        ret = []
        nodes_idx: Dict[int, int] = {}

        def arg_dump(arg) -> str:
            if False:
                while True:
                    i = 10
            if isinstance(arg, torch.fx.Node):
                return '%' + str(nodes_idx[id(arg)])
            return str(arg)
        for (i, node) in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [f'{key}={value}' for (key, value) in pytree.tree_map(arg_dump, node.kwargs).items()]
            target = node.target if node.op == 'call_function' else ''
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return '\n'.join(ret)
    assert graph_dump(x.graph) == graph_dump(y.graph)

def _add_spec(gm: torch.fx.GraphModule, spec) -> str:
    if False:
        print('Hello World!')
    i = 0
    while hasattr(gm, f'_spec_{i}'):
        i += 1
    name = f'_spec_{i}'
    setattr(gm, name, spec)
    return name

def _generate_flatten(gm: torch.fx.GraphModule, node, spec) -> torch.fx.Node:
    if False:
        print('Hello World!')
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))

def _generate_unflatten(gm: torch.fx.GraphModule, nodes, spec) -> torch.fx.Node:
    if False:
        print('Hello World!')
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))

class ModuleFrame:

    def __init__(self, flat_graph, seen_nodes, seen_modules, parent, module_stack, module_id, module_call_graph: Dict[str, ModuleCallSignature], graph_module=None):
        if False:
            for i in range(10):
                print('nop')
        self.flat_graph = flat_graph
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id
        self.module_call_graph = module_call_graph
        self.verbose = False
        self.fqn = self.module_stack[-1]
        if graph_module is not None:
            self.graph_module = graph_module
        else:
            self.graph_module = torch.fx.GraphModule({}, torch.fx.Graph(), self.fqn)
            self.graph_module.meta['module_call_signature'] = module_call_graph.get(self.fqn)
        if self.module_id in self.seen_modules:
            self.cached_graph_module = self.seen_modules[self.module_id]
        else:
            self.cached_graph_module = None
            self.seen_modules[self.module_id] = self.graph_module
        self.nodes = list(self.flat_graph.nodes)
        self.graph = self.graph_module.graph
        self.node_map: Dict[torch.fx.Node, torch.fx.Node] = {}
        self.node_to_placeholder = {}
        self.parent_call_module: Optional[torch.fx.Node] = None
        if parent is not None:
            accessor = compute_accessor(parent.fqn, self.fqn)
            parent.graph_module.add_submodule(accessor, self.graph_module if self.cached_graph_module is None else self.cached_graph_module)
            self.parent_call_module = parent.graph.call_module(accessor)
        signature = module_call_graph.get(self.fqn)
        if signature is not None and self.parent is not None:
            assert len(signature.in_spec.children_specs) == 2
            args_spec = signature.in_spec.children_specs[0]
            kwargs_spec = signature.in_spec.children_specs[1]
            assert args_spec.context is None
            assert kwargs_spec.context is not None
            with self.graph_module.graph.inserting_after(None):
                arg_nodes = []
                for idx in range(len(args_spec.children_specs)):
                    arg_nodes.append(self.graph_module.graph.placeholder(f'_positional_arg_{idx}'))
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph_module.graph.placeholder(name)
                flat_args = _generate_flatten(self.graph_module, (tuple(arg_nodes), kwarg_nodes), signature.in_spec)
                for (idx, arg) in enumerate(signature.inputs):
                    flat_arg_node = self.graph_module.graph.create_node(op='call_function', target=operator.getitem, args=(flat_args, idx), name=arg.name if not isinstance(arg, ConstantArgument) else f'_constant_{idx}')
                    if isinstance(arg, ConstantArgument):
                        continue
                    flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                    self.node_to_placeholder[self.seen_nodes[arg.name]] = flat_arg_node
            with self.parent.graph.inserting_before(self.parent_call_module):
                nodes: List[Optional[torch.fx.Node]] = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument) and input.value is None:
                        nodes.append(None)
                    else:
                        assert isinstance(input, (TensorArgument, SymIntArgument))
                        nodes.append(self.parent.remap_input(self.seen_nodes[input.name]))
                inputs_node = _generate_unflatten(self.parent.graph_module, nodes, signature.in_spec)
                args_node = self.parent.graph.call_function(operator.getitem, (inputs_node, 0))
                kwargs_node = self.parent.graph.call_function(operator.getitem, (inputs_node, 1))
                arg_nodes = [self.parent.graph.call_function(operator.getitem, (args_node, i)) for i in range(len(args_spec.children_specs))]
                kwarg_nodes = {k: self.parent.graph.call_function(operator.getitem, (kwargs_node, k)) for k in kwargs_spec.context}
            assert self.parent_call_module is not None
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes

    def add_placeholder(self, x):
        if False:
            return 10
        assert x.graph is self.flat_graph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def remap_input(self, x):
        if False:
            for i in range(10):
                print('nop')
        assert x.graph is self.flat_graph
        if x in self.node_map:
            return self.node_map[x]
        if x not in self.node_to_placeholder:
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
        return self.node_to_placeholder[x]

    def finalize_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        orig_outputs = []
        signature = self.module_call_graph.get(self.fqn)
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(output, (TensorArgument, SymIntArgument)):
                    orig_outputs.append(self.seen_nodes[output.name])
                else:
                    raise RuntimeError(f'Unsupported data type for output node: {output}')
            tree_out_node = _generate_unflatten(self.graph_module, tuple((self.node_map[self.seen_nodes[output.name]] for output in orig_outputs)), signature.out_spec)
            parent_out: Optional[torch.fx.Node] = _generate_flatten(self.parent.graph_module, self.parent_call_module, signature.out_spec)
            graph_outputs: Union[torch.fx.Node, List[torch.fx.Node]] = tree_out_node
        else:
            graph_outputs = []
            for orig_node in self.node_map.keys():
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break
            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]
        assert isinstance(graph_outputs, (list, torch.fx.Node))
        self.graph.output(graph_outputs)
        self.graph.lint()
        self.graph_module.recompile()
        if parent_out is None:
            return
        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for (i, orig_output) in enumerate(orig_outputs):
                proxy_out = torch.fx.Proxy(parent_out)[i].node
                self.parent.node_map[orig_output] = proxy_out
        if self.cached_graph_module is not None:
            _verify_graph_equivalence(self.cached_graph_module, self.graph_module)

    def copy_node(self, node):
        if False:
            return 10
        self.print('copying', node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        if False:
            while True:
                i = 10
        i = 0
        for node in self.flat_graph.nodes:
            self.print(i, node.meta.get('nn_module_stack'), node.format_node())
            i += 1
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == 'placeholder':
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]
        self.run_from(node_idx)
        for node in self.flat_graph.nodes:
            if node.op == 'output':
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.verbose:
            print(*args, **kwargs)

    def run_from(self, node_idx):
        if False:
            while True:
                i = 10
        module_idx = 0
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            assert node.op != 'placeholder'
            self.print()
            self.print('STEP', node_idx, node.format_node())
            self.print(self.module_stack)
            if node.op == 'output':
                if len(self.module_stack) == 1:
                    return node_idx
                self.finalize_outputs()
                return node_idx
            node_module_stack = [path for (path, ty) in node.meta['nn_module_stack'].values()] if 'nn_module_stack' in node.meta else self.module_stack
            if node_module_stack[:len(self.module_stack)] != self.module_stack:
                self.finalize_outputs()
                self.print('outlining', self.fqn)
                self.print(self.graph)
                return node_idx
            assert node_module_stack is not None
            if is_prefix(self.module_stack, node_module_stack):
                next_module = node_module_stack[len(self.module_stack)]
                self.print('Creating new stack frame for', next_module)
                node_idx = ModuleFrame(self.flat_graph, self.seen_nodes, self.seen_modules, self, self.module_stack + [next_module], list(node.meta['nn_module_stack'].keys())[len(self.module_stack)], self.module_call_graph).run_from(node_idx)
                module_idx += 1
                continue
            assert node_module_stack == self.module_stack
            self.copy_node(node)
            node_idx += 1

def _outline_submodules(orig_graph: torch.fx.Graph, root_module: torch.fx.GraphModule):
    if False:
        return 10
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, torch.nn.Module] = {}
    ModuleFrame(orig_graph, seen_nodes, seen_modules, None, [''], '', {entry.fqn: entry.signature for entry in root_module.module_call_graph if entry.signature}, graph_module=root_module).run_outer()

def _sink_params(module: GraphModule, inputs_to_state: Dict[str, str], scope: List[str]):
    if False:
        print('Hello World!')
    'Sink params and buffers from graph inputs into get_attr nodes.\n\n    Exported modules are purely functional, so they pass their parameters and\n    buffers in as inputs to the graph.\n\n    To replicate eager\'s semantics, we need to get them from the module state\n    via get_attr instead.\n\n    module: GraphModule, potentially containining nested submodules.\n    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.\n    scope: tracks where we are in the module hierarchy, so that we can emit the\n        right `getattr(self, "foo.bar")` calls, etc.\n    '
    for (name, submodule) in module._modules.items():
        _sink_params(cast(GraphModule, submodule), inputs_to_state, scope + [name])
    if not isinstance(module, GraphModule):
        return
    graph = module.graph
    inputs = filter(lambda n: n.op == 'placeholder', graph.nodes)
    call_module_nodes = filter(lambda n: n.op == 'call_module', graph.nodes)
    for node in call_module_nodes:
        node.args = tuple(filter(lambda n: n.name not in inputs_to_state, node.args))
    for node in inputs:
        if node.name not in inputs_to_state:
            continue
        if len(node.users) > 0:
            state_name = inputs_to_state[node.name].split('.')
            if state_name[:len(scope)] != scope:
                continue
            attr_path = state_name[len(scope):]
            state_attr = _recursive_getattr(module, attr_path)
            assert isinstance(state_attr, torch.Tensor)
            with graph.inserting_after(node):
                new_node = graph.create_node('get_attr', '.'.join(attr_path))
            node.replace_all_uses_with(new_node, propagate_meta=True)
        graph.erase_node(node)
    module.recompile()

def _recursive_getattr(obj, attr_path):
    if False:
        i = 10
        return i + 15
    for attr in attr_path:
        obj = getattr(obj, attr)
    return obj