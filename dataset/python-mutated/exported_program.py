import copy
import dataclasses
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, TYPE_CHECKING, Union
if TYPE_CHECKING:
    import sympy
    from torch.utils._sympy.value_ranges import ValueRanges
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from .graph_signature import _sig_to_specs, ArgumentSpec, ConstantArgument, ExportGraphSignature, InputKind, InputSpec, OutputKind, OutputSpec, SymIntArgument, TensorArgument
__all__ = ['ExportedProgram', 'ModuleCallEntry', 'ModuleCallSignature']
PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]

@dataclasses.dataclass
class ModuleCallSignature:
    inputs: List[ArgumentSpec]
    outputs: List[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None

class ExportedProgram:
    """
    Package of a program from :func:`export`. It contains
    an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing
    tensor values of all lifted parameters and buffers, and various metadata.

    You can call an ExportedProgram like the original callable traced by
    :func:`export` with the same calling convention.

    To perform transformations on the graph, use ``.module`` property to access
    an :class:`torch.fx.GraphModule`. You can then use
    `FX transformation <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    to rewrite the graph. Afterwards, you can simply use :func:`export`
    again to construct a correct ExportedProgram.
    """

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: torch.fx.Graph, graph_signature: ExportGraphSignature, state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]], range_constraints: 'Dict[sympy.Symbol, Any]', equality_constraints: List[Tuple[Any, Any]], module_call_graph: List[ModuleCallEntry], example_inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]=None, verifier: Optional[Type[Any]]=None, tensor_constants: Optional[Dict[str, torch.Tensor]]=None):
        if False:
            while True:
                i = 10
        from torch._export.exported_program import _create_graph_module_for_export
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import InputDim
        graph._codegen = torch.fx.graph.CodeGen()
        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)
        self._graph_signature: ExportGraphSignature = graph_signature
        self._state_dict: Dict[str, Any] = state_dict
        self._range_constraints: 'Dict[sympy.Symbol, ValueRanges]' = range_constraints
        self._equality_constraints: List[Tuple[InputDim, InputDim]] = equality_constraints
        self._module_call_graph: List[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs
        self._tensor_constants = tensor_constants or {}
        from torch._export.verifier import Verifier
        if verifier is None:
            verifier = Verifier
        assert issubclass(verifier, Verifier)
        self._verifier = verifier
        self.verifier().check(self)

    @property
    @compatibility(is_backward_compatible=False)
    def graph_module(self):
        if False:
            print('Hello World!')
        return self._graph_module

    @property
    @compatibility(is_backward_compatible=False)
    def graph(self):
        if False:
            print('Hello World!')
        return self.graph_module.graph

    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self):
        if False:
            while True:
                i = 10
        return self._graph_signature

    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self):
        if False:
            return 10
        return self._state_dict

    @compatibility(is_backward_compatible=False)
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns an iterator over original module's parameters.\n        "
        for (_, param) in self.named_parameters():
            yield param

    @compatibility(is_backward_compatible=False)
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        if False:
            return 10
        '\n        Returns an iterator over original module parameters, yielding\n        both the name of the parameter as well as the parameter itself.\n        '
        for param_name in self.graph_signature.parameters:
            yield (param_name, self.state_dict[param_name])

    @compatibility(is_backward_compatible=False)
    def buffers(self) -> Iterator[torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an iterator over original module buffers.\n        '
        for (_, buf) in self.named_buffers():
            yield buf

    @compatibility(is_backward_compatible=False)
    def named_buffers(self) -> Iterator[Tuple[str, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Returns an iterator over original module buffers, yielding\n        both the name of the buffer as well as the buffer itself.\n        '
        for buffer_name in self.graph_signature.buffers:
            yield (buffer_name, self.state_dict[buffer_name])

    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self):
        if False:
            print('Hello World!')
        return self._range_constraints

    @property
    @compatibility(is_backward_compatible=False)
    def equality_constraints(self):
        if False:
            return 10
        return self._equality_constraints

    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self):
        if False:
            print('Hello World!')
        return self._module_call_graph

    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self):
        if False:
            while True:
                i = 10
        return self._example_inputs

    @property
    @compatibility(is_backward_compatible=False)
    def call_spec(self):
        if False:
            print('Hello World!')
        from torch._export.exported_program import CallSpec
        if len(self.module_call_graph) == 0:
            return CallSpec(in_spec=None, out_spec=None)
        assert self.module_call_graph[0].fqn == ''
        return CallSpec(in_spec=self.module_call_graph[0].signature.in_spec, out_spec=self.module_call_graph[0].signature.out_spec)

    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any:
        if False:
            print('Hello World!')
        return self._verifier

    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._verifier.dialect

    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self):
        if False:
            return 10
        return self._tensor_constants

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        import torch._export.error as error
        from torch._export import combine_args_kwargs
        if self.call_spec.in_spec is not None:
            try:
                user_args = combine_args_kwargs(args, kwargs)
                args = fx_pytree.tree_flatten_spec(user_args, self.call_spec.in_spec, exact_structural_match=True)
            except Exception:
                (_, received_spec) = pytree.tree_flatten(user_args)
                raise TypeError(f'Trying to flatten user inputs with exported input tree spec: \n{self.call_spec.in_spec}\nbut actually got inputs with tree spec of: \n{received_spec}')
        ordered_params = tuple((self.state_dict[name] for name in self.graph_signature.parameters))
        ordered_buffers = tuple((self.state_dict[name] for name in self.graph_signature.buffers))
        if hasattr(self.graph_signature, 'lifted_tensor_constants'):
            ordered_tensor_constants = tuple((self.tensor_constants[name] for name in self.graph_signature.lifted_tensor_constants))
        else:
            ordered_tensor_constants = ()
        self._check_input_constraints(*ordered_params, *ordered_buffers, *ordered_tensor_constants, *args)
        res = torch.fx.Interpreter(self.graph_module).run(*ordered_params, *ordered_buffers, *ordered_tensor_constants, *args, enable_io_processing=False)
        if self.call_spec.out_spec is not None:
            mutation = self.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)
            mutated_buffers = res[:num_mutated]
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
                res = res[:assertion_dep_token_index]
            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)
            except Exception:
                (_, received_spec) = pytree.tree_flatten(res)
                raise error.InternalError(f'Trying to flatten user outputs with exported output tree spec: \n{self.call_spec.out_spec}\nbut actually got outputs with tree spec of: \n{received_spec}')
            finally:
                ix = 0
                for buffer in self.graph_signature.buffers_to_mutate.values():
                    self.state_dict[buffer] = mutated_buffers[ix]
                    ix += 1
        return res

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        graph_module = self.graph_module.print_readable(print_output=False).replace('\n', '\n    ')
        string = f'ExportedProgram:\n    {graph_module}\nGraph signature: {self.graph_signature}\nRange constraints: {self.range_constraints}\nEquality constraints: {self.equality_constraints}\n'
        return string

    def module(self, *, flat: bool=True) -> torch.nn.Module:
        if False:
            print('Hello World!')
        '\n        Returns a self contained GraphModule with all the parameters/buffers inlined.\n        '
        from torch._export.exported_program import unlift_exported_program_lifted_states
        from torch._export.unflatten import unflatten
        if flat:
            return unlift_exported_program_lifted_states(self)
        else:
            return unflatten(self)

    def run_decompositions(self, decomp_table: Optional[Dict[torch._ops.OperatorBase, Callable]]=None) -> 'ExportedProgram':
        if False:
            print('Hello World!')
        '\n        Run a set of decompositions on the exported program and returns a new\n        exported program. By default we will run the Core ATen decompositions to\n        get operators in the\n        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.\n\n        For now, we do not decompose joint graphs.\n        '
        from torch._decomp import core_aten_decompositions
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import _AddRuntimeAssertionsForInlineConstraintsPass, InputDim
        from torch._export.passes.lift_constant_tensor_pass import lift_constant_tensor_pass
        from torch._export.passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
        from torch._functorch.aot_autograd import aot_export_module

        def _get_placeholders(gm):
            if False:
                print('Hello World!')
            placeholders = []
            for node in gm.graph.nodes:
                if node.op != 'placeholder':
                    break
                placeholders.append(node)
            return placeholders
        decomp_table = decomp_table or core_aten_decompositions()
        old_placeholders = _get_placeholders(self.graph_module)
        old_outputs = list(self.graph.nodes)[-1].args[0]
        fake_args = [node.meta['val'] for node in old_placeholders]
        buffers_to_remove = [name for (name, _) in self.graph_module.named_buffers()]
        for name in buffers_to_remove:
            delattr(self.graph_module, name)
        (gm, graph_signature) = aot_export_module(self.graph_module, fake_args, decompositions=decomp_table, trace_joint=False)

        def update_arg(old_arg, new_ph):
            if False:
                return 10
            if isinstance(old_arg, ConstantArgument):
                return old_arg
            elif isinstance(old_arg, TensorArgument):
                return TensorArgument(name=new_ph.name)
            elif isinstance(old_arg, SymIntArgument):
                return SymIntArgument(name=new_ph.name)
            raise RuntimeError(f'Type of old_arg not supported: {type(old_arg)}')
        new_placeholders = _get_placeholders(gm)
        new_outputs = list(gm.graph.nodes)[-1].args[0]
        input_specs = [InputSpec(spec.kind, update_arg(spec.arg, new_placeholders[i]), spec.target) for (i, spec) in enumerate(self.graph_signature.input_specs)]
        output_specs = [OutputSpec(spec.kind, update_arg(spec.arg, new_outputs[i]), spec.target) for (i, spec) in enumerate(self.graph_signature.output_specs)]
        assert len(new_placeholders) == len(old_placeholders)
        old_new_placeholder_map = {old_node.name: new_node.name for (old_node, new_node) in zip(old_placeholders, new_placeholders)}
        new_graph_signature = ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)
        for (old_node, new_node) in zip(old_placeholders, new_placeholders):
            if not isinstance(old_node.meta['val'], torch.Tensor):
                new_node.meta['val'] = old_node.meta['val']
            if new_node.target in new_graph_signature.inputs_to_parameters or new_node.target in new_graph_signature.inputs_to_buffers:
                for (k, v) in old_node.meta.items():
                    new_node.meta[k] = v
        gm.meta.update(self.graph_module.meta)
        new_range_constraints = _get_updated_range_constraints(gm)
        new_equality_constraints = [(InputDim(old_new_placeholder_map[inp_dim1.input_name], inp_dim1.dim), InputDim(old_new_placeholder_map[inp_dim2.input_name], inp_dim2.dim)) for (inp_dim1, inp_dim2) in self.equality_constraints]
        state_dict = self.state_dict.copy()
        lift_constant_tensor_pass(gm, new_graph_signature, state_dict)
        _replace_sym_size_ops_pass(gm)
        exported_program = ExportedProgram(gm, gm.graph, new_graph_signature, state_dict, new_range_constraints, new_equality_constraints, copy.deepcopy(self.module_call_graph), self.example_inputs, self.verifier, self.tensor_constants)
        if len(new_range_constraints) > 0 or len(new_equality_constraints) > 0:
            exported_program = exported_program._transform(_AddRuntimeAssertionsForInlineConstraintsPass(new_range_constraints, new_equality_constraints))
        return exported_program

    def _transform(self, *passes: PassType) -> 'ExportedProgram':
        if False:
            for i in range(10):
                print('nop')
        pm = PassManager(list(passes))
        res = pm(self.graph_module)
        transformed_gm = res.graph_module if res is not None else self.graph_module
        assert transformed_gm is not None
        if transformed_gm is self.graph_module and (not res.modified):
            return self

        def _get_updated_graph_signature(old_signature: ExportGraphSignature, new_gm: torch.fx.GraphModule) -> ExportGraphSignature:
            if False:
                print('Hello World!')
            "\n            Update the graph signature's user_input/user_outputs.\n            "
            new_input_specs = []
            for (i, node) in enumerate(new_gm.graph.nodes):
                if node.op != 'placeholder':
                    break
                assert i < len(old_signature.input_specs), 'Number of inputs changed after transformation'
                old_input_spec = old_signature.input_specs[i]
                arg = old_input_spec.arg if isinstance(old_input_spec.arg, ConstantArgument) else type(old_input_spec.arg)(node.name)
                new_input_specs.append(InputSpec(old_input_spec.kind, arg, old_input_spec.target))
            output_node = list(new_gm.graph.nodes)[-1]
            assert output_node.op == 'output'
            new_output_specs = []
            for (i, node) in enumerate(output_node.args[0]):
                assert i < len(old_signature.output_specs), 'Number of outputs changed after transformation'
                old_output_spec = old_signature.output_specs[i]
                arg = old_output_spec.arg if isinstance(old_output_spec.arg, ConstantArgument) else type(old_output_spec.arg)(node.name)
                new_output_specs.append(OutputSpec(old_output_spec.kind, arg, old_output_spec.target))
            new_signature = ExportGraphSignature(input_specs=new_input_specs, output_specs=new_output_specs)
            return new_signature
        transformed_ep = ExportedProgram(transformed_gm, transformed_gm.graph, _get_updated_graph_signature(self.graph_signature, transformed_gm), self.state_dict, _get_updated_range_constraints(transformed_gm), copy.deepcopy(self.equality_constraints), copy.deepcopy(self._module_call_graph), self.example_inputs, self.verifier, self.tensor_constants)
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        transformed_ep.graph_module.meta.update(res.graph_module.meta)
        return transformed_ep

    def _check_input_constraints(self, *args):
        if False:
            for i in range(10):
                print('nop')
        from torch._export.utils import _check_input_constraints_for_graph
        _check_input_constraints_for_graph(self.graph, self.range_constraints, self.equality_constraints)(*args)

    def _validate(self):
        if False:
            while True:
                i = 10
        self.verifier().check(self)

def _get_updated_range_constraints(gm: torch.fx.GraphModule) -> 'Dict[sympy.Symbol, Any]':
    if False:
        i = 10
        return i + 15

    def get_shape_env(gm):
        if False:
            while True:
                i = 10
        vals = [node.meta['val'] for node in gm.graph.nodes if node.meta.get('val', None) is not None]
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode(vals)
        if fake_mode is not None:
            return fake_mode.shape_env
        for v in vals:
            if isinstance(v, torch.SymInt):
                return v.node.shape_env
    shape_env = get_shape_env(gm)
    if shape_env is None:
        return {}
    range_constraints = {k: v for (k, v) in shape_env.var_to_range.items() if k not in shape_env.replacements}
    return range_constraints