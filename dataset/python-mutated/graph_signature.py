import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
__all__ = ['ConstantArgument', 'ExportBackwardSignature', 'ExportGraphSignature', 'InputKind', 'InputSpec', 'OutputKind', 'OutputSpec', 'SymIntArgument', 'TensorArgument']

@dataclasses.dataclass
class TensorArgument:
    name: str

@dataclasses.dataclass
class SymIntArgument:
    name: str

@dataclasses.dataclass
class ConstantArgument:
    value: Union[int, float, bool, None]
ArgumentSpec = Union[TensorArgument, SymIntArgument, ConstantArgument]

class InputKind(Enum):
    USER_INPUT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    CONSTANT_TENSOR = auto()

@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: Optional[str]

    def __post_init__(self):
        if False:
            print('Hello World!')
        assert isinstance(self.arg, (TensorArgument, SymIntArgument, ConstantArgument))

class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()

@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: Optional[str]

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        assert isinstance(self.arg, (TensorArgument, SymIntArgument, ConstantArgument))

def _sig_to_specs(*, user_inputs: Set[str], inputs_to_parameters: Mapping[str, str], inputs_to_buffers: Mapping[str, str], user_outputs: Set[str], buffer_mutations: Mapping[str, str], grad_params: Mapping[str, str], grad_user_inputs: Mapping[str, str], loss_output: Optional[str], inputs: List[ArgumentSpec], outputs: List[ArgumentSpec]) -> Tuple[List[InputSpec], List[OutputSpec]]:
    if False:
        i = 10
        return i + 15

    def to_input_spec(i: ArgumentSpec) -> InputSpec:
        if False:
            print('Hello World!')
        if not isinstance(i, TensorArgument):
            return InputSpec(kind=InputKind.USER_INPUT, arg=i, target=None)
        name = i.name
        if name in user_inputs:
            return InputSpec(kind=InputKind.USER_INPUT, arg=i, target=None)
        elif name in inputs_to_parameters:
            return InputSpec(kind=InputKind.PARAMETER, arg=i, target=inputs_to_parameters[name])
        elif name in inputs_to_buffers:
            return InputSpec(kind=InputKind.BUFFER, arg=i, target=inputs_to_buffers[name])
        else:
            raise AssertionError(f'Unknown tensor input kind: {name}')

    def to_output_spec(o: ArgumentSpec) -> OutputSpec:
        if False:
            print('Hello World!')
        if not isinstance(o, TensorArgument):
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)
        name = o.name
        if name in user_outputs:
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)
        elif name in buffer_mutations:
            return OutputSpec(kind=OutputKind.BUFFER_MUTATION, arg=o, target=buffer_mutations[name])
        elif name in grad_params:
            return OutputSpec(kind=OutputKind.GRADIENT_TO_PARAMETER, arg=o, target=grad_params[name])
        elif name in grad_user_inputs:
            return OutputSpec(kind=OutputKind.GRADIENT_TO_USER_INPUT, arg=o, target=grad_user_inputs[name])
        elif name == loss_output:
            return OutputSpec(kind=OutputKind.LOSS_OUTPUT, arg=o, target=None)
        else:
            raise AssertionError(f'Unknown tensor output kind: {name}')
    input_specs = [to_input_spec(i) for i in inputs]
    output_specs = [to_output_spec(o) for o in outputs]
    return (input_specs, output_specs)

@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str

@dataclasses.dataclass
class ExportGraphSignature:
    """
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants gurantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    gurantees that parameters, buffers, and constant tensors are lifted out of
    the graph as inputs.  Similarly, any mutations to buffers are not included
    in the graph either, instead the updated values of mutated buffers are
    modeled as additional outputs of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self):
                super(CustomModule, self).__init__()

                # Define a parameter
                self.my_parameter = nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0) # In-place addition

                return output

    Resulting Graph would be::

        graph():
            %arg0_1 := placeholder[target=arg0_1]
            %arg1_1 := placeholder[target=arg1_1]
            %arg2_1 := placeholder[target=arg2_1]
            %arg3_1 := placeholder[target=arg3_1]
            %arg4_1 := placeholder[target=arg4_1]
            %add_tensor := call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %arg0_1), kwargs = {})
            %mul_tensor := call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %arg1_1), kwargs = {})
            %mul_tensor_1 := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %arg2_1), kwargs = {})
            %add_tensor_1 := call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor, %mul_tensor_1), kwargs = {})
            %add_tensor_2 := call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, 1.0), kwargs = {})
            return (add_tensor_2, add_tensor_1)

    Resulting ExportGraphSignature would be::

        ExportGraphSignature(
            input_specs=[
                InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='my_parameter'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg1_1'), target='my_buffer1'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg2_1'), target='my_buffer2'),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg3_1'), target=None),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg4_1'), target=None)
            ],
            output_specs=[
                OutputSpec(kind=<OutputKind.BUFFER_MUTATION: 3>, arg=TensorArgument(name='add_2'), target='my_buffer2'),
                OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add_1'), target=None)
            ]
        )
    """
    input_specs: List[InputSpec]
    output_specs: List[OutputSpec]

    @property
    def parameters(self) -> Collection[str]:
        if False:
            i = 10
            return i + 15
        return [s.target for s in self.input_specs if s.kind == InputKind.PARAMETER if isinstance(s.target, str)]

    @property
    def buffers(self) -> Collection[str]:
        if False:
            print('Hello World!')
        return [s.target for s in self.input_specs if s.kind == InputKind.BUFFER if isinstance(s.target, str)]

    @property
    def lifted_tensor_constants(self) -> Collection[str]:
        if False:
            while True:
                i = 10
        return [s.target for s in self.input_specs if s.kind == InputKind.CONSTANT_TENSOR if isinstance(s.target, str)]

    @property
    def user_inputs(self) -> Collection[str]:
        if False:
            print('Hello World!')
        return tuple((s.arg.name for s in self.input_specs if s.kind == InputKind.USER_INPUT and isinstance(s.arg, TensorArgument)))

    @property
    def user_outputs(self) -> Collection[str]:
        if False:
            print('Hello World!')
        return tuple((s.arg.name for s in self.output_specs if s.kind == OutputKind.USER_OUTPUT and isinstance(s.arg, TensorArgument)))

    @property
    def inputs_to_parameters(self) -> Mapping[str, str]:
        if False:
            i = 10
            return i + 15
        return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.PARAMETER and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def inputs_to_buffers(self) -> Mapping[str, str]:
        if False:
            return 10
        return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.BUFFER and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def buffers_to_mutate(self) -> Mapping[str, str]:
        if False:
            while True:
                i = 10
        return {s.arg.name: s.target for s in self.output_specs if s.kind == OutputKind.BUFFER_MUTATION and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]:
        if False:
            return 10
        return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.CONSTANT_TENSOR and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def backward_signature(self) -> Optional[ExportBackwardSignature]:
        if False:
            i = 10
            return i + 15
        loss_output = None
        gradients_to_parameters: Dict[str, str] = {}
        gradients_to_user_inputs: Dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind == OutputKind.LOSS_OUTPUT:
                assert loss_output is None
                assert isinstance(spec.arg, TensorArgument)
                loss_output = spec.arg.name
            elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_user_inputs[spec.arg.name] = spec.target
        if loss_output is None:
            return None
        return ExportBackwardSignature(loss_output=loss_output, gradients_to_parameters=gradients_to_parameters, gradients_to_user_inputs=gradients_to_user_inputs)

    @property
    def assertion_dep_token(self) -> Optional[Mapping[int, str]]:
        if False:
            for i in range(10):
                print('nop')
        return None

    def __post_init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        assert len(assertion_dep_token) == 1
        assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
        assert len(self.user_outputs) + len(self.buffers_to_mutate) == assertion_dep_token_index

    def replace_all_uses(self, old: str, new: str):
        if False:
            while True:
                i = 10
        '\n        Replace all uses of the old name with new name in the signature.\n        '
        assert isinstance(old, str)
        assert isinstance(new, str)
        for o in self.output_specs:
            if isinstance(o.arg, TensorArgument):
                if o.arg.name == old:
                    o.arg.name = new