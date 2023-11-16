import inspect
from typing import Dict, List, Union
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import registration

class _TorchSchema:

    def __init__(self, schema: Union[_C.FunctionSchema, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(schema, _C.FunctionSchema):
            self.name: str = schema.name
            self.overload_name: str = schema.overload_name
            self.arguments: List[str] = [arg.name for arg in schema.arguments]
            self.optional_arguments: List[str] = []
            self.returns: List[str] = [ret.name for ret in schema.returns]
            self.opsets: List[int] = []
        else:
            self.name = schema
            self.overload_name = ''
            self.arguments = []
            self.optional_arguments = []
            self.returns = []
            self.opsets = []

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        s = f'{self.name}.{self.overload_name}(' + ', '.join(self.arguments) + ') -> (' + ', '.join(self.returns) + ')' + ' in opsets ' + ', '.join((str(opset) for opset in self.opsets))
        return s

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, _TorchSchema):
            return False
        return self.name == other.name

    def is_aten(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.name.startswith('aten::')

    def is_backward(self) -> bool:
        if False:
            while True:
                i = 10
        return 'backward' in self.name

def _symbolic_argument_count(func):
    if False:
        i = 10
        return i + 15
    params = []
    signature = inspect.signature(func)
    optional_params = []
    for (name, parameter) in signature.parameters.items():
        if name in {'_outputs', 'g'}:
            continue
        if parameter.default is parameter.empty:
            optional_params.append(parameter)
        else:
            params.append(str(parameter))
    return params

def all_forward_schemas() -> Dict[str, _TorchSchema]:
    if False:
        while True:
            i = 10
    'Returns schemas for all TorchScript forward ops.'
    torch_schemas = [_TorchSchema(s) for s in _C._jit_get_all_schemas()]
    return {schema.name: schema for schema in torch_schemas if not schema.is_backward()}

def all_symbolics_schemas() -> Dict[str, _TorchSchema]:
    if False:
        print('Hello World!')
    'Returns schemas for all onnx supported ops.'
    symbolics_schemas = {}
    for name in registration.registry.all_functions():
        func_group = registration.registry.get_function_group(name)
        assert func_group is not None
        symbolics_schema = _TorchSchema(name)
        func = func_group.get(_constants.ONNX_MAX_OPSET)
        if func is not None:
            symbolics_schema.arguments = _symbolic_argument_count(func)
            symbolics_schema.opsets = list(range(func_group.get_min_supported(), _constants.ONNX_MAX_OPSET + 1))
        else:
            func = func_group.get(7)
            symbolics_schema.arguments = _symbolic_argument_count(func)
            symbolics_schema.opsets = list(range(7, _constants.ONNX_BASE_OPSET))
        symbolics_schemas[name] = symbolics_schema
    return symbolics_schemas