from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
T = TypeVar('T')
MAX_RAW_TENSOR_SIZE = 16

class InflatableArg(NamedTuple):
    """Helper type for bundled inputs.

    'value' is the compressed/deflated input that is stored in the model. Value
    must be of the same type as the argument to the function that it is a deflated
    input for.

    'fmt' is a formatable code string that is executed to inflate the compressed data into
    the appropriate input. It can use 'value' as an input to the format str. It must result
    in a value of the same type as 'value'.

    'fmt_fn' is a formatable function code string that is executed to inflate the compressed
    data into the appropriate input. It must result in a value of the same type as 'value'.
    The function name should be the formatable part of the string.

    Note: Only top level InflatableArgs can be inflated. i.e. you cannot place
    an inflatable arg inside of some other structure. You should instead create
    an inflatable arg such that the fmt code string returns the full structure
    of your input.
    """
    value: Any
    fmt: str = '{}'
    fmt_fn: str = ''

def bundle_inputs(model: torch.jit.ScriptModule, inputs: Union[Optional[Sequence[Tuple[Any, ...]]], Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]]], info: Optional[Union[List[str], Dict[Callable, List[str]]]]=None, *, _receive_inflate_expr: Optional[List[str]]=None) -> torch.jit.ScriptModule:
    if False:
        return 10
    "Create and return a copy of the specified model with inputs attached.\n\n    The original model is not mutated or changed in any way.\n\n    Models with bundled inputs can be invoked in a uniform manner by\n    benchmarking and code coverage tools.\n\n    If inputs is passed in as a list then the inputs will be bundled for 'forward'.\n    If inputs is instead passed in as a map then all the methods specified in the map\n    will have their corresponding inputs bundled. Info should match watchever type is\n    chosen for the inputs.\n\n    The returned model will support the following methods:\n\n        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`\n            Returns a list of tuples suitable for passing to the model like\n            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`\n\n        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`\n            Returns a dictionary mapping function names to a metadata dictionary.\n            This nested dictionary maps preset strings like:\n                'get_inputs_function_name' -> the name of a function attribute in this model that can be\n                    run to get back a list of inputs corresponding to that function.\n                'info' -> the user provided extra information about the bundled inputs\n\n    If forward has bundled inputs then these following functions will also be defined on the returned module:\n\n        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`\n            Returns a list of tuples suitable for passing to the model like\n            `for inp in model.get_all_bundled_inputs(): model(*inp)`\n\n        `get_num_bundled_inputs() -> int`\n            Equivalent to `len(model.get_all_bundled_inputs())`,\n            but slightly easier to call from C++.\n\n    Inputs can be specified in one of two ways:\n\n      - The model can define `_generate_bundled_inputs_for_<function_name>`.\n        If the user chooses this method inputs[<function>] should map to None\n\n      - The `inputs` argument to this function can be a dictionary mapping functions to a\n        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.\n        Alternatively if only bundling inputs for forward the map can be omitted and a singular list of inputs\n        can be provided instead.\n\n        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a\n        list of inputs, the inner tuple is the list of args that together make up one input.\n        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...\n        is the actual data that makes up the args, e.g. a tensor.\n\n    Info is an optional parameter that maps functions to a list of strings providing extra information about that\n    function's bundled inputs. Alternatively if only bundling inputs for forward the map can be omitted and\n    a singular list of information can be provided instead. This could be descriptions, expected outputs, etc.\n        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}\n\n    This function will attempt to optimize arguments so that (e.g.)\n    arguments like `torch.zeros(1000)` will be represented compactly.\n    Only top-level arguments will be optimized.\n    Tensors in lists or tuples will not.\n    "
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception('Only ScriptModule is supported.')
    (ignored_methods, ignored_attrs) = _get_bundled_inputs_attributes_and_methods(model)
    clone = torch._C._hack_do_not_use_clone_module_with_class(model._c, ignored_methods, ignored_attrs)
    cloned_module = wrap_cpp_module(clone)
    if isinstance(inputs, dict):
        assert isinstance(info, dict) or info is None
        augment_many_model_functions_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    else:
        assert isinstance(info, list) or info is None
        augment_model_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    return cloned_module

def augment_model_with_bundled_inputs(model: torch.jit.ScriptModule, inputs: Optional[Sequence[Tuple[Any, ...]]]=None, _receive_inflate_expr: Optional[List[str]]=None, info: Optional[List[str]]=None, skip_size_check=False) -> None:
    if False:
        while True:
            i = 10
    "Add bundled sample inputs to a model for the forward function.\n\n    Models with bundled inputs can be invoked in a uniform manner by\n    benchmarking and code coverage tools.\n\n    Augmented models will support the following methods:\n\n        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`\n            Returns a list of tuples suitable for passing to the model like\n            `for inp in model.get_all_bundled_inputs(): model(*inp)`\n\n        `get_num_bundled_inputs() -> int`\n            Equivalent to `len(model.get_all_bundled_inputs())`,\n            but slightly easier to call from C++.\n\n        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`\n            Returns a dictionary mapping function names to a metadata dictionary.\n            This nested dictionary maps preset strings like:\n                'get_inputs_function_name' -> the name of a function attribute in this model that can be\n                    run to get back a list of inputs corresponding to that function.\n                'info' -> the user provided extra information about the bundled inputs\n\n    Inputs can be specified in one of two ways:\n\n      - The model can define `_generate_bundled_inputs_for_forward`.\n        If the user chooses this method inputs should be None\n\n      - `inputs` is a list of inputs of form List[Tuple[Any, ...]]. A list of tuples where the elements\n        of each tuple are the args that make up one input.\n    "
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception('Only ScriptModule is supported.')
    forward: Callable = model.forward
    if not hasattr(forward, '__name__'):
        forward.__name__ = 'forward'
    augment_many_model_functions_with_bundled_inputs(model, inputs={forward: inputs}, _receive_inflate_expr=_receive_inflate_expr, info={forward: info} if info else None, skip_size_check=skip_size_check)

def augment_many_model_functions_with_bundled_inputs(model: torch.jit.ScriptModule, inputs: Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]], _receive_inflate_expr: Optional[List[str]]=None, info: Optional[Dict[Callable, List[str]]]=None, skip_size_check=False) -> None:
    if False:
        return 10
    "Add bundled sample inputs to a model for an arbitrary list of public functions.\n\n    Models with bundled inputs can be invoked in a uniform manner by\n    benchmarking and code coverage tools.\n\n    Augmented models will support the following methods:\n\n        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`\n            Returns a list of tuples suitable for passing to the model like\n            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`\n\n        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`\n            Returns a dictionary mapping function names to a metadata dictionary.\n            This nested dictionary maps preset strings like:\n                'get_inputs_function_name' -> the name of a function attribute in this model that can be\n                    run to get back a list of inputs corresponding to that function.\n                'info' -> the user provided extra information about the bundled inputs\n\n    If forward has bundled inputs then these following functions are also defined:\n\n        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`\n            Returns a list of tuples suitable for passing to the model like\n            `for inp in model.get_all_bundled_inputs(): model(*inp)`\n\n        `get_num_bundled_inputs() -> int`\n            Equivalent to `len(model.get_all_bundled_inputs())`,\n            but slightly easier to call from C++.\n\n    Inputs can be specified in one of two ways:\n\n      - The model can define `_generate_bundled_inputs_for_<function_name>`.\n        If the user chooses this method inputs[<function>] should map to None\n\n      - The `inputs` argument to this function can be a dictionary mapping functions to a\n        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.\n        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a\n        list of inputs, the inner tuple is the list of args that together make up one input.\n        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...\n        is the actual data that makes up the args, e.g. a tensor.\n\n    Info is an optional parameter that maps functions to a list of strings providing extra information about that\n    function's bundled inputs. This could be descriptions, expected outputs, etc.\n        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}\n\n    This function will attempt to optimize arguments so that (e.g.)\n    arguments like `torch.zeros(1000)` will be represented compactly.\n    Only top-level arguments will be optimized.\n    Tensors in lists or tuples will not.\n    "
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception('Only ScriptModule is supported.')
    if not inputs:
        raise Exception('Please provide inputs for at least 1 function')
    if hasattr(model, 'get_all_bundled_inputs') or hasattr(model, 'get_bundled_inputs_functions_and_info'):
        raise Exception("Models can only be augmented with bundled inputs once. This Model seems to have already been augmented with bundled inputs. Please start afresh with one that doesn't have bundled inputs.")
    get_bundled_inputs_functions_and_info_template = ''
    for (function, input_list) in inputs.items():
        if hasattr(function, '__name__'):
            function_name = function.__name__
        elif hasattr(function, 'name'):
            function_name = function.name
        else:
            raise Exception('At least one of your functions has no attribute name please ensure all have one. m.foo.name = "foo"')
        if input_list is not None and (not isinstance(input_list, Sequence)):
            raise TypeError(f'Error inputs for function {function_name} is not a Sequence')
        function_arg_types = [arg.type for arg in function.schema.arguments[1:]]
        deflated_inputs_type: ListType = ListType(TupleType(function_arg_types))
        model._c._register_attribute(f'_bundled_inputs_deflated_{function_name}', deflated_inputs_type, [])
        if hasattr(model, '_generate_bundled_inputs_for_' + function_name):
            if input_list is not None:
                raise Exception('inputs[{name}] is not None, but _generate_bundled_inputs_for_{name} is already defined'.format(name=function_name))
        elif input_list is None or len(input_list) == 0:
            raise Exception('inputs for {name} must be specified if _generate_bundled_inputs_for_{name} is not already defined'.format(name=function_name))
        else:
            deflated_inputs = []
            parts = []
            for (inp_idx, args) in enumerate(input_list):
                if not isinstance(args, Tuple) and (not isinstance(args, List)):
                    raise TypeError(f'Error bundled input for function {function_name} idx: {inp_idx} is not a Tuple or a List')
                deflated_args = []
                parts.append('(')
                for (arg_idx, arg) in enumerate(args):
                    inflate_helper_fn_name = _get_inflate_helper_fn_name(arg_idx, inp_idx, function_name)
                    (deflated, inflater, helper_definition) = _inflate_expr(arg, f'deflated[{inp_idx}][{arg_idx}]', inflate_helper_fn_name, skip_size_check=skip_size_check)
                    deflated_args.append(deflated)
                    parts.append(f'    {inflater},')
                    if helper_definition:
                        model.define(textwrap.dedent(helper_definition))
                deflated_inputs.append(tuple(deflated_args))
                parts.append('),')
            parts.append('')
            expr = '\n'.join(parts)
            if _receive_inflate_expr is not None:
                _receive_inflate_expr.append(expr)
            setattr(model, f'_bundled_inputs_deflated_{function_name}', deflated_inputs)
            definition = textwrap.dedent('\n                def _generate_bundled_inputs_for_{name}(self):\n                    deflated = self._bundled_inputs_deflated_{name}\n                    return [\n                {expr}\n                    ]\n                ').format(expr=expr, name=function_name)
            model.define(definition)
        model.define(textwrap.dedent('\n            def get_all_bundled_inputs_for_{name}(self):\n                all_inputs = self._generate_bundled_inputs_for_{name}()\n                assert all_inputs is not None\n                return all_inputs\n            ').format(name=function_name))
        inputs_info = repr(info[function]) if info and function in info else '[]'
        get_bundled_inputs_functions_and_info_template += f"\n            temp_dict : Dict[str,List[str]] = {{}}\n            info: List[str] = {inputs_info}\n\n            temp_dict['info'] = info\n            temp_dict['get_inputs_function_name'] = ['get_all_bundled_inputs_for_{function_name}']\n            all_inputs['{function_name}'] = temp_dict\n            "
        if function_name == 'forward':
            model.define(textwrap.dedent('\n                def get_all_bundled_inputs(self):\n                    return self.get_all_bundled_inputs_for_forward()\n                '))
            model.define(textwrap.dedent('\n                def get_num_bundled_inputs(self):\n                    return len(self.get_all_bundled_inputs_for_forward())\n                '))
    model.define(textwrap.dedent(f'\n        def get_bundled_inputs_functions_and_info(self):\n            all_inputs : Dict[str, Dict[str,List[str]]] = {{}}\n            {get_bundled_inputs_functions_and_info_template}\n            return all_inputs\n        '))

def _inflate_expr(arg: T, ref: str, inflate_helper_fn_name: str, skip_size_check: bool=False) -> Tuple[Union[T, torch.Tensor], str, Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(arg, InflatableArg):
        if arg.fmt_fn:
            if arg.fmt not in ['{}', '']:
                raise Exception(f"Bundled input argument at position '{ref}' has both arg.fmt_fn => \n{arg.fmt_fn} \n and arg.fmt  => {arg.fmt}. Please choose `arg.fmt` if the deflater is straightforward or `arg.fmt_fn` if you need a function.")
            helper_definition = arg.fmt_fn.format(inflate_helper_fn_name)
            expr = f'self.{inflate_helper_fn_name}({ref})'
            return (arg.value, expr, helper_definition)
        else:
            return (arg.value, arg.fmt.format(ref), None)
    if isinstance(arg, torch.Tensor):
        if arg._typed_storage().size() <= MAX_RAW_TENSOR_SIZE or skip_size_check:
            return (arg, ref, None)
        if arg.is_contiguous() and arg.numel() <= MAX_RAW_TENSOR_SIZE:
            return (arg.clone(), ref, None)
        for fmt in [torch.contiguous_format, torch.channels_last]:
            if arg.is_contiguous(memory_format=fmt) and (arg == arg.flatten()[0]).all().item():
                return (arg.flatten()[0].clone().expand(*arg.size()), f'{ref}.contiguous(memory_format={fmt})', None)
        raise Exception(f"Bundled input argument at position '{ref}' is a tensor with storage size {arg._typed_storage().size()}. You probably don't want to bundle this as an input. ")
    else:
        return (arg, ref, None)

def _get_bundled_inputs_attributes_and_methods(script_module: torch.jit.ScriptModule) -> Tuple[List[str], List[str]]:
    if False:
        while True:
            i = 10
    methods: List[str] = []
    attributes: List[str] = []
    if hasattr(script_module, 'get_all_bundled_inputs'):
        methods.append('get_all_bundled_inputs')
        methods.append('get_num_bundled_inputs')
        methods.append('run_on_bundled_input')
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        methods.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            methods.append('get_all_bundled_inputs_for_' + function_name)
            methods.append('_generate_bundled_inputs_for_' + function_name)
            attributes.append('_bundled_inputs_deflated_' + function_name)
            bundled_inputs_fn = getattr(script_module, f'get_all_bundled_inputs_for_{function_name}')
            num_bundled_inputs: int = len(bundled_inputs_fn())
            func = getattr(script_module, function_name)
            for arg_idx in range(len(func.schema.arguments) - 1):
                for input_idx in range(num_bundled_inputs):
                    helper_fn_name = _get_inflate_helper_fn_name(arg_idx=arg_idx, input_idx=input_idx, function_name=function_name)
                    if hasattr(script_module, helper_fn_name):
                        methods.append(helper_fn_name)
    return (methods, attributes)

def _get_inflate_helper_fn_name(arg_idx: int, input_idx: int, function_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'_inflate_helper_for_{function_name}_input_{input_idx}_arg_{arg_idx}'

def bundle_randn(*size, dtype=None):
    if False:
        i = 10
        return i + 15
    'Generate a tensor that will be inflated with torch.randn.'
    stub = torch.zeros(1, dtype=dtype).expand(*size)
    return InflatableArg(value=stub, fmt='torch.randn_like({})')

def bundle_large_tensor(t):
    if False:
        print('Hello World!')
    'Wrap a tensor to allow bundling regardless of size.'
    return InflatableArg(value=t, fmt='{}')