import functools
from inspect import getfullargspec
import numpy as np
import torch

def array_converter(to_torch=True, apply_to=tuple(), template_arg_name_=None, recover=True):
    if False:
        for i in range(10):
            print('nop')
    "Wrapper function for data-type agnostic processing.\n\n    First converts input arrays to PyTorch tensors or NumPy ndarrays\n    for middle calculation, then convert output to original data-type if\n    `recover=True`.\n\n    Args:\n        to_torch (Bool, optional): Whether convert to PyTorch tensors\n            for middle calculation. Defaults to True.\n        apply_to (tuple[str], optional): The arguments to which we apply\n            data-type conversion. Defaults to an empty tuple.\n        template_arg_name_ (str, optional): Argument serving as the template (\n            return arrays should have the same dtype and device\n            as the template). Defaults to None. If None, we will use the\n            first argument in `apply_to` as the template argument.\n        recover (Bool, optional): Whether or not recover the wrapped function\n            outputs to the `template_arg_name_` type. Defaults to True.\n\n    Raises:\n        ValueError: When template_arg_name_ is not among all args, or\n            when apply_to contains an arg which is not among all args,\n            a ValueError will be raised. When the template argument or\n            an argument to convert is a list or tuple, and cannot be\n            converted to a NumPy array, a ValueError will be raised.\n        TypeError: When the type of the template argument or\n                an argument to convert does not belong to the above range,\n                or the contents of such an list-or-tuple-type argument\n                do not share the same data type, a TypeError is raised.\n\n    Returns:\n        (function): wrapped function.\n\n    Example:\n        >>> import torch\n        >>> import numpy as np\n        >>>\n        >>> # Use torch addition for a + b,\n        >>> # and convert return values to the type of a\n        >>> @array_converter(apply_to=('a', 'b'))\n        >>> def simple_add(a, b):\n        >>>     return a + b\n        >>>\n        >>> a = np.array([1.1])\n        >>> b = np.array([2.2])\n        >>> simple_add(a, b)\n        >>>\n        >>> # Use numpy addition for a + b,\n        >>> # and convert return values to the type of b\n        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),\n        >>>                  template_arg_name_='b')\n        >>> def simple_add(a, b):\n        >>>     return a + b\n        >>>\n        >>> simple_add()\n        >>>\n        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),\n        >>> # and return the torch tensor\n        >>> @array_converter(apply_to=('a',), recover=False)\n        >>> def floor_or_ceil(a, flag=True):\n        >>>     return torch.floor(a) if flag else torch.ceil(a)\n        >>>\n        >>> floor_or_ceil(a, flag=False)\n    "

    def array_converter_wrapper(func):
        if False:
            for i in range(10):
                print('nop')
        'Outer wrapper for the function.'

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            if False:
                while True:
                    i = 10
            'Inner wrapper for the arguments.'
            if len(apply_to) == 0:
                return func(*args, **kwargs)
            func_name = func.__name__
            arg_spec = getfullargspec(func)
            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)
            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}
            all_arg_names = arg_names + kwonly_arg_names
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_
            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the argument list of function {func_name}')
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f'{arg_to_apply} is not an argument of {func_name}')
            new_args = []
            new_kwargs = {}
            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray
            for (i, arg_value) in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(converter.convert(input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)
                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value
            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=kwargs[arg_name], target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]
            new_args += nameless_args
            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data, tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for (k, v) in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data
            if recover:
                return recursive_recover(return_values)
            else:
                return return_values
        return new_func
    return array_converter_wrapper

class ArrayConverter:
    SUPPORTED_NON_ARRAY_TYPES = (int, float, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)

    def __init__(self, template_array=None):
        if False:
            for i in range(10):
                print('nop')
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array):
        if False:
            while True:
                i = 10
        'Set template array.\n\n        Args:\n            array (tuple | list | int | float | np.ndarray | torch.Tensor):\n                Template array.\n\n        Raises:\n            ValueError: If input is list or tuple and cannot be converted to\n                to a NumPy array, a ValueError is raised.\n            TypeError: If input type does not belong to the above range,\n                or the contents of a list or tuple do not share the\n                same data type, a TypeError is raised.\n        '
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'
        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print(f'The following list cannot be converted to a numpy array of supported dtype:\n{array}')
                raise
        elif isinstance(array, self.SUPPORTED_NON_ARRAY_TYPES):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f'Template type {self.array_type} is not supported.')

    def convert(self, input_array, target_type=None, target_array=None):
        if False:
            i = 10
            return i + 15
        "Convert input array to target data type.\n\n        Args:\n            input_array (tuple | list | np.ndarray |\n                torch.Tensor | int | float ):\n                Input array. Defaults to None.\n            target_type (<class 'np.ndarray'> | <class 'torch.Tensor'>,\n                optional):\n                Type to which input array is converted. Defaults to None.\n            target_array (np.ndarray | torch.Tensor, optional):\n                Template array to which input array is converted.\n                Defaults to None.\n\n        Raises:\n            ValueError: If input is list or tuple and cannot be converted to\n                to a NumPy array, a ValueError is raised.\n            TypeError: If input type does not belong to the above range,\n                or the contents of a list or tuple do not share the\n                same data type, a TypeError is raised.\n        "
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print(f'The input cannot be converted to a single-type numpy array:\n{input_array}')
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, 'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), 'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                converted_array = torch.tensor(input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), 'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array):
        if False:
            print('Hello World!')
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), 'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array