import inspect
import sys
import warnings
import numpy as np
import paddle
from paddle import _C_ops, profiler
from paddle.base.data_feeder import _PADDLE_DTYPE_2_NUMPY_DTYPE, convert_uint16_to_float
from paddle.profiler.utils import in_profiler_mode
from paddle.utils import deprecated
from .. import core, framework, unique_name
from ..framework import EagerParamBase, Parameter, Variable, _getitem_static, _setitem_impl_, _setitem_static, convert_np_dtype_to_dtype_
from .base import switch_to_static_graph
from .math_op_patch import monkey_patch_math_tensor
_grad_scalar = None

class TensorHookRemoveHelper:
    """
    A helper class that for removing Tensor gradient's hook.
    NOTE(wuweilong):the operation weakref.ref(tensor) will cause some unexpected errors in eager mode.
    """

    def __init__(self, tensor, hook_id):
        if False:
            i = 10
            return i + 15
        self._tensor = tensor
        self._hook_id = hook_id

    def remove(self):
        if False:
            i = 10
            return i + 15
        "\n        Remove reference Tensor's hook.\n\n        Returns:\n            bool: Return True if removed successfully\n        "
        tensor = self._tensor
        if tensor is not None:
            res = tensor._remove_grad_hook(self._hook_id)
            if res is True:
                return True
            else:
                warnings.warn('The backward hook (ID: %d) of Tensor `%s` you want to remove does not exist or has been removed.' % (self._hook_id, tensor.name), RuntimeWarning)
        return False
_already_patch_repr = False

def monkey_patch_tensor():
    if False:
        print('Hello World!')

    @switch_to_static_graph
    def _to_static_var(self, to_parameter=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        **Notes**:\n            **This API is ONLY available in Dygraph mode**\n\n        Transform a Tensor into static Variable with same attributes. It's a low level interface used\n        in dy2static and shall not be called directly.\n\n        Args:\n            to_parameter (bool): It takes effect only if the input a Tensor. If set True,\n                                 the Tensor will be converted into framework.Parameters. Otherwise, it will\n                                 be converted into framework.Variable. Default False.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle.base as base\n                >>> from paddle.base.dygraph.base import to_variable\n                >>> import numpy as np\n\n                >>> data = np.ones([3, 1024], dtype='float32')\n                >>> with base.dygraph.guard():\n                ...     tensor = to_variable(data)\n                ...     static_var = tensor._to_static_var()\n        "
        attr_not_need_keys = ['grad', 'T', 'place', '_place_str', 'data', 'grad_', 'strides', 'offset']
        param_keys = ['stop_gradient', 'trainable']
        if isinstance(self, EagerParamBase):
            attr_kwargs = self.__dict__.copy()
            for key in param_keys:
                attr_kwargs[key] = getattr(self, key)
        else:
            attr_names = []
            for name in dir(self):
                if name not in attr_not_need_keys:
                    if not inspect.ismethod(getattr(self, name)) and (not name.startswith('_')):
                        attr_names.append(name)
            attr_kwargs = {name: getattr(self, name) for name in attr_names}
        attr_keys = ['block', 'shape', 'dtype', 'type', 'name', 'persistable']
        for attr in attr_keys:
            attr_kwargs[attr] = getattr(self, attr, None)
        if 'block' in kwargs:
            attr_kwargs['block'] = kwargs['block']
        attr_kwargs.update(kwargs)
        if to_parameter or isinstance(self, EagerParamBase):
            del attr_kwargs['persistable']
            attr_kwargs['block'] = attr_kwargs['block'].program.global_block()
            static_var = Parameter(**attr_kwargs)
        else:
            static_var = Variable(**attr_kwargs)
        return static_var

    @framework.dygraph_only
    def set_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        **Notes**:\n            **This API is ONLY available in Dygraph mode**\n\n        Set a new value for this Variable.\n\n        Args:\n            value (Variable|np.ndarray): the new value.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle.base as base\n                >>> from paddle.base.dygraph.base import to_variable\n                >>> from paddle.nn import Linear\n                >>> import numpy as np\n\n                >>> data = np.ones([3, 1024], dtype=\'float32\')\n                >>> with base.dygraph.guard():\n                ...     linear = Linear(1024, 4)\n                ...     t = to_variable(data)\n                ...     linear(t)  # call with default weight\n                ...     custom_weight = np.random.randn(1024, 4).astype("float32")\n                ...     linear.weight.set_value(custom_weight)  # change existing weight\n                ...     out = linear(t)  # call with different weight\n        '
        base_tensor = core.eager.Tensor
        assert isinstance(value, (np.ndarray, base_tensor, dict, str)), 'Variable set_value function, arguments type only support Variable, numpy, Tensor, dict, string.'
        if isinstance(value, (dict, str)):
            assert len(self) == len(value), 'Variable length not match, Variable [ {} ] need tensor with length {} but load set tensor with length {}'.format(self.name, len(self), len(value))
            if isinstance(value, dict):
                self.value().set_vocab(value)
            else:
                self.value().set_string_list(value)
        else:
            assert self.shape == list(value.shape), 'Variable Shape not match, Variable [ {} ] need tensor with shape {} but load set tensor with shape {}'.format(self.name, self.shape, value.shape)
            if isinstance(value, base_tensor):
                dtype = value.dtype
            else:
                dtype = convert_np_dtype_to_dtype_(value.dtype)
            assert self.dtype == dtype, 'Variable dtype not match, Variable [ {} ] need tensor with dtype {}  but load tensor with dtype {}'.format(self.name, self.dtype, dtype)
            self.value().get_tensor().set(value, framework._current_expected_place())

    @framework.dygraph_only
    def backward(self, grad_tensor=None, retain_graph=False):
        if False:
            while True:
                i = 10
        '\n        Run backward of current Graph which starts from current Tensor.\n\n        The new gradient will accumulate on previous gradient.\n\n        You can clear gradient by ``Tensor.clear_grad()`` .\n\n        Args:\n            grad_tensor(Tensor, optional): initial gradient values of the current Tensor. If `grad_tensor` is None,\n            the initial gradient values of the current Tensor would be Tensor filled with 1.0;\n            if `grad_tensor` is not None, it must have the same length as the current Tensor.\n            The default value is None.\n\n            retain_graph(bool, optional): If False, the graph used to compute grads will be freed. If you would\n                like to add more ops to the built graph after calling this method( :code:`backward` ), set the parameter\n                :code:`retain_graph` to True, then the grads will be retained. Thus, setting it to False is much more memory-efficient.\n                Defaults to False.\n        Returns:\n            NoneType: None\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> x = paddle.to_tensor(5., stop_gradient=False)\n                >>> for i in range(5):\n                ...     y = paddle.pow(x, 4.0)\n                ...     y.backward()\n                ...     print("{}: {}".format(i, x.grad))\n                0: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                500.)\n                1: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                1000.)\n                2: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                1500.)\n                3: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                2000.)\n                4: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                2500.)\n\n                >>> x.clear_grad()\n                >>> print("{}".format(x.grad))\n                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                0.)\n\n                >>> grad_tensor=paddle.to_tensor(2.)\n                >>> for i in range(5):\n                ...     y = paddle.pow(x, 4.0)\n                ...     y.backward(grad_tensor)\n                ...     print("{}: {}".format(i, x.grad))\n                0: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                1000.)\n                1: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                2000.)\n                2: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                3000.)\n                3: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                4000.)\n                4: Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,\n                5000.)\n        '
        if framework.in_dygraph_mode():
            if in_profiler_mode():
                record_event = profiler.RecordEvent('Gradient Backward', profiler.TracerEventType.Backward)
                record_event.begin()
            if grad_tensor is not None:
                assert isinstance(grad_tensor, core.eager.Tensor), 'The type of grad_tensor must be paddle.Tensor'
                assert grad_tensor.shape == self.shape, 'Tensor shape not match, Tensor of grad_tensor [ {} ] with shape {} mismatch Tensor [ {} ] with shape {}'.format(grad_tensor.name, grad_tensor.shape, self.name, self.shape)
            if grad_tensor is None:
                grad_tensor = []
            else:
                grad_tensor = [grad_tensor]
            if _grad_scalar:
                self = _grad_scalar.scale(self)
            core.eager.run_backward([self], grad_tensor, retain_graph)
            if in_profiler_mode():
                record_event.end()
        else:
            raise ValueError('Variable.backward() is only available in DyGraph mode')

    @framework.dygraph_only
    @deprecated(since='2.1.0', level=1, reason='Please use tensor.grad, which returns the tensor value of the gradient.')
    def gradient(self):
        if False:
            i = 10
            return i + 15
        '\n        .. warning::\n          This API will be deprecated in the future, it is recommended to use\n          :code:`x.grad` which returns the tensor value of the gradient.\n\n        Get the Gradient of Current Tensor.\n\n        Returns:\n            ndarray: Numpy value of the gradient of current Tensor\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> x = paddle.to_tensor(5., stop_gradient=False)\n                >>> y = paddle.pow(x, 4.0)\n                >>> y.backward()\n                >>> print("grad of x: {}".format(x.gradient()))\n                grad of x: 500.0\n\n        '
        if self.grad is None:
            return None
        if self.grad.is_selected_rows():
            return (np.array(self.grad), np.array(self.grad.rows()))
        return np.array(self.grad)

    @framework.dygraph_only
    def register_hook(self, hook):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers a backward hook for current Tensor.\n\n        The hook will be called every time the gradient Tensor of current Tensor is computed.\n\n        The hook should not modify the input gradient Tensor, but it can optionally return\n        a new gradient Tensor which will be used in place of current Tensor\'s gradient.\n\n        The hook should have the following signature:\n\n            hook(grad) -> Tensor or None\n\n        Args:\n            hook(function): A backward hook to be registered for Tensor.grad\n\n        Returns:\n            TensorHookRemoveHelper: A helper object that can be used to remove the registered hook by calling `remove()` method.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> # hook function return None\n                >>> def print_hook_fn(grad):\n                ...     print(grad)\n                ...\n                >>> # hook function return Tensor\n                >>> def double_hook_fn(grad):\n                ...     grad = grad * 2\n                ...     return grad\n                ...\n                >>> x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)\n                >>> y = paddle.to_tensor([4., 5., 6., 7.], stop_gradient=False)\n                >>> z = paddle.to_tensor([1., 2., 3., 4.])\n\n                >>> # one Tensor can register multiple hooks\n                >>> h = x.register_hook(print_hook_fn)\n                >>> x.register_hook(double_hook_fn)\n\n                >>> w = x + y\n                >>> # register hook by lambda function\n                >>> w.register_hook(lambda grad: grad * 2)\n\n                >>> o = z.matmul(w)\n                >>> o.backward()\n                >>> # print_hook_fn print content in backward\n                Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=False,\n                [2., 4., 6., 8.])\n\n                >>> print("w.grad:", w.grad)\n                w.grad: None\n                >>> print("x.grad:", x.grad)\n                x.grad: Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=False,\n                [4. , 8. , 12., 16.])\n                >>> print("y.grad:", y.grad)\n                y.grad: Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=False,\n                [2., 4., 6., 8.])\n\n                >>> # remove hook\n                >>> h.remove()\n        '
        if self.stop_gradient is True:
            raise RuntimeError('Cannot register hook on a tensor that stop gradient.')
        hook_id = self._register_grad_hook(hook)
        helper = TensorHookRemoveHelper(self, hook_id)
        return helper

    @framework.dygraph_only
    def _to(self, device=None, dtype=None, blocking=None):
        if False:
            for i in range(10):
                print('nop')
        if device is None and dtype is None and (blocking is None):
            return self
        if device is not None:
            if isinstance(device, str):
                device = paddle.device._convert_to_place(device)
            elif isinstance(device, (core.CPUPlace, core.CUDAPlace, core.CUDAPinnedPlace, core.XPUPlace, core.CustomPlace)):
                pass
            else:
                raise ValueError('device value error, must be str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace(), paddle.XPUPlace() or paddle.CustomPlace(), but the type of device is ' + type(device).__name__)
        if blocking is None:
            blocking = True
        else:
            assert isinstance(blocking, bool), 'blocking value error, must be the True, False or None'

        def transform(t, device, dtype, blocking):
            if False:
                print('Hello World!')
            if device is None:
                device = t.place
            if dtype is None:
                dtype = t.dtype
            if type(dtype) is str:
                dtype = framework.convert_np_dtype_to_dtype_(dtype)
            if t.place.is_gpu_place():
                size_dtype = core.size_of_dtype(dtype)
                waiting_alloc_memory = (t._numel() * size_dtype / 256 + 1) * 256 * 1.2
                gpu_memory_available = core.gpu_memory_available()
                if gpu_memory_available < waiting_alloc_memory:
                    t_used = t._copy_to(paddle.CPUPlace(), blocking)
                    t._clear()
                else:
                    t_used = t
            else:
                t_used = t
            if dtype is not None and dtype != t_used.dtype:
                with paddle.base.framework._dygraph_place_guard(place=t_used.place):
                    t_casted = t_used.cast(dtype=dtype)
            else:
                t_casted = t_used
            if device is not None and (not t_casted.place._equals(device)):
                new_t = t_casted._copy_to(device, blocking)
            else:
                new_t = t_casted
            dst_tensor = t.value().get_tensor()
            src_tensor = new_t.value().get_tensor()
            dst_tensor._share_data_with(src_tensor)
            return t
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            return transform(self, device, dtype, blocking)

    @framework.dygraph_only
    def to(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Performs Tensor dtype and/or device conversion. A paddle.dtype and place\n        are inferred from the arguments of ``self.to(*args, **kwargs)``.There are\n        three ways to call `to`:\n\n            1. to(dtype, blocking=True)\n            2. to(device, dtype=None, blocking=True)\n            3. to(other, blocking=True)\n\n        Returns:\n            Tensor: self\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> tensorx = paddle.to_tensor([1,2,3])\n                >>> print(tensorx)\n                Tensor(shape=[3], dtype=int64, place=Place(gpu:0), stop_gradient=True,\n                    [1, 2, 3])\n\n                >>> tensorx = tensorx.to("cpu")\n                >>> print(tensorx.place)\n                Place(cpu)\n\n                >>> tensorx = tensorx.to("float32")\n                >>> print(tensorx.dtype)\n                paddle.float32\n\n                >>> tensorx = tensorx.to("gpu", "int16")\n                >>> print(tensorx)\n                Tensor(shape=[3], dtype=int16, place=Place(gpu:0), stop_gradient=True,\n                    [1, 2, 3])\n                >>> tensor2 = paddle.to_tensor([4,5,6])\n                >>> tensor2\n                Tensor(shape=[3], dtype=int64, place=Place(gpu:0), stop_gradient=True,\n                    [4, 5, 6])\n                >>> tensor2 = tensor2.to(tensorx)\n                >>> print(tensor2)\n                Tensor(shape=[3], dtype=int16, place=Place(gpu:0), stop_gradient=True,\n                    [4, 5, 6])\n        '
        device = None
        dtype = None
        blocking = None
        size_args = len(args)
        size_kwargs = len(kwargs)

        def get_device_dtype_from_tensor(other):
            if False:
                print('Hello World!')
            if other is not None:
                device = str(other.place)[6:-1]
                dtype = other.dtype
                return (device, dtype)
            else:
                return (None, None)
        if size_args + size_kwargs > 3 or size_args + size_kwargs == 0:
            raise TypeError('to() received too mant arguments - expected one of:\n                  * (Union[str, paddle.CPUPlace(), paddle.CUDAPlace(), paddle.CUDAPinnedPlace(), paddle.XPUPlace(), paddle.CustomPlace()]                 device, Union[str, paddle.dtype, numpy.dtype] dtype, bool blocking)\n                 * (Union[str, paddle.dtype, numpy.dtype] dtype, bool blocking)\n                 * (paddle.Tensor other, bool blocking) ')
        valid_keys = {'device', 'dtype', 'blocking', 'other'}
        valid_dtypes = ['bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128', 'bool']
        invalid_keys = set(kwargs.keys()) - valid_keys
        if len(invalid_keys) != 0:
            raise TypeError('to() got an unexpected keyword argument ' + list(invalid_keys)[0])
        if size_args > 0:
            if isinstance(args[0], paddle.Tensor):
                (device, dtype) = get_device_dtype_from_tensor(args[0])
                if size_args == 2:
                    blocking = args[1]
                else:
                    blocking = kwargs.get('blocking', None)
            elif isinstance(args[0], (paddle.dtype, np.dtype)) or (isinstance(args[0], str) and args[0].lower() in valid_dtypes):
                dtype = args[0]
                if size_args == 2:
                    blocking = args[1]
                else:
                    blocking = kwargs.get('blocking', None)
            else:
                device = args[0]
                if size_args == 2:
                    dtype = args[1]
                elif size_args == 3:
                    (dtype, blocking) = (args[1], args[2])
                else:
                    dtype = kwargs.get('dtype', None)
                    blocking = kwargs.get('blocking', None)
        else:
            device = kwargs.get('device', None)
            dtype = kwargs.get('dtype', None)
            blocking = kwargs.get('blocking', None)
            if device is None and dtype is None:
                (device, dtype) = get_device_dtype_from_tensor(kwargs.get('other', None))
        return self._to(device, dtype, blocking)

    @property
    def grad(self):
        if False:
            print('Hello World!')
        '\n        .. warning::\n          This API will return the tensor value of the gradient. If you want\n          to get the numpy value of the gradient, you can use :code:`x.grad.numpy()`.\n\n        Get the Gradient of Current Tensor.\n\n        Returns:\n            Tensor: the gradient of current Tensor\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> x = paddle.to_tensor(5., stop_gradient=False)\n                >>> y = paddle.pow(x, 4.0)\n                >>> y.backward()\n                >>> print("grad of x: {}".format(x.grad))\n                grad of x: Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=False, 500.)\n\n        '
        msg = "tensor.grad will return the tensor value of the gradient. This is an incompatible upgrade for tensor.grad API.  It's return type changes from numpy.ndarray in version 2.0 to paddle.Tensor in version 2.1.0.  If you want to get the numpy value of the gradient, you can use :code:`x.grad.numpy()`"
        warning_msg = '\x1b[93m\nWarning:\n%s \x1b[0m' % msg
        if sys.platform.lower() == 'win32':
            warning_msg = '\nWarning:\n%s ' % msg
        warnings.warn(warning_msg)
        return self._grad_ivar()

    def clear_grad(self):
        if False:
            return 10
        '\n        The alias of clear_gradient().\n        '
        self.clear_gradient()

    def item(self, *args):
        if False:
            while True:
                i = 10
        "\n        Convert element at specific position in Tensor into Python scalars. If the position is not specified, the Tensor must be a\n        single-element Tensor.\n\n        Args:\n            *args(int): The input coordinates. If it's single int, the data in the corresponding order of flattened Tensor will be returned.\n                Default: None, and it must be in the case where Tensor has only one element.\n\n        Returns(Python scalar): A Python scalar, whose dtype is corresponds to the dtype of Tensor.\n\n        Raises:\n            ValueError: If the Tensor has more than one element, there must be coordinates.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> x = paddle.to_tensor(1)\n                >>> print(x.item())\n                1\n                >>> print(type(x.item()))\n                <class 'int'>\n\n                >>> x = paddle.to_tensor(1.0)\n                >>> print(x.item())\n                1.0\n                >>> print(type(x.item()))\n                <class 'float'>\n\n                >>> x = paddle.to_tensor(True)\n                >>> print(x.item())\n                True\n                >>> print(type(x.item()))\n                <class 'bool'>\n\n                >>> x = paddle.to_tensor(1+1j)\n                >>> print(x.item())\n                (1+1j)\n                >>> print(type(x.item()))\n                <class 'complex'>\n\n                >>> x = paddle.to_tensor([[1.1, 2.2, 3.3]])\n                >>> print(x.item(2))\n                3.299999952316284\n                >>> print(x.item(0, 2))\n                3.299999952316284\n\n        "
        scalar = self._getitem_from_offset(*args)
        if scalar.dtype == np.uint16:
            return convert_uint16_to_float(scalar).item()
        return scalar.item()

    @property
    def inplace_version(self):
        if False:
            while True:
                i = 10
        '\n        The inplace version of current Tensor.\n        The version number is incremented whenever the current Tensor is modified through an inplace operation.\n\n        **Notes: This is a read-only property**\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> var = paddle.ones(shape=[4, 2, 3], dtype="float32")\n                >>> print(var.inplace_version)\n                0\n\n                >>> var[1] = 2.2\n                >>> print(var.inplace_version)\n                1\n\n        '
        return self._inplace_version()

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert a Tensor object to a readable string.\n\n        Returns(str): A readable string.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> paddle.seed(2023)\n                >>> x = paddle.rand([2, 5])\n                >>> print(x)\n                Tensor(shape=[2, 5], dtype=float32, place=Place(cpu), stop_gradient=True,\n                [[0.86583614, 0.52014720, 0.25960937, 0.90525323, 0.42400089],\n                 [0.40641287, 0.97020894, 0.74437362, 0.51785129, 0.73292869]])\n        '
        from paddle.tensor.to_string import tensor_to_string
        return tensor_to_string(self)

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        '\n        Deep copy Tensor, it will always performs Tensor copy.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> import copy\n                >>> x = paddle.to_tensor(2.)\n                >>> y = copy.deepcopy(x)\n                >>> print(x)\n                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,\n                2.)\n                >>> print(y)\n                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,\n                2.)\n        '
        new_tensor = core.eager.Tensor()
        new_tensor.name = self.name + unique_name.generate('_deepcopy')
        memo[id(self)] = new_tensor
        new_tensor.copy_(self, True)
        return new_tensor

    @property
    def block(self):
        if False:
            return 10
        return framework.default_main_program().global_block()

    def __nonzero__(self):
        if False:
            i = 10
            return i + 15
        numel = int(np.prod(self.shape))
        assert numel == 1, 'When Variable is used as the condition of if/while , Variable can only contain one element.'
        assert self._is_initialized(), 'tensor not initialized'
        return bool(np.array(self) > 0)

    def __bool__(self):
        if False:
            while True:
                i = 10
        return self.__nonzero__()

    def __array__(self, dtype=None):
        if False:
            return 10
        "\n        Returns a numpy array shows the value of current Tensor.\n\n        Returns:\n            ndarray: The numpy value of current Tensor.\n\n        Returns type:\n            ndarray: dtype is same as current Tensor\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> import numpy as np\n                >>> x = paddle.randn([2, 2])\n                >>> x_array = np.array(x)\n\n                >>> print(type(x_array))\n                <class 'numpy.ndarray'>\n                >>> print(x_array.shape)\n                (2, 2)\n        "
        array = self.numpy(False)
        if dtype:
            array = array.astype(dtype)
        return array

    def contain_tensor(item):
        if False:
            i = 10
            return i + 15
        if not isinstance(item, (tuple, list)):
            item = [item]
        for slice_item in item:
            if isinstance(slice_item, slice):
                if isinstance(slice_item.start, Variable) or isinstance(slice_item.stop, Variable) or isinstance(slice_item.step, Variable):
                    return True
            elif isinstance(slice_item, (Variable, np.ndarray)) and Variable.dtype != paddle.bool:
                return True
        return False

    def contain_tensor_or_list(item):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, tuple):
            item = (item,)
        for slice_item in item:
            if isinstance(slice_item, (list, np.ndarray, Variable, range, bool)):
                return True
        return False

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        if contain_tensor_or_list(item):
            return _getitem_static(self, item)
        else:
            return self._getitem_index_not_tensor(item)

    def __setitem__(self, item, value):
        if False:
            return 10

        def is_combine_index(item):
            if False:
                while True:
                    i = 10
            var_type = None
            item_type = None
            if isinstance(item, (tuple, list)):
                for slice_item in item:
                    if item_type is None:
                        item_type = type(slice_item)
                    elif type(slice_item) != item_type:
                        return True
                    if isinstance(slice_item, Variable):
                        if var_type is None:
                            var_type = slice_item.dtype
                        elif var_type != slice_item.dtype:
                            return True
                return False
            return False
        if contain_tensor_or_list(item):
            if core.is_compiled_with_xpu() and (not is_combine_index(item)):
                return _setitem_impl_(self, item, value)
            return _setitem_static(self, item, value)
        else:
            return self.__setitem_eager_tensor__(item, value)

    @framework.dygraph_only
    def _set_grad_ivar(self, value):
        if False:
            print('Hello World!')
        if isinstance(self, EagerParamBase):
            self.grad = value
            self._unset_fake_empty()
        else:
            raise TypeError('_set_grad_ivar is only supported for Parameter Tensor')

    @framework.dygraph_only
    def value(self):
        if False:
            i = 10
            return i + 15
        return self

    @framework.dygraph_only
    def _slice(self, begin_idx, end_idx):
        if False:
            return 10
        return core.eager.Tensor(self.get_tensor()._slice(begin_idx, end_idx))

    @framework.dygraph_only
    def _numel(self):
        if False:
            while True:
                i = 10
        return self.get_tensor()._numel()

    @framework.dygraph_only
    def _clear_data(self):
        if False:
            return 10
        self.get_tensor()._clear()

    @framework.dygraph_only
    def _use_gpudnn(self, use_gpudnn=True):
        if False:
            while True:
                i = 10
        return self._tensor_use_gpudnn(use_gpudnn)

    @framework.dygraph_only
    def _uva(self, device_id=0):
        if False:
            while True:
                i = 10
        "\n        Returns self tensor with the UVA(unified virtual addressing).\n\n        Args:\n            device_id(int, optional): The destination GPU device id. Default: None, means current device.\n\n        Examples:\n            .. code-block:: python\n\n                >>> # doctest: +REQUIRES(env:GPU)\n                >>> import paddle\n                >>> paddle.device.set_device('gpu')\n                >>> x = paddle.to_tensor([1, 2, 3], place=paddle.CPUPlace())\n                >>> x._uva()\n                >>> print(x)\n        "
        self._tensor_uva(device_id)

    @framework.dygraph_only
    def cpu(self):
        if False:
            i = 10
            return i + 15
        if self.place.is_cpu_place():
            return self
        else:
            res = self._copy_to(core.CPUPlace(), True)
            res.stop_gradient = self.stop_gradient
            res.persistable = self.persistable
            return res

    @framework.dygraph_only
    def cuda(self, device_id=None, blocking=True):
        if False:
            for i in range(10):
                print('nop')
        if device_id is None:
            res_place = framework._current_expected_place()
            if not isinstance(res_place, core.CUDAPlace):
                res_place = core.CUDAPlace(0)
        elif isinstance(device_id, int):
            res_place = core.CUDAPlace(device_id)
        else:
            raise ValueError('device_id must be int|None')
        if self.place._equals(res_place):
            return self
        else:
            res = self._copy_to(res_place, blocking)
            res.stop_gradient = self.stop_gradient
            res.persistable = self.persistable
            return res

    @framework.dygraph_only
    def pin_memory(self):
        if False:
            print('Hello World!')
        if self.place.is_cuda_pinned_place():
            return self
        else:
            res = self._copy_to(core.CUDAPinnedPlace(), True)
            res.stop_gradient = self.stop_gradient
            res.persistable = self.persistable
            return res

    @framework.dygraph_only
    def values(self):
        if False:
            i = 10
            return i + 15
        "\n        **Notes**:\n            **This API is ONLY available in Dygraph mode**\n        Get the values of current SparseTensor(COO or CSR).\n\n        Returns:\n            Tensor: A DenseTensor\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]\n                >>> values = [1, 2, 3, 4, 5]\n                >>> dense_shape = [3, 4]\n                >>> sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype='int32'), paddle.to_tensor(values, dtype='float32'), shape=dense_shape)\n                >>> print(sparse_x.values())\n                Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=True,\n                [1., 2., 3., 4., 5.])\n        "
        return _C_ops.sparse_values(self)

    @framework.dygraph_only
    def to_dense(self):
        if False:
            i = 10
            return i + 15
        "\n        **Notes**:\n            **This API is ONLY available in Dygraph mode**\n        Convert the current SparseTensor(COO or CSR) to DenseTensor.\n\n        Returns:\n            Tensor: A DenseTensor\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]\n                >>> values = [1, 2, 3, 4, 5]\n                >>> dense_shape = [3, 4]\n                >>> sparse_x = paddle.sparse.sparse_coo_tensor(paddle.to_tensor(indices, dtype='int64'), paddle.to_tensor(values, dtype='float32'), shape=dense_shape)\n                >>> dense_x = sparse_x.to_dense()\n                >>> print(dense_x)\n                Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,\n                [[0., 1., 0., 2.],\n                 [0., 0., 3., 0.],\n                 [4., 5., 0., 0.]])\n        "
        return _C_ops.sparse_to_dense(self)

    @framework.dygraph_only
    def to_sparse_coo(self, sparse_dim):
        if False:
            print('Hello World!')
        "\n        **Notes**:\n            **This API is ONLY available in Dygraph mode**\n        Convert the current DenseTensor to SparseTensor in COO format.\n\n        Returns:\n            Tensor: A SparseCooTensor\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> dense_x = [[0, 1, 0, 2], [0, 0, 3, 4]]\n                >>> dense_x = paddle.to_tensor(dense_x, dtype='float32')\n                >>> sparse_x = dense_x.to_sparse_coo(sparse_dim=2)\n                >>> print(sparse_x)\n                Tensor(shape=[2, 4], dtype=paddle.float32, place=Place(cpu), stop_gradient=True,\n                       indices=[[0, 0, 1, 1],\n                                [1, 3, 2, 3]],\n                       values=[1., 2., 3., 4.])\n        "
        return _C_ops.sparse_to_sparse_coo(self, sparse_dim)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(id(self))

    @framework.dygraph_only
    def coalesce(self, name=None):
        if False:
            print('Hello World!')
        '\n        the coalesced operator include sorted and merge, after coalesced, the indices of x is sorted and unique.\n\n        Parameters:\n            x (Tensor): the input SparseCooTensor.\n            name (str, optional): Name for the operation (optional, default is None).\n                For more information, please refer to :ref:`api_guide_Name`.\n\n        Returns:\n            Tensor: return the SparseCooTensor after coalesced.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> indices = [[0, 0, 1], [1, 1, 2]]\n                >>> values = [1.0, 2.0, 3.0]\n                >>> sp_x = paddle.sparse.sparse_coo_tensor(indices, values)\n                >>> sp_x = sp_x.coalesce()\n                >>> print(sp_x.indices())\n                Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,\n                [[0, 1],\n                [1, 2]])\n                >>> print(sp_x.values())\n                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,\n                [3., 3.])\n        '
        return _C_ops.sparse_coalesce(self)
    if not hasattr(core, 'eager'):
        return
    for (method_name, method) in (('__bool__', __bool__), ('__nonzero__', __nonzero__), ('_to_static_var', _to_static_var), ('set_value', set_value), ('block', block), ('backward', backward), ('clear_grad', clear_grad), ('inplace_version', inplace_version), ('gradient', gradient), ('register_hook', register_hook), ('__str__', __str__), ('__repr__', __str__), ('__deepcopy__', __deepcopy__), ('__module__', 'paddle'), ('__array__', __array__), ('__getitem__', __getitem__), ('item', item), ('__setitem__', __setitem__), ('_to', _to), ('to', to), ('values', values), ('to_dense', to_dense), ('to_sparse_coo', to_sparse_coo), ('coalesce', coalesce), ('_set_grad_ivar', _set_grad_ivar), ('value', value), ('cpu', cpu), ('cuda', cuda), ('pin_memory', pin_memory), ('_slice', _slice), ('_numel', _numel), ('_uva', _uva), ('_clear_data', _clear_data), ('__hash__', __hash__), ('_use_gpudnn', _use_gpudnn)):
        setattr(core.eager.Tensor, method_name, method)
    global _already_patch_repr
    if not _already_patch_repr:
        origin = core.VarDesc.VarType.__str__

        def dtype_str(dtype):
            if False:
                i = 10
                return i + 15
            if dtype in _PADDLE_DTYPE_2_NUMPY_DTYPE:
                numpy_dtype = _PADDLE_DTYPE_2_NUMPY_DTYPE[dtype]
                if numpy_dtype == 'uint16':
                    numpy_dtype = 'bfloat16'
                prefix = 'paddle.'
                return prefix + numpy_dtype
            else:
                return origin(dtype)
        core.VarDesc.VarType.__str__ = dtype_str
        _already_patch_repr = True
    monkey_patch_math_tensor()