import copy
import functools
import numpy as np
from operator import mul
from typing import Optional
import ivy
from .conversions import args_to_native, to_ivy
from .activations import _ArrayWithActivations
from .creation import _ArrayWithCreation
from .data_type import _ArrayWithDataTypes
from .device import _ArrayWithDevice
from .elementwise import _ArrayWithElementwise
from .general import _ArrayWithGeneral
from .gradients import _ArrayWithGradients
from .image import _ArrayWithImage
from .layers import _ArrayWithLayers
from .linear_algebra import _ArrayWithLinearAlgebra
from .losses import _ArrayWithLosses
from .manipulation import _ArrayWithManipulation
from .norms import _ArrayWithNorms
from .random import _ArrayWithRandom
from .searching import _ArrayWithSearching
from .set import _ArrayWithSet
from .sorting import _ArrayWithSorting
from .statistical import _ArrayWithStatistical
from .utility import _ArrayWithUtility
from ivy.func_wrapper import handle_view_indexing
from .experimental import _ArrayWithSearchingExperimental, _ArrayWithActivationsExperimental, _ArrayWithConversionsExperimental, _ArrayWithCreationExperimental, _ArrayWithData_typeExperimental, _ArrayWithDeviceExperimental, _ArrayWithElementWiseExperimental, _ArrayWithGeneralExperimental, _ArrayWithGradientsExperimental, _ArrayWithImageExperimental, _ArrayWithLayersExperimental, _ArrayWithLinearAlgebraExperimental, _ArrayWithLossesExperimental, _ArrayWithManipulationExperimental, _ArrayWithNormsExperimental, _ArrayWithRandomExperimental, _ArrayWithSetExperimental, _ArrayWithSortingExperimental, _ArrayWithStatisticalExperimental, _ArrayWithUtilityExperimental

class Array(_ArrayWithActivations, _ArrayWithCreation, _ArrayWithDataTypes, _ArrayWithDevice, _ArrayWithElementwise, _ArrayWithGeneral, _ArrayWithGradients, _ArrayWithImage, _ArrayWithLayers, _ArrayWithLinearAlgebra, _ArrayWithLosses, _ArrayWithManipulation, _ArrayWithNorms, _ArrayWithRandom, _ArrayWithSearching, _ArrayWithSet, _ArrayWithSorting, _ArrayWithStatistical, _ArrayWithUtility, _ArrayWithActivationsExperimental, _ArrayWithConversionsExperimental, _ArrayWithCreationExperimental, _ArrayWithData_typeExperimental, _ArrayWithDeviceExperimental, _ArrayWithElementWiseExperimental, _ArrayWithGeneralExperimental, _ArrayWithGradientsExperimental, _ArrayWithImageExperimental, _ArrayWithLayersExperimental, _ArrayWithLinearAlgebraExperimental, _ArrayWithLossesExperimental, _ArrayWithManipulationExperimental, _ArrayWithNormsExperimental, _ArrayWithRandomExperimental, _ArrayWithSearchingExperimental, _ArrayWithSetExperimental, _ArrayWithSortingExperimental, _ArrayWithStatisticalExperimental, _ArrayWithUtilityExperimental):

    def __init__(self, data, dynamic_backend=None):
        if False:
            while True:
                i = 10
        _ArrayWithActivations.__init__(self)
        _ArrayWithCreation.__init__(self)
        _ArrayWithDataTypes.__init__(self)
        _ArrayWithDevice.__init__(self)
        _ArrayWithElementwise.__init__(self)
        _ArrayWithGeneral.__init__(self)
        _ArrayWithGradients.__init__(self)
        _ArrayWithImage.__init__(self)
        _ArrayWithLayers.__init__(self)
        _ArrayWithLinearAlgebra.__init__(self)
        _ArrayWithLosses.__init__(self)
        _ArrayWithManipulation.__init__(self)
        _ArrayWithNorms.__init__(self)
        _ArrayWithRandom.__init__(self)
        _ArrayWithSearching.__init__(self)
        _ArrayWithSet.__init__(self)
        _ArrayWithSorting.__init__(self)
        _ArrayWithStatistical.__init__(self)
        _ArrayWithUtility.__init__(self)
        (_ArrayWithActivationsExperimental.__init__(self),)
        (_ArrayWithConversionsExperimental.__init__(self),)
        (_ArrayWithCreationExperimental.__init__(self),)
        (_ArrayWithData_typeExperimental.__init__(self),)
        (_ArrayWithDeviceExperimental.__init__(self),)
        (_ArrayWithElementWiseExperimental.__init__(self),)
        (_ArrayWithGeneralExperimental.__init__(self),)
        (_ArrayWithGradientsExperimental.__init__(self),)
        (_ArrayWithImageExperimental.__init__(self),)
        (_ArrayWithLayersExperimental.__init__(self),)
        (_ArrayWithLinearAlgebraExperimental.__init__(self),)
        (_ArrayWithLossesExperimental.__init__(self),)
        (_ArrayWithManipulationExperimental.__init__(self),)
        (_ArrayWithNormsExperimental.__init__(self),)
        (_ArrayWithRandomExperimental.__init__(self),)
        (_ArrayWithSearchingExperimental.__init__(self),)
        (_ArrayWithSetExperimental.__init__(self),)
        (_ArrayWithSortingExperimental.__init__(self),)
        (_ArrayWithStatisticalExperimental.__init__(self),)
        (_ArrayWithUtilityExperimental.__init__(self),)
        self._init(data, dynamic_backend)
        self._view_attributes(data)

    def _init(self, data, dynamic_backend=None):
        if False:
            i = 10
            return i + 15
        if ivy.is_ivy_array(data):
            self._data = data.data
        elif ivy.is_native_array(data):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = ivy.asarray(data)._data
        elif ivy.is_ivy_sparse_array(data):
            self._data = data._data
        elif ivy.is_native_sparse_array(data):
            self._data = data._data
        else:
            raise ivy.utils.exceptions.IvyException('data must be ivy array, native array or ndarray')
        self._size = None
        self._strides = None
        self._itemsize = None
        self._dtype = None
        self._device = None
        self._dev_str = None
        self._pre_repr = None
        self._post_repr = None
        self._backend = ivy.current_backend(self._data).backend
        if dynamic_backend is not None:
            self._dynamic_backend = dynamic_backend
        else:
            self._dynamic_backend = ivy.dynamic_backend
        self.weak_type = False

    def _view_attributes(self, data):
        if False:
            while True:
                i = 10
        self._base = None
        self._view_refs = []
        self._manipulation_stack = []
        self._torch_base = None
        self._torch_view_refs = []
        self._torch_manipulation = None

    @property
    def backend(self):
        if False:
            return 10
        return self._backend

    @property
    def dynamic_backend(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dynamic_backend

    @dynamic_backend.setter
    def dynamic_backend(self, value):
        if False:
            for i in range(10):
                print('nop')
        from ivy.functional.ivy.gradients import _variable
        from ivy.utils.backend.handler import _data_to_new_backend, _get_backend_for_arg
        if value:
            ivy_backend = ivy.with_backend(self._backend)
            if ivy_backend.gradients._is_variable(self.data):
                native_var = ivy_backend.gradients._variable_data(self)
                data = _data_to_new_backend(native_var, ivy_backend).data
                self._data = _variable(data).data
            else:
                self._data = _data_to_new_backend(self, ivy_backend).data
            self._backend = ivy.backend
        else:
            self._backend = _get_backend_for_arg(self.data.__class__.__module__).backend
        self._dynamic_backend = value

    @property
    def data(self) -> ivy.NativeArray:
        if False:
            while True:
                i = 10
        'The native array being wrapped in self.'
        return self._data

    @property
    def dtype(self) -> ivy.Dtype:
        if False:
            for i in range(10):
                print('nop')
        'Data type of the array elements.'
        if self._dtype is None:
            self._dtype = ivy.dtype(self._data)
        return self._dtype

    @property
    def device(self) -> ivy.Device:
        if False:
            i = 10
            return i + 15
        'Hardware device the array data resides on.'
        if self._device is None:
            self._device = ivy.dev(self._data)
        return self._device

    @property
    def mT(self) -> ivy.Array:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transpose of a matrix (or a stack of matrices).\n\n        Returns\n        -------\n        ret\n            array whose last two dimensions (axes) are permuted in reverse order\n            relative to original array (i.e., for an array instance having shape\n            ``(..., M, N)``, the returned array must have shape ``(..., N, M)``).\n            The returned array must have the same data type as the original array.\n        '
        ivy.utils.assertions.check_greater(len(self._data.shape), 2, allow_equal=True, as_array=False)
        return ivy.matrix_transpose(self._data)

    @property
    def ndim(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Number of array dimensions (axes).'
        return len(tuple(self._data.shape))

    @property
    def shape(self) -> ivy.Shape:
        if False:
            while True:
                i = 10
        'Array dimensions.'
        return ivy.Shape(self._data.shape)

    @property
    def size(self) -> Optional[int]:
        if False:
            return 10
        'Number of elements in the array.'
        if self._size is None:
            if ivy.current_backend_str() in ['numpy', 'jax']:
                self._size = self._data.size
                return self._size
            self._size = functools.reduce(mul, self._data.shape) if len(self._data.shape) > 0 else 1
        return self._size

    @property
    def itemsize(self) -> Optional[int]:
        if False:
            return 10
        'Size of array elements in bytes.'
        if self._itemsize is None:
            self._itemsize = ivy.itemsize(self._data)
        return self._itemsize

    @property
    def strides(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        'Get strides across each dimension.'
        if self._strides is None:
            self._strides = ivy.strides(self)
        return self._strides

    @property
    def T(self) -> ivy.Array:
        if False:
            i = 10
            return i + 15
        '\n        Transpose of the array.\n\n        Returns\n        -------\n        ret\n            two-dimensional array whose first and last dimensions (axes) are\n            permuted in reverse order relative to original array.\n        '
        ivy.utils.assertions.check_equal(len(self._data.shape), 2, as_array=False)
        return ivy.matrix_transpose(self._data)

    @property
    def base(self) -> ivy.Array:
        if False:
            while True:
                i = 10
        'Original array referenced by view.'
        return self._base

    @property
    def real(self) -> ivy.Array:
        if False:
            while True:
                i = 10
        '\n        Real part of the array.\n\n        Returns\n        -------\n        ret\n            array containing the real part of each element in the array.\n            The returned array must have the same shape and data type as\n            the original array.\n        '
        return ivy.real(self._data)

    @property
    def imag(self) -> ivy.Array:
        if False:
            print('Hello World!')
        '\n        Imaginary part of the array.\n\n        Returns\n        -------\n        ret\n            array containing the imaginary part of each element in the array.\n            The returned array must have the same shape and data type as\n            the original array.\n        '
        return ivy.imag(self._data)

    @data.setter
    def data(self, data):
        if False:
            while True:
                i = 10
        ivy.utils.assertions.check_true(ivy.is_native_array(data), 'data must be native array')
        self._init(data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        if False:
            i = 10
            return i + 15
        (args, kwargs) = args_to_native(*args, **kwargs)
        return func(*args, **kwargs)

    def __ivy_array_function__(self, func, types, args, kwargs):
        if False:
            return 10
        for t in types:
            if hasattr(t, '__ivy_array_function__') and t.__ivy_array_function__ is not ivy.Array.__ivy_array_function__ or (hasattr(ivy.NativeArray, '__ivy_array_function__') and t.__ivy_array_function__ is not ivy.NativeArray.__ivy_array_function__):
                return NotImplemented
        return func(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (args, kwargs) = args_to_native(*args, **kwargs)
        return self._data.__array__(*args, dtype=self.dtype, **kwargs)

    def __array_prepare__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        (args, kwargs) = args_to_native(*args, **kwargs)
        return self._data.__array_prepare__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        if False:
            return 10
        (args, kwargs) = args_to_native(*args, **kwargs)
        return self._data.__array_ufunc__(*args, **kwargs)

    def __array_wrap__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        (args, kwargs) = args_to_native(*args, **kwargs)
        return self._data.__array_wrap__(*args, **kwargs)

    def __array_namespace__(self, api_version=None):
        if False:
            for i in range(10):
                print('nop')
        return ivy

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self._dev_str is None:
            self._dev_str = ivy.as_ivy_dev(self.device)
            self._pre_repr = 'ivy.array'
            if 'gpu' in self._dev_str:
                self._post_repr = f', dev={self._dev_str})'
            else:
                self._post_repr = ')'
        sig_fig = ivy.array_significant_figures
        dec_vals = ivy.array_decimal_values
        backend = ivy.with_backend(self.backend)
        arr_np = backend.to_numpy(self._data)
        rep = np.array(ivy.vec_sig_fig(arr_np, sig_fig)) if self.size > 0 else np.array(arr_np)
        with np.printoptions(precision=dec_vals):
            repr = rep.__repr__()[:-1].partition(', dtype')[0].partition(', dev')[0]
            return self._pre_repr + repr[repr.find('('):] + self._post_repr.format(ivy.current_backend_str())

    def __dir__(self):
        if False:
            i = 10
            return i + 15
        return self._data.__dir__()

    def __getattribute__(self, item):
        if False:
            i = 10
            return i + 15
        return super().__getattribute__(item)

    def __getattr__(self, item):
        if False:
            for i in range(10):
                print('nop')
        try:
            attr = self._data.__getattribute__(item)
        except AttributeError:
            attr = self._data.__getattr__(item)
        return to_ivy(attr)

    @handle_view_indexing
    def __getitem__(self, query):
        if False:
            i = 10
            return i + 15
        return ivy.get_item(self._data, query)

    def __setitem__(self, query, val):
        if False:
            for i in range(10):
                print('nop')
        self._data = ivy.set_item(self._data, query, val)._data

    def __contains__(self, key):
        if False:
            print('Hello World!')
        return self._data.__contains__(key)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        data_dict = {}
        data_dict['data'] = self.data
        data_dict['backend'] = self.backend
        data_dict['device_str'] = ivy.as_ivy_dev(self.device)
        return data_dict

    def __setstate__(self, state):
        if False:
            return 10
        ivy.set_backend(state['backend']) if state['backend'] is not None and len(state['backend']) > 0 else ivy.current_backend(state['data'])
        ivy_array = ivy.array(state['data'])
        ivy.previous_backend()
        self.__dict__ = ivy_array.__dict__

    def __pos__(self):
        if False:
            for i in range(10):
                print('nop')
        return ivy.positive(self._data)

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        return ivy.negative(self._data)

    def __pow__(self, power):
        if False:
            print('Hello World!')
        '\n        ivy.Array special method variant of ivy.pow. This method simply wraps the\n        function, and so the docstring for ivy.pow also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            Input array or float.\n        power\n            Array or float power. Must be compatible with ``self``\n            (see :ref:`broadcasting`). Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise sums. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        With :class:`ivy.Array` input:\n\n        >>> x = ivy.array([1, 2, 3])\n        >>> y = x ** 2\n        >>> print(y)\n        ivy.array([1, 4, 9])\n\n        >>> x = ivy.array([1.2, 2.1, 3.5])\n        >>> y = x ** 2.9\n        >>> print(y)\n        ivy.array([ 1.69678056,  8.59876156, 37.82660675])\n        '
        return ivy.pow(self._data, power)

    def __rpow__(self, power):
        if False:
            return 10
        return ivy.pow(power, self._data)

    def __ipow__(self, power):
        if False:
            return 10
        return ivy.pow(self._data, power)

    def __add__(self, other):
        if False:
            return 10
        '\n        ivy.Array special method variant of ivy.add. This method simply wraps the\n        function, and so the docstring for ivy.add also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have a numeric data type.\n        other\n            second input array. Must be compatible with ``self``\n            (see :ref:`broadcasting`). Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise sums. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        >>> x = ivy.array([1, 2, 3])\n        >>> y = ivy.array([4, 5, 6])\n        >>> z = x + y\n        >>> print(z)\n        ivy.array([5, 7, 9])\n        '
        return ivy.add(self._data, other)

    def __radd__(self, other):
        if False:
            return 10
        '\n        ivy.Array reverse special method variant of ivy.add. This method simply wraps\n        the function, and so the docstring for ivy.add also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have a numeric data type.\n        other\n            second input array. Must be compatible with ``self``\n            (see :ref:`broadcasting`). Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise sums. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        >>> x = 1\n        >>> y = ivy.array([4, 5, 6])\n        >>> z = x + y\n        >>> print(z)\n        ivy.array([5, 6, 7])\n        '
        return ivy.add(other, self._data)

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        return ivy.add(self._data, other)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        ivy.Array special method variant of ivy.subtract. This method simply wraps the\n        function, and so the docstring for ivy.subtract also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have a numeric data type.\n        other\n            second input array. Must be compatible with ``self``\n            (see :ref:`broadcasting`). Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise differences. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances only:\n\n        >>> x = ivy.array([1, 2, 3])\n        >>> y = ivy.array([4, 5, 6])\n        >>> z = x - y\n        >>> print(z)\n        ivy.array([-3, -3, -3])\n        '
        return ivy.subtract(self._data, other)

    def __rsub__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        ivy.Array reverse special method variant of ivy.subtract. This method simply\n        wraps the function, and so the docstring for ivy.subtract also applies to this\n        method with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have a numeric data type.\n        other\n            second input array. Must be compatible with ``self``\n            (see :ref:`broadcasting`). Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise differences. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        >>> x = 1\n        >>> y = ivy.array([4, 5, 6])\n        >>> z = x - y\n        >>> print(z)\n        ivy.array([-3, -4, -5])\n        '
        return ivy.subtract(other, self._data)

    def __isub__(self, other):
        if False:
            print('Hello World!')
        return ivy.subtract(self._data, other)

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        return ivy.multiply(self._data, other)

    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return ivy.multiply(other, self._data)

    def __imul__(self, other):
        if False:
            print('Hello World!')
        return ivy.multiply(self._data, other)

    def __mod__(self, other):
        if False:
            return 10
        return ivy.remainder(self._data, other)

    def __rmod__(self, other):
        if False:
            return 10
        return ivy.remainder(other, self._data)

    def __imod__(self, other):
        if False:
            while True:
                i = 10
        return ivy.remainder(self._data, other)

    def __divmod__(self, other):
        if False:
            return 10
        return (ivy.divide(self._data, other), ivy.remainder(self._data, other))

    def __rdivmod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return (ivy.divide(other, self._data), ivy.remainder(other, self._data))

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        ivy.Array reverse special method variant of ivy.divide. This method simply wraps\n        the function, and so the docstring for ivy.divide also applies to this method\n        with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have a numeric data type.\n        other\n            second input array. Must be compatible with ``self``\n            (see :ref:`broadcasting`). Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        >>> x = ivy.array([1, 2, 3])\n        >>> y = ivy.array([4, 5, 6])\n        >>> z = x / y\n        >>> print(z)\n        ivy.array([0.25      , 0.40000001, 0.5       ])\n        '
        return ivy.divide(self._data, other)

    def __rtruediv__(self, other):
        if False:
            return 10
        return ivy.divide(other, self._data)

    def __itruediv__(self, other):
        if False:
            print('Hello World!')
        return ivy.divide(self._data, other)

    def __floordiv__(self, other):
        if False:
            print('Hello World!')
        return ivy.floor_divide(self._data, other)

    def __rfloordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return ivy.floor_divide(other, self._data)

    def __ifloordiv__(self, other):
        if False:
            while True:
                i = 10
        return ivy.floor_divide(self._data, other)

    def __matmul__(self, other):
        if False:
            i = 10
            return i + 15
        return ivy.matmul(self._data, other)

    def __rmatmul__(self, other):
        if False:
            i = 10
            return i + 15
        return ivy.matmul(other, self._data)

    def __imatmul__(self, other):
        if False:
            return 10
        return ivy.matmul(self._data, other)

    def __abs__(self):
        if False:
            return 10
        '\n        ivy.Array special method variant of ivy.abs. This method simply wraps the\n        function, and so the docstring for ivy.abs also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            input array. Should have a numeric data type.\n\n        Returns\n        -------\n        ret\n            an array containing the absolute value of each element\n            in ``self``. The returned array must have the same data\n            type as ``self``.\n\n        Examples\n        --------\n        With :class:`ivy.Array` input:\n\n        >>> x = ivy.array([6, -2, 0, -1])\n        >>> print(abs(x))\n        ivy.array([6, 2, 0, 1])\n\n        >>> x = ivy.array([-1.2, 1.2])\n        >>> print(abs(x))\n        ivy.array([1.2, 1.2])\n        '
        return ivy.abs(self._data)

    def __float__(self):
        if False:
            print('Hello World!')
        if hasattr(self._data, '__float__'):
            if 'complex' in self.dtype:
                res = float(self.real)
            else:
                res = self._data.__float__()
        else:
            res = float(ivy.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_ivy(res)

    def __int__(self):
        if False:
            return 10
        if hasattr(self._data, '__int__'):
            if 'complex' in self.dtype:
                res = int(self.real)
            else:
                res = self._data.__int__()
        else:
            res = int(ivy.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_ivy(res)

    def __complex__(self):
        if False:
            i = 10
            return i + 15
        res = complex(ivy.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_ivy(res)

    def __bool__(self):
        if False:
            print('Hello World!')
        return self._data.__bool__()

    def __dlpack__(self, stream=None):
        if False:
            i = 10
            return i + 15
        return ivy.to_dlpack(self)

    def __dlpack_device__(self):
        if False:
            while True:
                i = 10
        return self._data.__dlpack_device__()

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        ivy.Array special method variant of ivy.less. This method simply wraps the\n        function, and so the docstring for ivy.less also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. May have any data type.\n        other\n            second input array. Must be compatible with x1 (with Broadcasting). May have any\n            data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type of bool.\n\n        Examples\n        --------\n        >>> x = ivy.array([6, 2, 3])\n        >>> y = ivy.array([4, 5, 3])\n        >>> z = x < y\n        >>> print(z)\n        ivy.array([ False, True, False])\n        '
        return ivy.less(self._data, other)

    def __le__(self, other):
        if False:
            print('Hello World!')
        '\n        ivy.Array special method variant of ivy.less_equal. This method simply wraps the\n        function, and so the docstring for ivy.less_equal also applies to this method\n        with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. May have any data type.\n        other\n            second input array. Must be compatible with x1 (with Broadcasting). May have any\n            data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type of bool.\n\n        Examples\n        --------\n        >>> x = ivy.array([6, 2, 3])\n        >>> y = ivy.array([4, 5, 3])\n        >>> z = x <= y\n        >>> print(z)\n        ivy.array([ False, True, True])\n        '
        return ivy.less_equal(self._data, other)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        '\n        ivy.Array special method variant of ivy.equal. This method simply wraps the\n        function, and so the docstring for ivy.equal also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. May have any data type.\n        other\n            second input array. Must be compatible with x1 (with Broadcasting). May have any\n            data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type of bool.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances:\n\n        >>> x1 = ivy.array([1, 0, 1, 1])\n        >>> x2 = ivy.array([1, 0, 0, -1])\n        >>> y = x1 == x2\n        >>> print(y)\n        ivy.array([True, True, False, False])\n\n        >>> x1 = ivy.array([1, 0, 1, 0])\n        >>> x2 = ivy.array([0, 1, 0, 1])\n        >>> y = x1 == x2\n        >>> print(y)\n        ivy.array([False, False, False, False])\n        '
        return ivy.equal(self._data, other)

    def __ne__(self, other):
        if False:
            return 10
        '\n        ivy.Array special method variant of ivy.not_equal. This method simply wraps the\n        function, and so the docstring for ivy.not_equal also applies to this method\n        with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. May have any data type.\n        other\n            second input array. Must be compatible with x1 (with Broadcasting). May have any\n            data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type of bool.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances:\n\n        >>> x1 = ivy.array([1, 0, 1, 1])\n        >>> x2 = ivy.array([1, 0, 0, -1])\n        >>> y = x1 != x2\n        >>> print(y)\n        ivy.array([False, False, True, True])\n\n        >>> x1 = ivy.array([1, 0, 1, 0])\n        >>> x2 = ivy.array([0, 1, 0, 1])\n        >>> y = x1 != x2\n        >>> print(y)\n        ivy.array([True, True, True, True])\n        '
        return ivy.not_equal(self._data, other)

    def __gt__(self, other):
        if False:
            return 10
        '\n        ivy.Array special method variant of ivy.greater. This method simply wraps the\n        function, and so the docstring for ivy.greater also applies to this method with\n        minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. May have any data type.\n        other\n            second input array. Must be compatible with x1 (with Broadcasting). May have any\n            data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type of bool.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances:\n\n        >>> x = ivy.array([6, 2, 3])\n        >>> y = ivy.array([4, 5, 3])\n        >>> z = x > y\n        >>> print(z)\n        ivy.array([True,False,False])\n\n        With mix of :class:`ivy.Array` and :class:`ivy.Container` instances:\n\n        >>> x = ivy.array([[5.1, 2.3, -3.6]])\n        >>> y = ivy.Container(a=ivy.array([[4.], [5.1], [6.]]),b=ivy.array([[-3.6], [6.], [7.]]))\n        >>> z = x > y\n        >>> print(z)\n        {\n            a: ivy.array([[True, False, False],\n                          [False, False, False],\n                          [False, False, False]]),\n            b: ivy.array([[True, True, False],\n                          [False, False, False],\n                          [False, False, False]])\n        }\n        '
        return ivy.greater(self._data, other)

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        '\n        ivy.Array special method variant of ivy.greater_equal. This method simply wraps\n        the function, and so the docstring for ivy.bitwise_xor also applies to this\n        method with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. May have any data type.\n        other\n            second input array. Must be compatible with x1 (with Broadcasting). May have any\n            data type.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type of bool.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances:\n\n        >>> x = ivy.array([6, 2, 3])\n        >>> y = ivy.array([4, 5, 6])\n        >>> z = x >= y\n        >>> print(z)\n        ivy.array([True,False,False])\n\n        With mix of :class:`ivy.Array` and :class:`ivy.Container` instances:\n\n        >>> x = ivy.array([[5.1, 2.3, -3.6]])\n        >>> y = ivy.Container(a=ivy.array([[4.], [5.1], [6.]]),b=ivy.array([[5.], [6.], [7.]]))\n        >>> z = x >= y\n        >>> print(z)\n        {\n            a: ivy.array([[True, False, False],\n                          [True, False, False],\n                          [False, False, False]]),\n            b: ivy.array([[True, False, False],\n                          [False, False, False],\n                          [False, False, False]])\n        }\n        '
        return ivy.greater_equal(self._data, other)

    def __and__(self, other):
        if False:
            return 10
        return ivy.bitwise_and(self._data, other)

    def __rand__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return ivy.bitwise_and(other, self._data)

    def __iand__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return ivy.bitwise_and(self._data, other)

    def __or__(self, other):
        if False:
            print('Hello World!')
        return ivy.bitwise_or(self._data, other)

    def __ror__(self, other):
        if False:
            i = 10
            return i + 15
        return ivy.bitwise_or(other, self._data)

    def __ior__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return ivy.bitwise_or(self._data, other)

    def __invert__(self):
        if False:
            for i in range(10):
                print('nop')
        return ivy.bitwise_invert(self._data)

    def __xor__(self, other):
        if False:
            print('Hello World!')
        '\n        ivy.Array special method variant of ivy.bitwise_xor. This method simply wraps\n        the function, and so the docstring for ivy.bitwise_xor also applies to this\n        method with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have an integer or boolean data type.\n        other\n            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).\n            Should have an integer or boolean data type.\n        out\n            optional output array, for writing the result to. It must have a shape that the\n            inputs broadcast to.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have a\n            data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances:\n\n        >>> a = ivy.array([1, 2, 3])\n        >>> b = ivy.array([3, 2, 1])\n        >>> y = a ^ b\n        >>> print(y)\n        ivy.array([2,0,2])\n\n        With mix of :class:`ivy.Array` and :class:`ivy.Container` instances:\n\n        >>> x = ivy.Container(a = ivy.array([-67, 21]))\n        >>> y = ivy.array([12, 13])\n        >>> z = x ^ y\n        >>> print(z)\n        {a: ivy.array([-79, 24])}\n        '
        return ivy.bitwise_xor(self._data, other)

    def __rxor__(self, other):
        if False:
            i = 10
            return i + 15
        return ivy.bitwise_xor(other, self._data)

    def __ixor__(self, other):
        if False:
            return 10
        return ivy.bitwise_xor(self._data, other)

    def __lshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return ivy.bitwise_left_shift(self._data, other)

    def __rlshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return ivy.bitwise_left_shift(other, self._data)

    def __ilshift__(self, other):
        if False:
            while True:
                i = 10
        return ivy.bitwise_left_shift(self._data, other)

    def __rshift__(self, other):
        if False:
            print('Hello World!')
        '\n        ivy.Array special method variant of ivy.bitwise_right_shift. This method simply\n        wraps the function, and so the docstring for ivy.bitwise_right_shift also\n        applies to this method with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have an integer data type.\n        other\n            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).\n            Should have an integer data type. Each element must be greater than or equal\n            to ``0``.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have\n            a data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        With :class:`ivy.Array` instances only:\n\n        >>> a = ivy.array([2, 3, 4])\n        >>> b = ivy.array([0, 1, 2])\n        >>> y = a >> b\n        >>> print(y)\n        ivy.array([2, 1, 1])\n        '
        return ivy.bitwise_right_shift(self._data, other)

    def __rrshift__(self, other):
        if False:
            return 10
        '\n        ivy.Array reverse special method variant of ivy.bitwise_right_shift. This method\n        simply wraps the function, and so the docstring for ivy.bitwise_right_shift also\n        applies to this method with minimal changes.\n\n        Parameters\n        ----------\n        self\n            first input array. Should have an integer data type.\n        other\n            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).\n            Should have an integer data type. Each element must be greater than or equal\n            to ``0``.\n\n        Returns\n        -------\n        ret\n            an array containing the element-wise results. The returned array must have\n            a data type determined by :ref:`type-promotion`.\n\n        Examples\n        --------\n        >>> a = 32\n        >>> b = ivy.array([0, 1, 2])\n        >>> y = a >> b\n        >>> print(y)\n        ivy.array([32, 16,  8])\n        '
        return ivy.bitwise_right_shift(other, self._data)

    def __irshift__(self, other):
        if False:
            return 10
        return ivy.bitwise_right_shift(self._data, other)

    def __deepcopy__(self, memodict={}):
        if False:
            for i in range(10):
                print('nop')
        try:
            return to_ivy(self._data.__deepcopy__(memodict))
        except AttributeError:
            if ivy.current_backend_str() == 'jax':
                np_array = copy.deepcopy(self._data)
                jax_array = ivy.array(np_array)
                return to_ivy(jax_array)
            return to_ivy(copy.deepcopy(self._data))
        except RuntimeError:
            from ivy.functional.ivy.gradients import _is_variable
            if _is_variable(self):
                return to_ivy(copy.deepcopy(ivy.stop_gradient(self)._data))
            return to_ivy(copy.deepcopy(self._data))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        if not len(self._data.shape):
            return 0
        try:
            return len(self._data)
        except TypeError:
            return self._data.shape[0]

    def __iter__(self):
        if False:
            print('Hello World!')
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d ivy.Array not supported')
        if ivy.current_backend_str() == 'paddle':
            if self.dtype in ['int8', 'int16', 'uint8', 'float16']:
                return iter([to_ivy(i) for i in ivy.unstack(self._data)])
            elif self.ndim == 1:
                return iter([to_ivy(i).squeeze(axis=0) for i in self._data])
        return iter([to_ivy(i) for i in self._data])