import warnings
from paddle.base.libpaddle import DataType
from . import OpResult
_already_patch_opresult = False
_supported_int_dtype_ = [DataType.BOOL, DataType.UINT8, DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]

def create_tensor_with_batchsize(ref_var, value, dtype):
    if False:
        return 10
    assert isinstance(ref_var, OpResult)
    value = float(value)
    batch_dim = -1
    out_shape = []
    for (i, d) in enumerate(ref_var.shape):
        if d < 0:
            if batch_dim < 0:
                batch_dim = i
                out_shape.append(d)
            else:
                out_shape.append(1)
        else:
            out_shape.append(d)
    assert batch_dim != -1
    from paddle import _C_ops
    from paddle.framework import core
    out = _C_ops.full_batch_size_like(ref_var, out_shape, dtype, value, batch_dim, batch_dim, core.Place())
    out.stop_gradient = True
    return out

def monkey_patch_opresult():
    if False:
        for i in range(10):
            print('nop')

    def safe_get_dtype(var):
        if False:
            for i in range(10):
                print('nop')
        try:
            dtype = var.dtype
        except:
            raise ValueError('Cannot get data type from var')
        return dtype

    def place(self):
        if False:
            return 10
        "\n        OpResult don't have 'place' interface in static graph mode\n        But this interface can greatly facilitate dy2static.\n        So we give a warnning here and return None.\n        "
        warnings.warn("OpResult do not have 'place' interface for pir graph mode, try not to use it. None will be returned.")

    @property
    def _ndim(self):
        if False:
            while True:
                i = 10
        "\n        Returns the dimension of current OpResult\n\n        Returns:\n            the dimension\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> paddle.enable_static()\n\n                >>> # create a static OpResult\n                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])\n                >>> # print the dimension of the OpResult\n                >>> print(x.ndim)\n                3\n        "
        return len(self.shape)

    def ndimension(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns the dimension of current OpResult\n\n        Returns:\n            the dimension\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> paddle.enable_static()\n\n                >>> # create a static OpResult\n                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])\n                >>> # print the dimension of the OpResult\n                >>> print(x.ndimension())\n                3\n        "
        return len(self.shape)

    def dim(self):
        if False:
            while True:
                i = 10
        "\n        Returns the dimension of current OpResult\n\n        Returns:\n            the dimension\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> paddle.enable_static()\n\n                >>> # create a static OpResult\n                >>> x = paddle.static.data(name='x', shape=[3, 2, 1])\n                >>> # print the dimension of the OpResult\n                >>> print(x.dim())\n                3\n        "
        return len(self.shape)

    def _item(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        In order to be compatible with the item interface introduced by the dynamic graph, it does nothing but returns self.\n        It will check that the shape must be a 1-D tensor\n        '
        if len(self.shape) > 1:
            raise TypeError(f'Required input var should be 1-D OpResult, but received {self.shape}')
        return self

    def astype(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        '\n        **Notes**:\n\n        Cast a OpResult to a specified data type.\n\n        Args:\n\n            self(OpResult): The source OpResult\n\n            dtype: The target data type\n\n        Returns:\n            OpResult: OpResult with new dtype\n\n        Examples:\n            In Static Graph Mode:\n\n            .. code-block:: python\n\n                >>> import paddle\n                >>> paddle.enable_static()\n                >>> startup_prog = paddle.static.Program()\n                >>> main_prog = paddle.static.Program()\n                >>> with paddle.static.program_guard(startup_prog, main_prog):\n                ...     original_value = paddle.static.data(name = "new_value", shape=[2,2], dtype=\'float32\')\n                ...     new_value = original_value.astype(\'int64\')\n                ...     print("new value\'s dtype is: {}".format(new_value.dtype))\n                ...\n                new OpResult\'s dtype is: paddle.int64\n\n        '
        from paddle import _C_ops
        if not isinstance(dtype, DataType):
            dtype = paddle.pir.core.convert_np_dtype_to_dtype_(dtype)
        return _C_ops.cast(self, dtype)

    def _scalar_add_(var, value):
        if False:
            while True:
                i = 10
        return paddle.scale(var, 1.0, value)

    def _scalar_sub_(var, value):
        if False:
            i = 10
            return i + 15
        return paddle.scale(var, 1.0, -value)

    def _scalar_rsub_(var, value):
        if False:
            return 10
        return paddle.scale(var, -1.0, value)

    def _scalar_mul_(var, value):
        if False:
            i = 10
            return i + 15
        return paddle.scale(var, value, 0.0)

    def _scalar_div_(var, value):
        if False:
            print('Hello World!')
        return paddle.scale(var, 1.0 / value, 0.0)

    def _binary_creator_(method_name, python_api, reverse=False, scalar_method=None):
        if False:
            return 10

        def __impl__(self, other_var):
            if False:
                i = 10
                return i + 15
            if isinstance(other_var, float):
                if self.dtype in _supported_int_dtype_:
                    self = astype(self, DataType.FLOAT32)
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            elif isinstance(other_var, int):
                other_var = float(other_var)
                if python_api == paddle.divide and self.dtype in _supported_int_dtype_:
                    paddle.cast(self, DataType.FLOAT32)
                if scalar_method is not None:
                    return scalar_method(self, other_var)
            else:
                pass
            lhs_dtype = safe_get_dtype(self)
            other_var_opresult = other_var
            if not isinstance(other_var, OpResult):
                if reverse:
                    for elem in self.shape:
                        if elem < 0:
                            other_var_opresult = create_tensor_with_batchsize(self, other_var, lhs_dtype)
                            break
                    else:
                        other_var_opresult = paddle.tensor.creation.fill_constant(self.shape, lhs_dtype, other_var)
                else:
                    other_var_opresult = paddle.tensor.creation.fill_constant([], lhs_dtype, other_var)
            rhs_dtype = safe_get_dtype(other_var_opresult)
            if lhs_dtype != rhs_dtype:
                other_var_opresult = paddle.cast(other_var_opresult, lhs_dtype)
            if reverse:
                tmp = self
                self = other_var_opresult
                other_var_opresult = tmp
            if python_api == paddle.divide and self.dtype in _supported_int_dtype_:
                self = paddle.cast(self, DataType.FLOAT32)
                other_var_opresult = paddle.cast(other_var_opresult, DataType.FLOAT32)
            out = python_api(self, other_var_opresult)
            return out
        __impl__.__doc__ = '\n            Args:\n                self(OpResult): left hand OpResult\n                other_var(OpResult|float|int): right hand OpResult\n\n            Returns:\n                OpResult\n            '
        __impl__.__name__ = method_name
        return __impl__
    import paddle
    opresult_methods = [('place', place), ('item', _item), ('dim', dim), ('ndimension', ndimension), ('ndim', _ndim), ('astype', astype), ('__add__', _binary_creator_('__add__', paddle.tensor.add, False, _scalar_add_)), ('__radd__', _binary_creator_('__radd__', paddle.tensor.add, False, _scalar_add_)), ('__sub__', _binary_creator_('__sub__', paddle.tensor.subtract, False, _scalar_sub_)), ('__rsub__', _binary_creator_('__rsub__', paddle.tensor.subtract, True, _scalar_rsub_)), ('__mul__', _binary_creator_('__mul__', paddle.tensor.multiply, False, _scalar_mul_)), ('__rmul__', _binary_creator_('__rmul__', paddle.tensor.multiply, False, _scalar_mul_)), ('__div__', _binary_creator_('__div__', paddle.tensor.divide, False, _scalar_div_)), ('__truediv__', _binary_creator_('__truediv__', paddle.tensor.divide, False, _scalar_div_)), ('__rdiv__', _binary_creator_('__rdiv__', paddle.tensor.divide, True, None)), ('__rtruediv__', _binary_creator_('__rtruediv__', paddle.tensor.divide, True, None)), ('__pow__', _binary_creator_('__pow__', paddle.tensor.pow, False, None)), ('__rpow__', _binary_creator_('__rpow__', paddle.tensor.pow, True, None)), ('__floordiv__', _binary_creator_('__floordiv__', paddle.tensor.floor_divide, False, None)), ('__mod__', _binary_creator_('__mod__', paddle.tensor.remainder, False, None)), ('__matmul__', _binary_creator_('__matmul__', paddle.tensor.matmul, False, None)), ('__ne__', _binary_creator_('__ne__', paddle.tensor.not_equal, False, None)), ('__lt__', _binary_creator_('__lt__', paddle.tensor.less_than, False, None)), ('__le__', _binary_creator_('__le__', paddle.tensor.less_equal, False, None)), ('__gt__', _binary_creator_('__gt__', paddle.tensor.greater_than, False, None)), ('__ge__', _binary_creator_('__ge__', paddle.tensor.greater_equal, False, None))]
    global _already_patch_opresult
    if not _already_patch_opresult:
        for method in opresult_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(OpResult, method_name, method_impl)
        import paddle.tensor
        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(OpResult, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(OpResult, method_name, method_impl)
        for (magic_method, origin_method) in paddle.tensor.magic_method_func:
            impl = getattr(paddle.tensor, origin_method, None)
            if impl:
                setattr(OpResult, magic_method, impl)
        from ..base.variable_index import _getitem_static
        OpResult.__getitem__ = _getitem_static
        _already_patch_opresult = True