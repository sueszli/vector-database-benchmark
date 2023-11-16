import operator
from ._op_reqs import *
from ._utils import promoted_primitive_type, broadcast_shapes
'\nElementwise Binary Op Superclass\n'

class elementwise_binary(Operation):
    input_spec = InputSpec(x=ScalarOrTensorInputType(), y=ScalarOrTensorInputType())

    def __init__(self, **kwargs):
        if False:
            return 10
        super(elementwise_binary, self).__init__(**kwargs)

    def type_inference(self):
        if False:
            i = 10
            return i + 15
        typea = self.x.sym_type
        typeb = self.y.sym_type
        primitive_type = promoted_primitive_type(typea, typeb)
        if primitive_type is None:
            raise ValueError('Incompatible primitive types in broadcast operation')
        primitive_type = self.get_dtype(primitive_type)
        if not types.is_tensor(typea) and (not types.is_tensor(typeb)):
            return primitive_type
        if types.is_tensor(typea) and (not types.is_tensor(typeb)):
            return types.tensor(primitive_type, typea.get_shape())
        if not types.is_tensor(typea) and types.is_tensor(typeb):
            return types.tensor(primitive_type, typeb.get_shape())
        shapea = list(typea.get_shape())
        shapeb = list(typeb.get_shape())
        ret_shape = broadcast_shapes(shapea, shapeb)
        return types.tensor(primitive_type, ret_shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        if False:
            return 10
        return self._cast_check_value_inferene(self.x.val, self.y.val)

    def get_operator(self):
        if False:
            i = 10
            return i + 15
        '\n        All subclasses have to implement this.\n        '
        raise NotImplementedError()

    def get_dtype(self, promoted_dtype):
        if False:
            print('Hello World!')
        '\n        Override if output primitive type is different from input types\n        (e.g., less, greater)\n        '
        return promoted_dtype

    def _cast_check_value_inferene(self, a, b):
        if False:
            while True:
                i = 10
        '\n        If one of the input is tensor, cast the result to tensor.\n        '
        to_cast = any([isinstance(x, np.ndarray) for x in [a, b]])
        result = self.get_operator()(a, b)
        return result if not to_cast else np.array(result)
'\nElementwise Binary Op Implmentation(s)\n'

@register_op(doc_str='')
class add(elementwise_binary):
    """
    Add two inputs element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor of the same type and shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(add, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            i = 10
            return i + 15
        return operator.add

@register_op(doc_str='')
class equal(elementwise_binary):
    """
    Return ``x==y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(equal, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            i = 10
            return i + 15
        return np.equal

    def get_dtype(self, promoted_dtype):
        if False:
            i = 10
            return i + 15
        return types.bool

@register_op(doc_str='')
class floor_div(elementwise_binary):
    """
    Return the floor_div values of two inputs element-wise.
    That is the largest integer ``t``, and ``t <= x/y``.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor of the same type and shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(floor_div, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            i = 10
            return i + 15
        return operator.floordiv

@register_op(doc_str='')
class greater(elementwise_binary):
    """
    Return ``x > y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(greater, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            for i in range(10):
                print('nop')
        return operator.gt

    def get_dtype(self, promoted_dtype):
        if False:
            while True:
                i = 10
        return types.bool

@register_op(doc_str='')
class greater_equal(elementwise_binary):
    """
    Return ``x >= y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(greater_equal, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            while True:
                i = 10
        return operator.ge

    def get_dtype(self, promoted_dtype):
        if False:
            return 10
        return types.bool

@register_op(doc_str='')
class less(elementwise_binary):
    """
    Return ``x < y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(less, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            while True:
                i = 10
        return operator.lt

    def get_dtype(self, promoted_dtype):
        if False:
            print('Hello World!')
        return types.bool

@register_op(doc_str='')
class less_equal(elementwise_binary):
    """
    Return ``x <= y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(less_equal, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            print('Hello World!')
        return operator.le

    def get_dtype(self, promoted_dtype):
        if False:
            i = 10
            return i + 15
        return types.bool

@register_op(doc_str='')
class logical_and(elementwise_binary):
    """
    Return ``x & y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, bool> (Required)
    y: tensor<*?, bool> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(logical_and, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            i = 10
            return i + 15
        return np.logical_and

    def get_dtype(self, promoted_dtype):
        if False:
            i = 10
            return i + 15
        return types.bool

@register_op(doc_str='')
class logical_or(elementwise_binary):
    """
    Return ``x || y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, bool> (Required)
    y: tensor<*?, bool> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(logical_or, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            return 10
        return np.logical_or

    def get_dtype(self, promoted_dtype):
        if False:
            for i in range(10):
                print('nop')
        return types.bool

@register_op(doc_str='')
class logical_xor(elementwise_binary):
    """
    Return ``x ^ y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, bool> (Required)
    y: tensor<*?, bool> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(logical_xor, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            while True:
                i = 10
        return np.logical_xor

    def get_dtype(self, promoted_dtype):
        if False:
            print('Hello World!')
        return types.bool

@register_op(doc_str='')
class maximum(elementwise_binary):
    """
    Return ``max(x,y)`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(maximum, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            print('Hello World!')
        return np.maximum

@register_op(doc_str='')
class minimum(elementwise_binary):
    """
    Return ``min(x,y)`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(minimum, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            i = 10
            return i + 15
        return np.minimum

@register_op(doc_str='')
class mod(elementwise_binary):
    """
    Return ``x % y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(mod, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            for i in range(10):
                print('nop')
        return operator.mod

@register_op(doc_str='')
class mul(elementwise_binary):
    """
    Return ``x * y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(mul, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            return 10
        return operator.mul

@register_op(doc_str='')
class not_equal(elementwise_binary):
    """
    Return ``x != y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, bool> (Required)
    y: tensor<*?, bool> (Required)

    Returns
    -------
    tensor<*?, bool>
        * a bool tensor with the same shape as inputs.
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(not_equal, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            for i in range(10):
                print('nop')
        return operator.ne

    def get_dtype(self, promoted_dtype):
        if False:
            print('Hello World!')
        return types.bool

@register_op(doc_str='')
class real_div(elementwise_binary):
    """
    Return the true division ``x / y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(real_div, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            print('Hello World!')
        return operator.truediv

    def get_dtype(self, promoted_dtype):
        if False:
            for i in range(10):
                print('nop')
        return types.float

@register_op(doc_str='')
class pow(elementwise_binary):
    """
    Return ``pow(x,y)`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(pow, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            return 10
        return operator.pow

@register_op(doc_str='')
class sub(elementwise_binary):
    """
    Return ``x - y`` element-wise.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
    y: tensor<*?, T> (Required)

    Returns
    -------
    tensor<*?, T>
        * a tensor with the same shape and type as inputs.

    Attributes
    ----------
    T: fp32
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(sub, self).__init__(**kwargs)

    def get_operator(self):
        if False:
            return 10
        return operator.sub