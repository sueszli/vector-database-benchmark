import paddle
from paddle.base import core
__all__ = []

def with_mateclass(meta, *bases):
    if False:
        print('Hello World!')

    class impl(meta):

        def __new__(cls, name, temp_bases, attrs):
            if False:
                i = 10
                return i + 15
            return meta(name, bases, attrs)
    return type.__new__(impl, 'impl', (), {})

class PyLayerContext:
    """
    ``PyLayerContext`` can assist the :ref:`api_paddle_autograd_PyLayer` in implementing certain functionalities.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.autograd import PyLayer

            >>> class cus_tanh(PyLayer):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         # ctx is a object of PyLayerContext.
            ...         y = paddle.tanh(x)
            ...         ctx.save_for_backward(y)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         # ctx is a object of PyLayerContext.
            ...         y, = ctx.saved_tensor()
            ...         grad = dy * (1 - paddle.square(y))
            ...         return grad
    """

    def save_for_backward(self, *tensors):
        if False:
            i = 10
            return i + 15
        '\n        Saves given tensors that backward need. Use ``saved_tensor`` in the `backward` to get the saved tensors.\n\n        Note:\n            This API should be called at most once, and only inside `forward`.\n\n        Args:\n            tensors(list of Tensors): Tensors to be stored.\n\n        Returns:\n            None\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> from paddle.autograd import PyLayer\n\n                >>> class cus_tanh(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         # ctx is a context object that store some objects for backward.\n                ...         y = paddle.tanh(x)\n                ...         # Pass tensors to backward.\n                ...         ctx.save_for_backward(y)\n                ...         return y\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, dy):\n                ...         # Get the tensors passed by forward.\n                ...         y, = ctx.saved_tensor()\n                ...         grad = dy * (1 - paddle.square(y))\n                ...         return grad\n\n        '
        self.container = tensors

    def saved_tensor(self):
        if False:
            while True:
                i = 10
        '\n        Get the tensors stored by ``save_for_backward``.\n\n        Returns:\n            list of Tensors or None: If context contains tensors stored by `save_for_backward`,\n            then return these tensors, otherwise return None.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> from paddle.autograd import PyLayer\n\n                >>> class cus_tanh(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         # ctx is a context object that store some objects for backward.\n                ...         y = paddle.tanh(x)\n                ...         # Pass tensors to backward.\n                ...         ctx.save_for_backward(y)\n                ...         return y\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, dy):\n                ...         # Get the tensors passed by forward.\n                ...         y, = ctx.saved_tensor()\n                ...         grad = dy * (1 - paddle.square(y))\n                ...         return grad\n        '
        return self.container

    def mark_not_inplace(self, *args):
        if False:
            while True:
                i = 10
        '\n        Marks inputs as not inplace.\n        This should be called at most once, only from inside the `forward` method,\n        and all arguments should be Tensor inputs.\n\n        If the Tensor returned by `forward` method is the same as the Tensor input of forward,\n        and this Tensor is marked as not_inplace, then Paddle will help the user create a new Tensor as output.\n        Thereby preventing the auto grad information of the input Tensor from being overwritten.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n\n                >>> class Exp(paddle.autograd.PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         ctx.mark_not_inplace(x)\n                ...         return x\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, grad_output):\n                ...         out = grad_output.exp()\n                ...         return out\n\n                >>> paddle.seed(2023)\n                >>> x = paddle.randn((1, 1))\n                >>> x.stop_gradient = False\n                >>> attn_layers = []\n                >>> for idx in range(0, 2):\n                ...     attn_layers.append(Exp())\n\n                >>> for step in range(0, 2):\n                ...     a = x\n                ...     for j in range(0,2):\n                ...         a = attn_layers[j].apply(x)\n                ...     a.backward()\n        '
        self.not_inplace_tensors = args

    def mark_non_differentiable(self, *args):
        if False:
            while True:
                i = 10
        '\n        Marks outputs as non-differentiable.\n        This should be called at most once, only from inside the `forward` method,\n        and all arguments should be tensor outputs.\n\n        This will mark outputs as not requiring gradients, increasing the\n        efficiency of backward computation. You still need to accept a gradient\n        for each output in `backward`, but it\'s always going to\n        be a zero tensor with the same shape as the shape of a corresponding\n        output.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> from paddle.autograd import PyLayer\n                >>> import numpy as np\n\n                >>> class Tanh(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         a = x + x\n                ...         b = x + x + x\n                ...         ctx.mark_non_differentiable(a)\n                ...         return a, b\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, grad_a, grad_b):\n                ...         assert np.equal(grad_a.numpy(), paddle.zeros([1]).numpy())\n                ...         assert np.equal(grad_b.numpy(), paddle.ones([1], dtype="float64").numpy())\n                ...         return grad_b\n\n                >>> x = paddle.ones([1], dtype="float64")\n                >>> x.stop_gradient = False\n                >>> a, b = Tanh.apply(x)\n                >>> b.sum().backward()\n        '
        self.non_differentiable = args

    def set_materialize_grads(self, value: bool):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets whether to materialize output grad tensors. Default is True.\n\n        This should be called only from inside the `forward` method.\n\n        If True, undefined output grad tensors will be expanded to tensors full\n        of zeros prior to calling the `backward` method.\n\n        If False, undefined output grad tensors will be None.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> from paddle.autograd import PyLayer\n                >>> import numpy as np\n\n                >>> class Tanh(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         return x+x+x, x+x\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, grad, grad2):\n                ...         assert np.equal(grad2.numpy(), paddle.zeros([1]).numpy())\n                ...         return grad\n\n                >>> class Tanh2(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         ctx.set_materialize_grads(False)\n                ...         return x+x+x, x+x\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, grad, grad2):\n                ...         assert grad2==None\n                ...         return grad\n\n                >>> x = paddle.ones([1], dtype="float64")\n                >>> x.stop_gradient = False\n                >>> Tanh.apply(x)[0].backward()\n\n                >>> x2 = paddle.ones([1], dtype="float64")\n                >>> x2.stop_gradient = False\n                >>> Tanh2.apply(x2)[0].backward()\n        '
        self.materialize_grads = value

class PyLayerBackward(core.eager.PyLayer, PyLayerContext):

    def backward(self, *args):
        if False:
            return 10
        return self._forward_cls.backward(self, *args)

class PyLayerMeta(type):

    def __init__(cls, name, bases, attrs):
        if False:
            return 10
        cls._backward_function = type(name + '_backward', (PyLayerBackward,), {'_forward_cls': cls})
        return super().__init__(name, bases, attrs)

class PyLayer(with_mateclass(PyLayerMeta, core.eager.PyLayer, PyLayerContext)):
    """
    Paddle implements Python custom operators on the PaddlePaddle framework by creating a subclass of
    ``PyLayer``, which must comply with the following rules:

    1. The subclass must contain static ``forward`` and ``backward`` functions, with the first argument being
    :ref:`api_paddle_autograd_PyLayerContext`. If a returned value in ``backward`` corresponds to a ``Tensor`` that
    requires gradients in ``forward``, the returned value must be a ``Tensor``.

    2. Except for the first argument, other arguments of ``backward`` are gradients of the output ``Tensors``
    of ``forward``. Therefore, the number of input ``Tensor`` in ``backward`` must be the same as the number
    of output ``Tensor`` in ``forward``. If you need to use input ``Tensor`` from ``forward`` in ``backward``,
    you can save these ``Tensors`` by inputting them into :ref:`api_paddle_autograd_PyLayerContext`'s
    ``save_for_backward`` method and use them in ``backward`` later.

    3. The output of ``backward`` can be ``Tensor`` or ``list/tuple(Tensor)``, which are gradients of the
    output ``Tensor`` of ``forward``. Therefore, the number of output ``Tensor`` in ``backward`` is the same
    as the number of input ``Tensor`` in ``forward``.

    After building the custom operator, apply it by running the ``apply`` method.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.autograd import PyLayer

            >>> class cus_tanh(PyLayer):
            ...     @staticmethod
            ...     def forward(ctx, x):
            ...         y = paddle.tanh(x)
            ...         # Pass tensors to backward.
            ...         ctx.save_for_backward(y)
            ...         return y
            ...
            ...     @staticmethod
            ...     def backward(ctx, dy):
            ...         # Get the tensors passed by forward.
            ...         y, = ctx.saved_tensor()
            ...         grad = dy * (1 - paddle.square(y))
            ...         return grad

            >>> paddle.seed(2023)
            >>> data = paddle.randn([2, 3], dtype="float64")
            >>> data.stop_gradient = False
            >>> z = cus_tanh.apply(data)
            >>> z.mean().backward()

            >>> print(data.grad)
            Tensor(shape=[2, 3], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[0.16604150, 0.05858341, 0.14051214],
             [0.15677770, 0.01564609, 0.02991660]])
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        It is to be overloaded by subclasses. It must accept a object of :ref:`api_paddle_autograd_PyLayerContext` as\n        the first argument, followed by any number of arguments (tensors or other types).\n        `None` can not be included in the returned result.\n\n        Args:\n            *args(tuple): input of PyLayer.\n            **kwargs(dict): input of PyLayer.\n\n        Returns:\n            tensors or other types : output of PyLayer.\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> from paddle.autograd import PyLayer\n\n                >>> class cus_tanh(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         y = paddle.tanh(x)\n                ...         # Pass tensors to backward.\n                ...         ctx.save_for_backward(y)\n                ...         return y\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, dy):\n                ...         # Get the tensors passed by forward.\n                ...         y, = ctx.saved_tensor()\n                ...         grad = dy * (1 - paddle.square(y))\n                ...         return grad\n        '
        raise NotImplementedError('You must implement the forward function for PyLayer.')

    @staticmethod
    def backward(ctx, *args):
        if False:
            for i in range(10):
                print('nop')
        "\n        This is a function to calculate the gradient. It is to be overloaded by subclasses.\n        It must accept a object of :ref:`api_paddle_autograd_PyLayerContext` as the first\n        argument, and the rest arguments are the gradient of forward's output tensors.\n        Output tensors of backward are the gradient of forward's input tensors.\n\n        Args:\n            *args(tuple): The gradient of forward's output tensor(s).\n            **kwargs(dict): The gradient of forward's output tensor(s).\n\n        Returns:\n            Tensor or list of Tensors: The gradient of forward's input tensor(s).\n\n        Examples:\n            .. code-block:: python\n\n                >>> import paddle\n                >>> from paddle.autograd import PyLayer\n\n                >>> class cus_tanh(PyLayer):\n                ...     @staticmethod\n                ...     def forward(ctx, x):\n                ...         y = paddle.tanh(x)\n                ...         # Pass tensors to backward.\n                ...         ctx.save_for_backward(y)\n                ...         return y\n                ...\n                ...     @staticmethod\n                ...     def backward(ctx, dy):\n                ...         # Get the tensors passed by forward.\n                ...         y, = ctx.saved_tensor()\n                ...         grad = dy * (1 - paddle.square(y))\n                ...         return grad\n        "
        raise NotImplementedError('You must implement the backward function for PyLayer.')

def once_differentiable(backward):
    if False:
        i = 10
        return i + 15

    def wrapper(ctx, *args):
        if False:
            print('Hello World!')
        with paddle.base.dygraph.no_grad():
            outputs = backward(ctx, *args)
        return outputs
    return wrapper