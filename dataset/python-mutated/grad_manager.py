import weakref
from typing import Callable, Iterable, List, Union
from ..core._imperative_rt.core2 import get_auto_format_convert, pop_scope, push_scope, set_auto_format_convert, set_option
from ..core.autodiff.grad import Grad
from ..core.tensor.dtype import is_differentible_dtype
from ..logger import get_logger
from ..tensor import Tensor
from ..utils.future import Future
logger = get_logger(__name__)
backwarding_grad_manager = None

def get_backwarding_grad_manager():
    if False:
        print('Hello World!')
    return backwarding_grad_manager

class AttachSpec:
    __slots__ = ('tensor', 'callbacks')

class GradManager:
    """GradManager computes gradients or more generally, vector-Jacobian product, by reverse mode
    automatic differentiation (a.k.a. back propagation).

    Reverse mode autodiff normally reuses many intermediate tensors for best computation efficiency.
    In a read-eval-print-loop (REPL) environment however, it is impossible to known how the user
    would take gradients later thus which tensors to keep. To solve this problem, the user must
    somehow declare beforehand which gradient could possibly be taken. With GradManager, users are
    required to call the :meth:`attach` method on a tensor if they want to take gradients with
    respect to it later. Furthermore, any computation on a tensor before it is attached is
    completely ignored from the autodiff perspective, so :meth:`attach` must be called before any
    computation that needs differentiation.

    For example, the following symbolic differentiation code

    .. code-block::

        x = get_x()
        y = f(x)
        dy = ones_like(y)
        dx = vjp(y, x, dy) # vector-Jacobian product

    can be rewriten using GradManager for REPL environment as

    .. code-block::

        with GradManager() as gm:
            x = get_x()
            gm.attach(x) # must be placed before any computation on x that needs differentiation
            y = f(x)
            dy = ones_like(y)
            gm.backward(y, dy) # doesn't need x, already known via attach()
            dx = x.grad # backward() saves result to .grad attribute

    A more realistic example of training a neural network would be like

    .. code-block::

        gm = GradManager()
        gm.attach(model.parameters())

        for data in dataset:
            with gm:
                loss = model(data)
                gm.backward(loss)
            # gradients w.r.t. parameters is accumulated into their .grad attributes

    You can also use ``record()`` and ``release()`` method instead of ``with`` context:

    .. code-block::

        gm = GradManager()
        gm.attach(model.parameters())

        for data in dataset:
            gm.record()
            loss = model(data)
            gm.backward(loss)
            # backward() will clear recorded history and free resources
            # call release() if backward() is not called
            # gm.release()

    For your convenience, GradManager may (not must) be reused. As shown in the examples, you
    only need to attach a tensor once and GradManager will remember it afterwards.
    However, a single GradManager can record only one computation history at a time. To run
    multiple differentiations simultaneously or perform high order differentiation, create
    as many GradManager as you need.

    .. note::

        Mutable tensors introduce ambiguities when doing symbolic differentiation: which version
        of the tensor are we referring to? For attached tensors, GradManager resolves this
        ambiguity by "snapshoting" them on first encounter, either on :meth:`record` (or entering
        with statement) if tensor is attached before :meth:`record`, or on :meth:`attach` if
        GradManager is already recording. Attached tensors will then be interpreted as their
        snapshotted version for differentiation purpose. The same ambiguity on the first parameter
        of :meth:`backward` is simply resolved by using the latest version.

    Typically, in data parallel, we would like to average the gradients across
    processes. Users will finally get the averaged gradients if an "AllReduce"
    callback is registered as follows:

    .. code-block::

        import megengine.distributed as dist

        gm = GradManager()
        gm.attach(model.parameters(), callback=dist.make_allreduce_cb("MEAN"))
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._attach_specs = {}
        self._recording = False
        self._grad = None
        self._after_backward_callback = []
        self._gradients = {}

    def attached_tensors(self):
        if False:
            return 10
        'Return attached tensor list from :meth:`attach`.'
        return [spec.tensor() for spec in self._attach_specs.values()]

    def attach(self, tensors: Iterable[Tensor], callbacks=None):
        if False:
            i = 10
            return i + 15
        'Instruct GradManager to track operations on tensors, so that gradients with respect\n        to those tensors could be evaluated later.\n\n        :meth:`attach` also accepts a list of callbacks, which will be called with the tensor and\n        its gradient during :meth:`backward`. The signature of callbacks should look like:\n\n            .. code-block::\n\n                def callback(tensor: Tensor, grad: Tensor) -> Tensor:\n                    ...\n                    # returned grad is passed to subsequent callbacks\n                    # and finally accumulated to the .grad attribute of tensor\n                    return grad\n\n        :meth:`attach` calls with overlapping tensors will result in their callbacks concatenated,\n        independently for each tensor. For example,\n\n            .. code-block::\n\n                gm.attach([x, y], callbacks=[f])\n                gm.attach([y], callbacks=[g])\n\n        is equivalent to\n\n            .. code-block::\n\n                gm.attach([x], callbacks=[f])\n                gm.attach([y], callbacks=[f, g])\n\n        The effect of :meth:`attach` will persist across multiple uses of the GradManager. When\n        reusing a GradManager, it is likely a mistake to call :meth:`attach` on the same set of\n        tensors and callbacks repeatedly, which may grow the callback list indefinitely.\n\n        .. note::\n\n            When reusing a GradManager, it is sometimes desirable to attach temporary tensors each\n            time, e.g. for computing gradients of inputs of a neural network. GradManager tries to\n            accommodate such usages by holding weak references to attached tensors. Most of the\n            times, this should be enough to prevent resource leak. Unfortunately, there are still\n            some pitfalls left:\n\n                - Callbacks should not hold strong references, directly or indirectly, to attached\n                  tensors. Any strong reference, including those from callbacks, will prevent\n                  garbage collection (even by the cycle collector!) of a attached tensor, until\n                  the GradManager object is garbage collected.\n\n            Please also note that GradManager might hold additional strong references to attached\n            tensors when it is in use. This note only covers potential resource leaks across\n            multiple uses of a GradManager, which is unrelated to whether resources is timely\n            released within a single use.\n\n        Args:\n            tensors: tensor or list of tensors to track\n            callbacks: callback or list of callbacks\n        '
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callable):
            callbacks = [callbacks]
        if isinstance(tensors, Tensor):
            tensors = [tensors]

        def make_spec(tensor):
            if False:
                i = 10
                return i + 15
            selfref = weakref.ref(self)
            key = id(tensor)

            def deleter(_):
                if False:
                    print('Hello World!')
                self = selfref()
                if self is not None:
                    del self._attach_specs[key]
            spec = AttachSpec()
            spec.tensor = weakref.ref(tensor, deleter)
            spec.callbacks = []
            return spec
        for x in tensors:
            assert isinstance(x, Tensor), 'Object to be attached should be Tensor'
            assert is_differentible_dtype(x.dtype), 'Only tensors of floating point dtype can be attached to get gradients, get tensor dtype: {} and shape: {}'.format(x.dtype, x.shape)
            spec = self._attach_specs.get(id(x))
            new_attach = spec is None
            if spec is None:
                spec = make_spec(x)
                self._attach_specs[id(x)] = spec
            spec.callbacks.extend(callbacks)
            if new_attach and self._recording:
                self._do_record(spec)
        return self

    def _register_after_backward_callback(self, callback):
        if False:
            i = 10
            return i + 15
        self._after_backward_callback.append(callback)
        return self

    def backward(self, y: Union[Tensor, List[Tensor]]=None, dy: Union[Tensor, List[Tensor]]=None):
        if False:
            while True:
                i = 10
        'Compute gradients (or vector-Jacobian product) for all attached tensors, accumulate to\n        corresponding .grad attribute, and release resources along the way.\n\n        :meth:`backward` computes the vector-Jacobian product :math:`dx_j = \\sum_{i} dy_i J_{ij}`\n        where :math:`J_{ij} = ∂y_i/∂x_j` is the Jacobian matrix between vector variables :math:`y`\n        and :math:`x`, with all vectors involved represented as a list of tensors, in the sense of\n        direct sums (or flatten-and-concatenate). :math:`y` and :math:`dy` are passed as the first\n        and second parameter respectively, whereas :math:`x` is directly taken from the list of\n        all attached tensors. The result :math:`dx` is also not returned. Instead, it is directly\n        accumulated into the .grad attribute of matching attached tensors (a.k.a. :math:`x`). This\n        can be done unambiguously since :math:`dx` as a list of tensors has the same structure as\n        :math:`x`.\n\n        If :math:`y` is a scalar and :math:`dy` is chosen to be 1, the vector-Jacobian product\n        yield gradient of :math:`y` with repect to :math:`x` as a special case. In that case,\n        you will be able to omit the :math:`dy` parameter and :meth:`backward` will automatically\n        use 1 for it and compute the gradient.\n\n        :meth:`backward` consumes all resources held by this GradManager and releases them in the\n        process of this call. When the call successfully finishes, the GradManager will be put back\n        to an inactive state.\n\n        Args:\n            y: tensor or list of tensors\n            dy: tensor or list of tensors. Defaults to 1 if y is scalar\n        '
        push_scope('backward')
        set_option('record_computing_path', 0)
        _origin_auto_format = get_auto_format_convert()
        from ..functional import ones_like
        global backwarding_grad_manager
        cache = backwarding_grad_manager
        backwarding_grad_manager = self
        if not self._recording:
            raise RuntimeError('no computation history. did you forget record() or call a method that clears the history?')
        assert self._grad is not None
        if y is None:
            ys = []
        elif isinstance(y, (tuple, list)):
            ys = y
        else:
            ys = [y]
        if dy is None:
            dys = [ones_like(y) for y in ys]
        elif isinstance(dy, (tuple, list)):
            dys = dy
        else:
            dys = [dy]
        try:
            self._grad(ys, dys)
            for callback in self._after_backward_callback:
                callback()
            for (id_, grad) in self._gradients.items():
                if isinstance(grad, Future):
                    grad = grad.get()
                spec = self._attach_specs.get(id_)
                tensor = spec and spec.tensor()
                if tensor is not None:
                    if tensor.grad is None:
                        tensor.grad = grad
                    else:
                        tensor.grad += grad
        finally:
            self.release()
            backwarding_grad_manager = cache
            set_option('record_computing_path', 1)
            pop_scope('backward')

    def record(self):
        if False:
            for i in range(10):
                print('nop')
        'Start recording operations\n\n        After this call, you will be able to call :meth:`backward`.\n        '
        if self._recording:
            raise RuntimeError('already recording')
        grad = Grad()
        self._recording = True
        self._grad = grad
        grad.__enter__()
        for spec in self._attach_specs.values():
            self._do_record(spec)

    def _do_record(self, spec):
        if False:
            i = 10
            return i + 15
        tensor = spec.tensor()
        if tensor is None:
            return

        def callback(grad, callbacks=spec.callbacks):
            if False:
                i = 10
                return i + 15
            from ..functional import ones_like
            for cb in callbacks:
                grad = cb(tensor, grad)
            self._gradients[id(tensor)] = grad
        self._grad.wrt(tensor, callback=callback)

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        'Stop recording operations and release resources kept for gradient computation\n\n        After this call, you will not be able to call :meth:`backward`.\n        '
        if self._grad is not None:
            self._grad.__exit__(None, None, None)
            self._grad = None
        self._recording = False
        self._gradients = dict()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        self.release()

    def __or__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, GradManager):
            return GradManagerGroup([self, other])
        return NotImplemented
    __ror__ = __or__

class GradManagerGroup:

    def __init__(self, gms) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._gms = list(gms)

    def merge_with(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, GradManager):
            other = GradManagerGroup([other])
        elif not isinstance(other, GradManagerGroup):
            return NotImplemented
        return GradManagerGroup([*self._gms, *other._gms])
    __or__ = merge_with
    __ror__ = merge_with

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        Grad.stack.append([])
        Grad.begin_group()
        for gm in self._gms:
            gm.record()
            assert gm._grad is not None
        Grad.end_group()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        for gm in reversed(self._gms):
            gm.release()
            assert gm._grad is None