import chainer
from chainer import backend
import numpy

class AllGather(chainer.Function):
    """Collective all-gather communication."""

    def __init__(self, comm):
        if False:
            return 10
        chainer.utils.experimental('chainermn.functions.AllGather')
        self.comm = comm

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        x_dtype = x.dtype
        if numpy.float16 == x_dtype:
            x = x.astype(numpy.float32)
        ret = self.comm.allgather(x)
        if numpy.float16 == x_dtype:
            ret = tuple([item.astype(x_dtype) for item in ret])
        return ret

    def backward(self, inputs, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(*inputs)
        grad_dtype = grad_outputs[0].dtype
        if numpy.float16 == grad_dtype:
            grad_outputs = tuple([item.astype(numpy.float32) for item in grad_outputs])
        gxs = self.comm.alltoall(grad_outputs)
        gx = xp.stack(gxs).sum(axis=0)
        if numpy.float16 == grad_dtype:
            gx = gx.astype(grad_dtype)
        return (gx,)

class AllToAll(chainer.Function):
    """Collective all-to-all communication."""

    def __init__(self, comm):
        if False:
            while True:
                i = 10
        chainer.utils.experimental('chainermn.functions.AllToAll')
        self.comm = comm

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        if len(inputs) != self.comm.size:
            raise ValueError('The length of inputs must be same as communicator size.')
        xs_dtype = inputs[0].dtype
        if numpy.float16 == xs_dtype:
            xs = tuple([x.astype(numpy.float32) for x in inputs])
        else:
            xs = tuple([x for x in inputs])
        ret = self.comm.alltoall(xs)
        if numpy.float16 == xs_dtype:
            ret = tuple([item.astype(xs_dtype) for item in ret])
        return ret

    def backward(self, inputs, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        assert self.comm.size == len(grad_outputs)
        xs_dtype = inputs[0].dtype
        if numpy.float16 == xs_dtype:
            gys = tuple([gy.astype(numpy.float32) for gy in grad_outputs])
        else:
            gys = tuple([gy for gy in grad_outputs])
        ret = self.comm.alltoall(gys)
        if numpy.float16 == xs_dtype:
            ret = tuple([item.astype(xs_dtype) for item in ret])
        return ret

class Bcast(chainer.Function):
    """Collective broadcast communication."""

    def __init__(self, comm, root):
        if False:
            for i in range(10):
                print('nop')
        chainer.utils.experimental('chainermn.functions.Bcast')
        self.comm = comm
        self.root = root

    def __call__(self, *inputs):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*inputs)
        if inputs == ():
            dummy_var = chainer.Variable(xp.array([], dtype=chainer.config.dtype))
            dummy_var.name = 'dummy_var'
            return super(Bcast, self).__call__(dummy_var)
        else:
            return super(Bcast, self).__call__(*inputs)

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        x_dtype = inputs[0].dtype
        if self.comm.rank == self.root:
            (x,) = inputs
            if numpy.float16 == x_dtype:
                x = x.astype(numpy.float32)
        else:
            x = None
        x = (self.comm.bcast(x, self.root),)
        if numpy.float16 == x_dtype:
            x = tuple([item.astype(x_dtype) for item in x])
        return x

    def backward(self, inputs, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        (gx,) = grad_outputs
        gx_dtype = gx.dtype
        if numpy.float16 == gx_dtype:
            gx = gx.astype(numpy.float32)
        gxs = self.comm.gather(gx, self.root)
        if self.comm.rank == self.root:
            xp = backend.get_array_module(*gxs)
            gxs = xp.stack(gxs)
            _sum = (gxs.sum(axis=0),)
            if numpy.float16 == gx_dtype:
                _sum = tuple([item.astype(gx_dtype) for item in _sum])
            return _sum
        else:
            return (None,)

class Gather(chainer.Function):
    """Collective gather communication."""

    def __init__(self, comm, root):
        if False:
            return 10
        chainer.utils.experimental('chainermn.functions.Gather')
        self.comm = comm
        self.root = root

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*inputs)
        (x,) = inputs
        x_dtype = x.dtype
        if numpy.float16 == x_dtype:
            x = x.astype(numpy.float32)
        ys = self.comm.gather(x, self.root)
        if self.comm.rank == self.root:
            if numpy.float16 == x_dtype:
                ys = tuple([item.astype(x_dtype) for item in ys])
            return ys
        else:
            return (xp.array([], dtype=x_dtype),)

    def backward(self, inputs, grad_outputs):
        if False:
            i = 10
            return i + 15
        input_dtype = inputs[0].dtype
        if self.comm.rank == self.root and numpy.float16 == input_dtype:
            grad_outputs = tuple([item.astype(numpy.float32) for item in grad_outputs])
        ret = (self.comm.scatter(grad_outputs, self.root),)
        if numpy.float16 == input_dtype:
            ret = tuple([item.astype(input_dtype) for item in ret])
        return ret

class Scatter(chainer.Function):
    """Collective scatter communication."""

    def __init__(self, comm, root):
        if False:
            while True:
                i = 10
        chainer.utils.experimental('chainermn.functions.Scatter')
        self.comm = comm
        self.root = root

    def __call__(self, *inputs):
        if False:
            return 10
        xp = backend.get_array_module(*inputs)
        if inputs == ():
            dummy_var = chainer.Variable(xp.array([], dtype=chainer.config.dtype))
            dummy_var.name = 'dummy_var'
            return super(Scatter, self).__call__(dummy_var)
        else:
            return super(Scatter, self).__call__(*inputs)

    def forward(self, inputs):
        if False:
            return 10
        input_dtype = inputs[0].dtype
        if self.comm.rank == self.root:
            if numpy.float16 == input_dtype:
                inputs = tuple([item.astype(numpy.float32) for item in inputs])
            y = self.comm.scatter(inputs, self.root)
        else:
            y = self.comm.scatter(None, self.root)
        if numpy.float16 == input_dtype:
            y = y.astype(input_dtype)
        return (y,)

    def backward(self, inputs, grad_outputs):
        if False:
            return 10
        xp = backend.get_array_module(*inputs)
        (gy,) = grad_outputs
        gy_dtype = gy.dtype
        if numpy.float16 == gy_dtype:
            gy = gy.astype(numpy.float32)
        gxs = self.comm.gather(gy, self.root)
        if self.comm.rank == self.root:
            if numpy.float16 == gy_dtype:
                gxs = tuple([item.astype(gy_dtype) for item in gxs])
            return gxs
        else:
            if inputs == ():
                dummy_var = tuple([xp.array([], dtype=xp.float32)])
            else:
                dummy_var = tuple([xp.zeros_like(x) for x in inputs])
            return dummy_var

def allgather(comm, x):
    if False:
        return 10
    'Differentiable all-gather communication between workers.\n\n    This function invokes gather communications among processes specified\n    by the communicator. Backward will be invoked as well as the ordinary\n    chainer functions, where gradients are reduced to each process.\n\n    The received array will be on the current CUDA device on the invoking\n    process if ``x`` is on GPU. Please be aware that the current CUDA device\n    is intended one.\n    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n\n    Args:\n        comm: ChainerMN communicator.\n        x (chainer.Variables): Variables to send.\n\n    Returns:\n        ys (list of chainer.Variables): Received variables.\n    '
    chainer.utils.experimental('chainermn.functions.all_gather')
    return AllGather(comm)(x)

def alltoall(comm, xs):
    if False:
        return 10
    'Differentiable all-to-all communication between workers.\n\n    This function invokes all-to-all communications among processes specified\n    by the communicator. Backward will be invoked as well as the ordinary\n    chainer functions, just passing input gradients back.\n    Unlike point-to-point communication such as ``chainermn.functions.send``\n    and ``chainermn.functions.recv``, users need not to care about\n    delegate variables, since ``backward()`` will not be invoked until\n    all gradients from output direction arrive.\n    Please refer to ``chainermn.functions.pseudo_connect`` about the detail\n    of delegate variables.\n\n    The received array will be on the current CUDA device on the invoking\n    process if ``xs`` is on GPU. Please be aware that the current CUDA device\n    is intended one.\n    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n\n    Args:\n        comm: ChainerMN communicator.\n        xs (list of chainer.Variables): Variables to send.\n\n    Returns:\n        ys (list of chainer.Variables): Received variables.\n    '
    chainer.utils.experimental('chainermn.functions.all_to_all')
    if len(xs) != comm.size:
        raise ValueError('The length of xs must be same as communicator size.')
    return AllToAll(comm)(*xs)

def bcast(comm, x, root=0):
    if False:
        return 10
    'Differentiable broadcast communication between workers.\n\n    This function invokes broadcast communications among processes specified\n    by the communicator. Backward will be invoked as well as the ordinary\n    chainer functions, where gradients are gathered to the root process\n    and summed up.\n\n    The received array will be on the current CUDA device if ``x`` on the\n    invoking process is on GPU. Please be aware that the current CUDA device\n    is intended one.\n    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n\n    Args:\n        comm: ChainerMN communicator.\n        x (chainer.Variable): Variable to be sent.\n\n    Returns:\n        y (chainer.Variable): Broadcasted variable.\n    '
    chainer.utils.experimental('chainermn.functions.bcast')
    if comm.rank == root:
        return Bcast(comm, root)(x)
    else:
        return Bcast(comm, root)()

def gather(comm, x, root=0):
    if False:
        return 10
    'Differentiable gather communication between workers.\n\n    This function invokes gather communications among processes specified\n    by the communicator. Backward will be invoked as well as the ordinary\n    chainer functions, where gradients are scattered from the root process\n    to each slave.\n\n    The received array will be on the current CUDA device if ``x`` on the\n    root process is on GPU. Please be aware that the current CUDA device\n    is intended one.\n    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n\n    Args:\n        comm: ChainerMN communicator.\n        x (chainer.Variable): Variable to be sent.\n\n    Returns:\n        ys (chainer.Variable):\n            Gathered variables. ``None`` for slaves.\n    '
    chainer.utils.experimental('chainermn.functions.gather')
    return Gather(comm, root)(x)

def scatter(comm, xs, root=0):
    if False:
        i = 10
        return i + 15
    'Differentiable scatter communication between workers.\n\n    This function invokes scatter communications among processes specified\n    by the communicator. Backward will be invoked as well as the ordinary\n    chainer functions, where gradients are gathered to the root process.\n\n    The received array will be on the current CUDA device if ``xs`` on the\n    root process is on GPU. Please be aware that the current CUDA device\n    is intended one.\n    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n\n    Args:\n        comm: ChainerMN communicator.\n        xs (list of chainer.Variable):\n            Variables to be scattered for master process.\n            ``None`` for slave process.\n\n    Returns:\n        y (chainer.Variable): Scattered variable.\n    '
    chainer.utils.experimental('chainermn.functions.scatter')
    if comm.rank == root:
        return Scatter(comm, root)(*xs)
    else:
        return Scatter(comm, root)()