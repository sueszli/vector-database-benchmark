import chainer
from chainer import backend
import chainer.utils

class Send(chainer.Function):
    """Send elements to target process."""

    def __init__(self, comm, peer_rank, peer_tag):
        if False:
            for i in range(10):
                print('nop')
        chainer.utils.experimental('chainermn.functions.Send')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag

    @property
    def label(self):
        if False:
            i = 10
            return i + 15
        return '{} (peer_rank: {})'.format(self.__class__.__name__, self.peer_rank)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(*inputs)
        xs = inputs[:-1]
        if len(xs) == 1:
            xs = xs[0]
        self.comm.send(xs, self.peer_rank, self.peer_tag)
        return (xp.array([], dtype=xp.float32),)

    def backward(self, inputs, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(*inputs)
        dummy_grad = xp.array([], dtype=xp.float32)
        grad = self.comm.recv(self.peer_rank, self.peer_tag)
        if isinstance(grad, tuple):
            return tuple([xp.array(gy) for gy in grad] + [dummy_grad])
        else:
            return (xp.array(grad), dummy_grad)

class Recv(chainer.Function):
    """Receive elements from target process."""

    def __init__(self, comm, peer_rank, peer_tag):
        if False:
            return 10
        chainer.utils.experimental('chainermn.functions.Recv')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag

    def __call__(self, *inputs):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*inputs)
        if inputs == ():
            dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
            dummy_var.name = 'dummy_var'
            return super(Recv, self).__call__(dummy_var)
        else:
            return super(Recv, self).__call__(*inputs)

    @property
    def label(self):
        if False:
            print('Hello World!')
        return '{} (peer_rank: {})'.format(self.__class__.__name__, self.peer_rank)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        data = self.comm.recv(self.peer_rank, self.peer_tag)
        if not isinstance(data, tuple):
            data = tuple([data])
        return data

    def backward(self, inputs, grad_outputs):
        if False:
            return 10
        xp = backend.get_array_module(*inputs)
        self.comm.send(grad_outputs, self.peer_rank, self.peer_tag)
        if inputs == ():
            dummy_var = tuple([xp.array([], dtype=xp.float32)])
        else:
            dummy_var = tuple([xp.zeros_like(x) for x in inputs])
        return dummy_var

def send(x, communicator, rank, tag=0):
    if False:
        print('Hello World!')
    'Send elements to target process.\n\n    This function returns a dummy variable only holding the computational\n    graph. If ``backward()`` is invoked by this dummy variable, it will\n    try to receive gradients from the target process and send them back\n    to the parent nodes.\n\n    Args:\n        x (~chainer.Variable): Variable holding a matrix which you would like\n            to send.\n        communicator (chainer.communicators.CommunicatorBase):\n            ChainerMN communicator.\n        rank (int): Target process specifier.\n        tag (int): Optional message ID (MPI feature).\n\n    Returns:\n        ~chainer.Variable:\n            A dummy variable with no actual data, only holding the\n            computational graph. Please refer\n            ``chainermn.functions.pseudo_connect`` for detail.\n\n    '
    chainer.utils.experimental('chainermn.functions.send')
    if rank == communicator.rank:
        raise ValueError('rank must be different from communicator rank, otherwise deadlock occurs')
    xp = backend.get_array_module(*x)
    dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
    if isinstance(x, list) or isinstance(x, tuple):
        inputs = x + type(x)([dummy_var])
        delegate_variable = Send(communicator, peer_rank=rank, peer_tag=tag)(*inputs)
    else:
        delegate_variable = Send(communicator, peer_rank=rank, peer_tag=tag)(x, dummy_var)
    delegate_variable.name = 'delegate_variable'
    return delegate_variable

def recv(communicator, rank, delegate_variable=None, tag=0, force_tuple=False):
    if False:
        print('Hello World!')
    'Receive elements from target process.\n\n    This function returns data received from target process. If ``backward()``\n    is invoked, it will try to send gradients to the target process.\n    The received array will be on the current CUDA device if the corresponding\n    ``send()`` is invoked with arrays on GPU.\n    Please be aware that the current CUDA device is intended one.\n    (``https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device``)\n\n    .. note::\n        If you define non-connected computational graph on one process,\n        you have to use ``delegate_variable`` to specify the output of\n        previous computational graph component.\n        Otherwise ``backward()`` does not work well.\n        Please refer ``chainermn.functions.pseudo_connect`` for detail.\n\n    Args:\n        communicator (chainer.communicators.CommunicatorBase):\n            ChainerMN communicator.\n        rank (int): Target process specifier.\n        delegate_variable (chainer.Variable):\n            Pointer to the other non-connected component.\n        tag (int): Optional message ID (MPI feature).\n        force_tuple (bool): If ``False`` (the default) a Variable will be\n            returned when the number of outputs is one. Otherwise, this\n            method returns a tuple even when the number of outputs is one.\n\n    Returns:\n        ~chainer.Variable:\n            Data received from target process. If ``backward()`` is invoked\n            by this variable, it will send gradients to the target process.\n\n    '
    chainer.utils.experimental('chainermn.functions.recv')
    if rank == communicator.rank:
        raise ValueError('rank must be different from communicator rank, otherwise deadlock occurs')
    if delegate_variable is None:
        res = Recv(communicator, peer_rank=rank, peer_tag=tag)()
    else:
        delegate_variable.name = 'delegate_variable'
        res = Recv(communicator, peer_rank=rank, peer_tag=tag)(delegate_variable)
    if force_tuple and (not isinstance(res, tuple)):
        return tuple([res])
    else:
        return res