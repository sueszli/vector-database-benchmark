import functools
import chainer
import copy
from chainer.backends.cuda import cupy
import chainer.testing
import chainer.testing.attr
import numpy
import pytest
import chainermn
from chainer.functions import sigmoid
from chainer.functions import mean_squared_error as mse

class Param(object):

    def __init__(self, param):
        if False:
            return 10
        self.dtype = None
        self.__dict__.update(param)
params = [Param(p) for p in [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': chainer.mixed16}]]
function = sigmoid
evaluation = mse

def create_communicator(gpu, param):
    if False:
        while True:
            i = 10
    if gpu:
        communicator = chainermn.create_communicator('flat')
        device = communicator.intra_rank
        chainer.cuda.get_device_from_id(device).use()
    else:
        communicator = chainermn.create_communicator('naive')
    if communicator.size < 2:
        pytest.skip('This test is for multinode')
    return communicator

def create_x(gpu, param, communicator):
    if False:
        while True:
            i = 10
    x = chainer.Variable(numpy.arange(10).reshape(1, 10).astype(param.dtype) / 10)
    if gpu:
        x.to_gpu()
    return x

def create_models(gpu, param, communicator):
    if False:
        print('Hello World!')
    model = chainer.links.Linear(10, 10, initialW=_init_w(communicator.rank))
    entire_model = [chainer.links.Linear(10, 10, initialW=_init_w(l)) for l in range(communicator.size)]
    if gpu:
        device = cupy.cuda.Device()
        model.to_device(device)
        for model_ in entire_model:
            model_.to_device(device)
    return (model, entire_model)

def _init_w(l):
    if False:
        i = 10
        return i + 15
    return 1.0 * numpy.arange(100).reshape(10, 10) / ((l + 1) * 100)

def check_communication(gpu, param):
    if False:
        while True:
            i = 10
    with chainer.using_config('dtype', param.dtype):
        communicator = create_communicator(gpu, param)
        rank_send = (communicator.rank + 1) % communicator.size
        rank_recv = (communicator.rank - 1) % communicator.size
        x = create_x(gpu, param, communicator)
        (model, entire_model) = create_models(gpu, param, communicator)
        if communicator.rank == 0:
            y = function(model(x))
            err = chainermn.functions.send(y, communicator, rank_send)
            err.backward()
            grad = model.W.grad
            x_ = x
            for l in range(communicator.size):
                x_ = function(entire_model[l](x_))
            err_ = evaluation(x_, x)
            err_.backward()
            grad_expected = entire_model[0].W.grad
            chainer.testing.assert_allclose(grad, grad_expected)
        elif communicator.rank == communicator.size - 1:
            x_ = chainermn.functions.recv(communicator, rank_recv)
            y = function(model(x_))
            err = evaluation(y, x)
            err.backward()
            x_ = x
            for l in range(communicator.size):
                x_ = function(entire_model[l](x_))
            y_expect = x_
            chainer.testing.assert_allclose(y.data, y_expect.data)
        else:
            x_ = chainermn.functions.recv(communicator, rank_recv)
            y = function(model(x_))
            err = chainermn.functions.send(y, communicator, rank_send)
            err.backward()

@pytest.mark.parametrize('param', params)
def test_communication_cpu(param):
    if False:
        for i in range(10):
            print('nop')
    check_communication(False, param)

@chainer.testing.attr.gpu
@pytest.mark.parametrize('param', params)
def test_communication_gpu(param):
    if False:
        print('Hello World!')
    check_communication(True, param)

def check_retain(gpu, param):
    if False:
        print('Hello World!')
    with chainer.using_config('dtype', param.dtype):
        communicator = create_communicator(gpu, param)
        rank_send = (communicator.rank + 1) % communicator.size
        rank_recv = (communicator.rank - 1) % communicator.size
        x = create_x(gpu, param, communicator)
        (model, entire_model) = create_models(gpu, param, communicator)
        if communicator.rank == 0:
            t = copy.copy(x)
            y = function(model(x))
            dlg = chainermn.functions.send(y, communicator, rank_send)
            x_ = chainermn.functions.recv(communicator, rank_recv, delegate_variable=dlg)
            err = evaluation(x_, t)
            err.backward()
            assert x.grad is not None
        else:
            x_ = chainermn.functions.recv(communicator, rank_recv)
            y = function(model(x_))
            err = chainermn.functions.send(y, communicator, rank_send)
            err.backward()

@pytest.mark.parametrize('param', params)
def test_retain_cpu(param):
    if False:
        i = 10
        return i + 15
    check_retain(False, param)

@chainer.testing.attr.gpu
@pytest.mark.parametrize('param', params)
def test_retain_gpu(param):
    if False:
        return 10
    check_retain(True, param)

def check_tuple_communication(length, gpu, param):
    if False:
        while True:
            i = 10
    with chainer.using_config('dtype', param.dtype):
        communicator = create_communicator(gpu, param)
        rank_send = (communicator.rank + 1) % communicator.size
        rank_recv = (communicator.rank - 1) % communicator.size
        x = create_x(gpu, param, communicator)
        (model, entire_model) = create_models(gpu, param, communicator)
        if communicator.rank == 0:
            y = []
            for i in range(length):
                _y = function(model(x))
                y.append(_y)
            err = chainermn.functions.send(y, communicator, rank_send)
            err.backward()
        elif communicator.rank == communicator.size - 1:
            y = chainermn.functions.recv(communicator, rank_recv, force_tuple=True)
            assert isinstance(y, tuple)
            z = functools.reduce(lambda x, y: x + y, y)
            err = evaluation(z, x)
            err.backward()
        else:
            y = chainermn.functions.recv(communicator, rank_recv)
            err = chainermn.functions.send(y, communicator, rank_send)
            err.backward()
lengths = [1, 2]

@pytest.mark.parametrize('length', lengths)
@pytest.mark.parametrize('param', params)
def test_tuple_communication_cpu(length, param):
    if False:
        print('Hello World!')
    check_tuple_communication(length, False, param)

@chainer.testing.attr.gpu
@pytest.mark.parametrize('length', lengths)
@pytest.mark.parametrize('param', params)
def test_tuple_communication_gpu(length, param):
    if False:
        while True:
            i = 10
    check_tuple_communication(length, True, param)

@pytest.mark.parametrize('param', params)
def test_non_variable_send(param):
    if False:
        for i in range(10):
            print('nop')
    'Checks if backward will be called even if inputs are not Variable.\n\n    This test confirms whether deadlock occurs when numpy/cupy array is\n    given as an input of send.\n    In this case, the input will be converted to chainer Variable without\n    ``requires_grad``, thus ``backward`` will not be called without any\n    modification.\n    '
    communicator = chainermn.create_communicator('naive')
    if communicator.size < 2:
        pytest.skip('This test is for multinode')
    rank_send = (communicator.rank + 1) % communicator.size
    rank_recv = (communicator.rank - 1) % communicator.size
    if communicator.rank == 0:
        x = numpy.ones((1, 10)).astype(param.dtype)
        phi = chainermn.functions.send(x, communicator, rank=rank_send)
        (x,) = chainermn.functions.pseudo_connect(phi, x)
        y = chainer.functions.sum(x)
        t = numpy.array(0).astype(param.dtype)
        z = chainer.functions.mean_squared_error(y, t)
        z.backward()
    elif communicator.rank == communicator.size - 1:
        x = chainermn.functions.recv(communicator, rank=rank_recv)
        y = chainer.functions.sum(x)
        t = numpy.array(0).astype(param.dtype)
        z = chainer.functions.mean_squared_error(y, t)
        z.backward()
    else:
        x = chainermn.functions.recv(communicator, rank=rank_recv)
        phi = chainermn.functions.send(x, communicator, rank=rank_send)
        phi.backward()