import chainer
import chainer.testing
import chainer.testing.attr
import numpy
import pytest
import chainermn
import chainermn.functions

class Param(object):

    def __init__(self, param):
        if False:
            print('Hello World!')
        self.dtype = None
        self.__dict__.update(param)
params = [Param(p) for p in [{'dtype': numpy.float16}, {'dtype': numpy.float32}, {'dtype': chainer.mixed16}]]

def get_communicator(gpu):
    if False:
        i = 10
        return i + 15
    numpy.random.seed(42)
    if gpu:
        communicator = chainermn.create_communicator('flat')
        device = communicator.intra_rank
        chainer.cuda.get_device_from_id(device).use()
    else:
        communicator = chainermn.create_communicator('naive')
    if communicator.size < 2:
        pytest.skip('This test is for multinode')
    return communicator

def check_all_gather(xs, communicator):
    if False:
        print('Hello World!')
    x = xs[communicator.rank]
    ys = chainermn.functions.allgather(communicator, x)
    e = 0
    for (i, y) in enumerate(ys):
        e += chainer.functions.mean_squared_error(y, xs[i])
    e.backward()
    assert 0 == e.data
    assert None is not x.grad

@pytest.mark.parametrize('param', params)
def test_all_gather_cpu(param):
    if False:
        for i in range(10):
            print('nop')
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(10, i + 1)).astype(param.dtype)) for i in range(communicator.size)]
        check_all_gather(xs, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_all_gather_gpu(param):
    if False:
        while True:
            i = 10
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(10, i + 1)).astype(param.dtype)) for i in range(communicator.size)]
        for x in xs:
            x.to_gpu()
        check_all_gather(xs, communicator)

def check_all_to_all(xs, communicator):
    if False:
        while True:
            i = 10
    ys = chainermn.functions.alltoall(communicator, xs)
    y = chainer.functions.sum(ys[0])
    for _y in ys[1:]:
        y += chainer.functions.sum(_y)
    y.backward()
    assert None is not xs[0].grad

@pytest.mark.parametrize('param', params)
def test_all_to_all_cpu(param):
    if False:
        print('Hello World!')
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        data = [chainer.Variable(numpy.zeros((communicator.rank, i), dtype=param.dtype)) for i in range(communicator.size)]
        check_all_to_all(data, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_all_to_all_gpu(param):
    if False:
        return 10
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        data = [chainer.Variable(numpy.zeros((communicator.rank + 1, i + 1), dtype=param.dtype)) for i in range(communicator.size)]
        for x in data:
            x.to_gpu()
        check_all_to_all(data, communicator)

def check_bcast(x, communicator):
    if False:
        print('Hello World!')
    root = 0
    if communicator.rank == root:
        y = chainermn.functions.bcast(communicator, x, root)
    else:
        y = chainermn.functions.bcast(communicator, None, root)
    e = chainer.functions.mean_squared_error(y, x)
    e.backward()
    if communicator.rank == root:
        assert 0 == e.data
        assert None is not x.grad

@pytest.mark.parametrize('param', params)
def test_bcast_cpu(param):
    if False:
        while True:
            i = 10
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        x = chainer.Variable(numpy.random.normal(size=(100, 100)).astype(param.dtype))
        check_bcast(x, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_bcast_gpu(param):
    if False:
        print('Hello World!')
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        x = chainer.Variable(numpy.random.normal(size=(100, 100)).astype(param.dtype))
        x.to_gpu()
        check_bcast(x, communicator)

def check_gather(xs, communicator):
    if False:
        for i in range(10):
            print('nop')
    root = 0
    x = xs[communicator.rank]
    if communicator.rank == root:
        ys = chainermn.functions.gather(communicator, x, root)
        e = 0
        for (i, y) in enumerate(ys):
            e += chainer.functions.mean_squared_error(y, xs[i])
        e.backward()
        assert 0 == e.data
    else:
        phi = chainermn.functions.gather(communicator, x, root)
        phi.backward()
    assert None is not x.grad

@pytest.mark.parametrize('param', params)
def test_gather_cpu(param):
    if False:
        for i in range(10):
            print('nop')
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(100, 100)).astype(param.dtype)) for _ in range(communicator.size)]
        check_gather(xs, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_gather_gpu(param):
    if False:
        while True:
            i = 10
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(100, 100)).astype(param.dtype)) for _ in range(communicator.size)]
        for x in xs:
            x.to_gpu()
        check_gather(xs, communicator)

@pytest.mark.parametrize('param', params)
def test_gatherv_cpu(param):
    if False:
        print('Hello World!')
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(i + 1, i + 1)).astype(param.dtype)) for i in range(communicator.size)]
        check_gather(xs, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_gatherv_gpu(param):
    if False:
        for i in range(10):
            print('nop')
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(i + 1, i + 1)).astype(param.dtype)) for i in range(communicator.size)]
        for x in xs:
            x.to_gpu()
        check_gather(xs, communicator)

def check_scatter(xs, communicator):
    if False:
        for i in range(10):
            print('nop')
    root = 0
    y = chainermn.functions.scatter(communicator, xs if communicator.rank == root else None, root)
    x = xs[communicator.rank]
    e = chainer.functions.mean_squared_error(y, x)
    e.backward()
    assert 0 == e.data
    assert None is not x.grad

@pytest.mark.parametrize('param', params)
def test_scatter_cpu(param):
    if False:
        return 10
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(100, 100)).astype(param.dtype)) for _ in range(communicator.size)]
        check_scatter(xs, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_scatter_gpu(param):
    if False:
        i = 10
        return i + 15
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(100, 100)).astype(param.dtype)) for _ in range(communicator.size)]
        for x in xs:
            x.to_gpu()
        check_scatter(xs, communicator)

@pytest.mark.parametrize('param', params)
def test_scatterv_cpu(param):
    if False:
        print('Hello World!')
    communicator = get_communicator(False)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(i + 1, i + 1)).astype(param.dtype)) for i in range(communicator.size)]
        check_scatter(xs, communicator)

@pytest.mark.parametrize('param', params)
@chainer.testing.attr.gpu
def test_scatterv_gpu(param):
    if False:
        for i in range(10):
            print('nop')
    communicator = get_communicator(True)
    with chainer.using_config('dtype', param.dtype):
        xs = [chainer.Variable(numpy.random.normal(size=(i + 1, i + 1)).astype(param.dtype)) for i in range(communicator.size)]
        for x in xs:
            x.to_gpu()
        check_scatter(xs, communicator)