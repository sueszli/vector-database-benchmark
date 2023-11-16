import chainer
import chainer.testing
import chainer.testing.attr
import unittest
import chainermn
from chainermn.testing.device import get_device

class ExampleModel(chainer.Chain):

    def __init__(self, n_in=3, n_units=5, n_out=2):
        if False:
            i = 10
            return i + 15
        super(ExampleModel, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(n_in, n_units, nobias=True)
            self.bn1 = chainer.links.BatchNormalization(n_units)
            self.l2 = chainer.links.Linear(n_units, n_units, nobias=True)
            self.bn2 = chainer.links.BatchNormalization(n_units)
            self.l3 = chainer.links.Linear(n_units, n_out)

class TestAllreducePersistent(unittest.TestCase):

    def _test(self, comm, model, use_gpu, use_chx):
        if False:
            while True:
                i = 10
        if use_gpu:
            chainer.cuda.get_device_from_id(comm.intra_rank).use()
        device = get_device(comm.intra_rank if use_gpu else None, use_chx)
        model.to_device(device)
        rank = comm.rank
        model.bn1.avg_mean.fill(rank * 1)
        model.bn2.avg_mean.fill(rank * 2)
        model.bn1.avg_var.fill(rank * 3)
        model.bn2.avg_var.fill(rank * 4)
        allreduce_persistent = chainermn.extensions.AllreducePersistent(model, comm)
        allreduce_persistent()
        avg_rank = (comm.size - 1) / 2.0
        chainer.testing.assert_allclose(model.bn1.avg_mean, avg_rank * 1)
        chainer.testing.assert_allclose(model.bn2.avg_mean, avg_rank * 2)
        chainer.testing.assert_allclose(model.bn1.avg_var, avg_rank * 3)
        chainer.testing.assert_allclose(model.bn2.avg_var, avg_rank * 4)

    def test_allreduce_persistent_cpu(self):
        if False:
            return 10
        comm = chainermn.create_communicator('naive')
        model = ExampleModel()
        self._test(comm, model, False, False)
        self._test(comm, model, False, True)

    @chainer.testing.attr.gpu
    def test_allreduce_persistent_gpu(self):
        if False:
            while True:
                i = 10
        comm = chainermn.create_communicator('flat')
        model = ExampleModel()
        self._test(comm, model, True, False)
        self._test(comm, model, True, True)