import chainer
import chainer.testing
import chainer.testing.attr
import chainermn
from chainermn.iterators.multi_node_iterator import _build_ctrl_msg
from chainermn.iterators.multi_node_iterator import _parse_ctrl_msg
import numpy as np
import platform
import pytest
from six.moves import range
import unittest

class DummySerializer(chainer.serializer.Serializer):

    def __init__(self, target):
        if False:
            while True:
                i = 10
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __call__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.target[key] = value
        return self.target[key]

class DummyDeserializer(chainer.serializer.Deserializer):

    def __init__(self, target):
        if False:
            for i in range(10):
                print('nop')
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __call__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            value = self.target[key]
        elif isinstance(value, np.ndarray):
            np.copyto(value, self.target[key])
        else:
            value = type(value)(np.asarray(self.target[key]))
        return value

@chainer.testing.parameterize(*chainer.testing.product({'paired_dataset': [True, False], 'iterator_class': [chainer.iterators.SerialIterator, chainer.iterators.MultiprocessIterator]}))
class TestMultiNodeIterator(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        if self.iterator_class == chainer.iterators.MultiprocessIterator and int(platform.python_version_tuple()[0]) < 3:
            pytest.skip('This test requires Python version >= 3')
        self.communicator = chainermn.create_communicator('naive')
        if self.communicator.size < 2:
            pytest.skip('This test is for multinode only')
        self.N = 100
        if self.paired_dataset:
            self.dataset = list(zip(np.arange(self.N).astype(np.float32), np.arange(self.N).astype(np.float32)))
        else:
            self.dataset = np.arange(self.N).astype(np.float32)

    def test_mn_iterator(self):
        if False:
            while True:
                i = 10
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(self.iterator_class(self.dataset, batch_size=bs, shuffle=True), self.communicator)
        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == 0:
                    for rank_from in range(1, self.communicator.size):
                        _batch = self.communicator.mpi_comm.recv(source=rank_from)
                        self.assertEqual(batch, _batch)
                else:
                    self.communicator.mpi_comm.ssend(batch, dest=0)

    def test_mn_iterator_frag(self):
        if False:
            while True:
                i = 10
        bs = 7
        iterator = chainermn.iterators.create_multi_node_iterator(self.iterator_class(self.dataset, batch_size=bs, shuffle=True), self.communicator)
        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == 0:
                    for rank_from in range(1, self.communicator.size):
                        _batch = self.communicator.mpi_comm.recv(source=rank_from)
                        self.assertEqual(batch, _batch)
                else:
                    self.communicator.mpi_comm.ssend(batch, dest=0)

    def test_mn_iterator_change_master(self):
        if False:
            i = 10
            return i + 15
        rank_master = 1
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(self.iterator_class(self.dataset, batch_size=bs, shuffle=True), self.communicator, rank_master)
        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == rank_master:
                    rank_slaves = [i for i in range(self.communicator.size) if i != rank_master]
                    for rank_from in rank_slaves:
                        _batch = self.communicator.mpi_comm.recv(source=rank_from)
                        self.assertEqual(batch, _batch)
                else:
                    self.communicator.mpi_comm.ssend(batch, dest=rank_master)

    def test_mn_iterator_no_repeat(self):
        if False:
            print('Hello World!')
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(self.iterator_class(self.dataset, batch_size=bs, shuffle=True, repeat=False), self.communicator)
        for e in range(3):
            try:
                while True:
                    batch = iterator.next()
                    if self.communicator.rank == 0:
                        for rank_from in range(1, self.communicator.size):
                            _batch = self.communicator.mpi_comm.recv(source=rank_from)
                            self.assertEqual(batch, _batch)
                    else:
                        self.communicator.mpi_comm.ssend(batch, dest=0)
            except StopIteration:
                continue

    def test_overwrite_order(self):
        if False:
            print('Hello World!')
        "Tests behavior on serialization.\n\n        This test confirms that master's batch order can be overwritten,\n        while slave's batch order cannot be overwritten, since slave must\n        always distribute the completely same batch as master.\n        "
        bs = 4
        rank_master = 0
        iterator = chainermn.iterators.create_multi_node_iterator(self.iterator_class(self.dataset, batch_size=bs, shuffle=True, repeat=False), self.communicator, rank_master=rank_master)
        target = dict()
        iterator.serialize(DummySerializer(target))
        order = target['order']
        new_order = np.roll(order, 1)
        target['order'] = new_order
        iterator.serialize(DummyDeserializer(target))
        if self.communicator.rank == rank_master:
            self.assertEqual(iterator._state.order.tolist(), new_order.tolist())
        else:
            self.assertEqual(iterator._order.tolist(), order.tolist())

class TestMultiNodeIteratorDataType(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.communicator = chainermn.create_communicator('naive')
        if self.communicator.size < 2:
            pytest.skip('This test is for multinode only')

    def test_invalid_type(self):
        if False:
            return 10
        self.N = 10
        self.dataset = ['test'] * self.N
        bs = 1
        iterator = chainermn.iterators.create_multi_node_iterator(chainer.iterators.SerialIterator(self.dataset, batch_size=bs, shuffle=True), self.communicator)
        with self.assertRaises(TypeError):
            iterator.next()

class TestCntrlMessageConversion(unittest.TestCase):

    def test_conversion(self):
        if False:
            while True:
                i = 10
        stop = True
        is_valid_data_type = True
        is_paired_dataset = True
        is_new_epoch = True
        current_position = 0
        msg = _build_ctrl_msg(stop, is_valid_data_type, is_paired_dataset, is_new_epoch, current_position)
        np.testing.assert_array_equal(msg, _build_ctrl_msg(*_parse_ctrl_msg(msg)))