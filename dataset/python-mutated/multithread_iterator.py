from __future__ import division
from multiprocessing import pool
import numpy
from chainer.dataset import iterator
from chainer.iterators import _statemachine
from chainer.iterators.order_samplers import ShuffleOrderSampler

class MultithreadIterator(iterator.Iterator):
    """Dataset iterator that loads examples in parallel.

    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker threads. It uses the standard :mod:`threading`
    module to parallelize the loading.

    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        n_threads (int): Number of worker threads.
        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguments: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.

    """

    def __init__(self, dataset, batch_size, repeat=True, shuffle=None, n_threads=1, order_sampler=None):
        if False:
            while True:
                i = 10
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        if self._shuffle is not None:
            if order_sampler is not None:
                raise ValueError('`shuffle` is not `None` and a custom `order_sampler` is set. Please set `shuffle` to `None` to use the custom order sampler.')
            elif self._shuffle:
                order_sampler = ShuffleOrderSampler()
        elif order_sampler is None:
            order_sampler = ShuffleOrderSampler()
        self.order_sampler = order_sampler
        self.n_threads = n_threads
        self._pool = None
        self.reset()

    def reset(self):
        if False:
            return 10
        if self.order_sampler is None:
            order = None
        else:
            order = self.order_sampler(numpy.arange(len(self.dataset)), 0)
        self._state = _statemachine.IteratorState(0, 0, False, order)
        self._previous_epoch_detail = -1.0
        self._next = None

    def finalize(self):
        if False:
            print('Hello World!')
        pool = self._pool
        self._next = None
        self._pool = None
        if pool is not None:
            pool.terminate()

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._next is None:
            self._invoke_prefetch()
        batch = self._get()
        self._invoke_prefetch()
        return batch
    next = __next__

    @property
    def current_position(self):
        if False:
            while True:
                i = 10
        return self._state.current_position

    @property
    def epoch(self):
        if False:
            i = 10
            return i + 15
        return self._state.epoch

    @property
    def is_new_epoch(self):
        if False:
            i = 10
            return i + 15
        return self._state.is_new_epoch

    @property
    def epoch_detail(self):
        if False:
            while True:
                i = 10
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        if False:
            while True:
                i = 10
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        if False:
            for i in range(10):
                print('nop')
        current_position = serializer('current_position', self.current_position)
        epoch = serializer('epoch', self.epoch)
        is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        order = serializer('_order', self._state.order)
        self._state = _statemachine.IteratorState(current_position, epoch, is_new_epoch, order)
        self._previous_epoch_detail = serializer('previous_epoch_detail', self._previous_epoch_detail)
        if self._previous_epoch_detail is None:
            self._previous_epoch_detail = -1.0
        self._next = None

    @staticmethod
    def _read(args):
        if False:
            while True:
                i = 10
        (dataset, index) = args
        return dataset[index]

    def _invoke_prefetch(self):
        if False:
            return 10
        assert self._next is None
        (self._next_state, indices) = _statemachine.iterator_statemachine(self._state, self.batch_size, self.repeat, self.order_sampler, len(self.dataset))
        if indices is None:
            self._next = None
        else:
            if self._pool is None:
                self._pool = pool.ThreadPool(self.n_threads)
            args = [(self.dataset, index) for index in indices]
            self._next = self._pool.map_async(MultithreadIterator._read, args)

    def _get(self):
        if False:
            for i in range(10):
                print('nop')
        self._previous_epoch_detail = self.epoch_detail
        self._state = self._next_state
        next = self._next
        if next is None:
            raise StopIteration
        self._next = None
        while not next.ready():
            next.wait(0.5)
        batch = [data for data in next.get()]
        return batch

    @property
    def _epoch_size(self):
        if False:
            return 10
        order = self._state.order
        if order is None:
            epoch_size = len(self.dataset)
        else:
            epoch_size = len(order)
        return epoch_size

    @property
    def repeat(self):
        if False:
            while True:
                i = 10
        return self._repeat