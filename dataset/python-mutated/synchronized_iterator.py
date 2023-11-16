import chainer
import numpy

class _SynchronizedIterator(chainer.dataset.iterator.Iterator):

    def __init__(self, actual_iterator, communicator):
        if False:
            while True:
                i = 10
        if not hasattr(actual_iterator, 'order_sampler'):
            raise ValueError('actual_iterator must have order_sampler')
        else:
            super(_SynchronizedIterator, self).__setattr__('actual_iterator', actual_iterator)
        self.communicator = communicator
        if self.communicator.rank == 0:
            seed = numpy.random.randint(0, 2 ** 32 - 1)
        else:
            seed = None
        seed = self.communicator.bcast_obj(seed, root=0)
        rng = numpy.random.RandomState(seed)
        self.actual_iterator.order_sampler = chainer.iterators.ShuffleOrderSampler(rng)
        self.actual_iterator.reset()

    def __getattr__(self, attr_name):
        if False:
            while True:
                i = 10
        return getattr(self.actual_iterator, attr_name)

    def __setattr__(self, attr_name, value):
        if False:
            for i in range(10):
                print('nop')
        setattr(self.actual_iterator, attr_name, value)

    def __next__(self):
        if False:
            i = 10
            return i + 15
        return self.actual_iterator.__next__()

    def serialize(self, serializer):
        if False:
            print('Hello World!')
        self.actual_iterator.serialize(serializer)

def create_synchronized_iterator(actual_iterator, communicator):
    if False:
        print('Hello World!')
    'Create a synchronized iterator from a Chainer iterator.\n\n    This iterator shares the same batches on multiple processes,\n    using the same random number generators to maintain the order of batch\n    shuffling same.\n\n    Here is an example situation.\n    When we train a sequence-to-sequence model, where the encoder and\n    the decoder is located on two different processes, we want to share\n    the same batches on each process, thus inputs for the encoder and\n    output teacher signals for the decoder become consistent.\n\n    In order to use the synchronized iterator, first create the iterator\n    from Chainer iterator and ChainerMN communicator::\n\n        iterator = chainermn.iterators.create_synchronized_iterator(\n            chainer.iterators.SerialIterator(\n                dataset, batch_size, shuffle=True),\n            communicator)\n\n    Then you can use it as the ordinary Chainer iterator::\n\n        updater = chainer.training.StandardUpdater(iterator, optimizer)\n        trainer = training.Trainer(updater)\n        trainer.run()\n\n    The resulting iterator shares the same shuffling order among processes\n    in the specified communicator.\n\n    Args:\n        actual_iterator: Chainer iterator\n            (e.g., ``chainer.iterators.SerialIterator``).\n        communicator: ChainerMN communicator.\n\n    Returns:\n        The synchronized iterator based on ``actual_iterator``.\n    '
    chainer.utils.experimental('chainermn.iterators.create_synchronized_iterator')
    return _SynchronizedIterator(actual_iterator, communicator)