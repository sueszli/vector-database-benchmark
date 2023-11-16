class CyclingIterator:
    """
    An iterator decorator that cycles through the
    underlying iterator "n" times. Useful to "unroll"
    the dataset across multiple training epochs.

    The generator function is called as ``generator_fn(epoch)``
    to obtain the underlying iterator, where ``epoch`` is a
    number less than or equal to ``n`` representing the ``k``th cycle

    For example if ``generator_fn`` always returns ``[1,2,3]``
    then ``CyclingIterator(n=2, generator_fn)`` will iterate through
    ``[1,2,3,1,2,3]``
    """

    def __init__(self, n: int, generator_fn, start_epoch=0):
        if False:
            return 10
        self._n = n
        self._epoch = start_epoch
        self._generator_fn = generator_fn
        self._iter = generator_fn(self._epoch)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        try:
            return next(self._iter)
        except StopIteration as eod:
            if self._epoch < self._n - 1:
                self._epoch += 1
                self._iter = self._generator_fn(self._epoch)
                return self.__next__()
            else:
                raise eod