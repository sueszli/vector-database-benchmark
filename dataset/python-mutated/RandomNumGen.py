"""RandomNumGen module: contains the RandomNumGen class"""
__all__ = ['randHash', 'RandomNumGen']
from direct.directnotify import DirectNotifyGlobal
from panda3d.core import Mersenne

def randHash(num):
    if False:
        for i in range(10):
            print('nop')
    ' this returns a random 16-bit integer, given a seed integer.\n    It will always return the same output given the same input.\n    This is useful for repeatably mapping numbers with predictable\n    bit patterns (i.e. doIds or zoneIds) to numbers with random bit patterns\n    '
    rng = RandomNumGen(num)
    return rng.randint(0, (1 << 16) - 1)

class RandomNumGen:
    notify = DirectNotifyGlobal.directNotify.newCategory('RandomNumGen')

    def __init__(self, seed):
        if False:
            i = 10
            return i + 15
        'seed must be an integer or another RandomNumGen'
        if isinstance(seed, RandomNumGen):
            rng = seed
            seed = rng.randint(0, 1 << 16)
        self.notify.debug('seed: ' + str(seed))
        seed = int(seed)
        rng = Mersenne(seed)
        self.__rng = rng

    def __rand(self, N):
        if False:
            i = 10
            return i + 15
        'returns integer in [0..N)'
        assert N >= 0
        assert N <= 2147483647
        return int(self.__rng.getUint31() * N >> 31)

    def choice(self, seq):
        if False:
            print('Hello World!')
        'returns a random element from seq'
        return seq[self.__rand(len(seq))]

    def shuffle(self, x):
        if False:
            while True:
                i = 10
        'randomly shuffles x in-place'
        for i in range(len(x) - 1, 0, -1):
            j = int(self.__rand(i + 1))
            (x[i], x[j]) = (x[j], x[i])

    def randrange(self, start, stop=None, step=1):
        if False:
            for i in range(10):
                print('nop')
        'randrange([start,] stop[, step])\n        same as choice(range(start, stop[, step])) without construction\n        of a list'
        istart = int(start)
        if istart != start:
            raise ValueError('non-integer arg 1 for randrange()')
        if stop is None:
            if istart > 0:
                return self.__rand(istart)
            raise ValueError('empty range for randrange()')
        istop = int(stop)
        if istop != stop:
            raise ValueError('non-integer stop for randrange()')
        if step == 1:
            if istart < istop:
                return istart + self.__rand(istop - istart)
            raise ValueError('empty range for randrange()')
        istep = int(step)
        if istep != step:
            raise ValueError('non-integer step for randrange()')
        if istep > 0:
            n = (istop - istart + istep - 1) / istep
        elif istep < 0:
            n = (istop - istart + istep + 1) / istep
        else:
            raise ValueError('zero step for randrange()')
        if n <= 0:
            raise ValueError('empty range for randrange()')
        return istart + istep * int(self.__rand(n))

    def randint(self, a, b):
        if False:
            return 10
        'returns integer in [a, b]'
        assert a <= b
        range = b - a + 1
        r = self.__rand(range)
        return a + r

    def random(self):
        if False:
            return 10
        'returns random float in [0.0, 1.0)'
        return float(self.__rng.getUint31()) / float(1 << 31)