import sys
import random
import math

class RDDSamplerBase:

    def __init__(self, withReplacement, seed=None):
        if False:
            i = 10
            return i + 15
        self._seed = seed if seed is not None else random.randint(0, sys.maxsize)
        self._withReplacement = withReplacement
        self._random = None

    def initRandomGenerator(self, split):
        if False:
            print('Hello World!')
        self._random = random.Random(self._seed ^ split)
        for _ in range(10):
            self._random.randint(0, 1)

    def getUniformSample(self):
        if False:
            print('Hello World!')
        return self._random.random()

    def getPoissonSample(self, mean):
        if False:
            i = 10
            return i + 15
        if mean < 20.0:
            lda = math.exp(-mean)
            p = self._random.random()
            k = 0
            while p > lda:
                k += 1
                p *= self._random.random()
        else:
            p = self._random.expovariate(mean)
            k = 0
            while p < 1.0:
                k += 1
                p += self._random.expovariate(mean)
        return k

    def func(self, split, iterator):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class RDDSampler(RDDSamplerBase):

    def __init__(self, withReplacement, fraction, seed=None):
        if False:
            i = 10
            return i + 15
        RDDSamplerBase.__init__(self, withReplacement, seed)
        self._fraction = fraction

    def func(self, split, iterator):
        if False:
            while True:
                i = 10
        self.initRandomGenerator(split)
        if self._withReplacement:
            for obj in iterator:
                count = self.getPoissonSample(self._fraction)
                for _ in range(0, count):
                    yield obj
        else:
            for obj in iterator:
                if self.getUniformSample() < self._fraction:
                    yield obj

class RDDRangeSampler(RDDSamplerBase):

    def __init__(self, lowerBound, upperBound, seed=None):
        if False:
            print('Hello World!')
        RDDSamplerBase.__init__(self, False, seed)
        self._lowerBound = lowerBound
        self._upperBound = upperBound

    def func(self, split, iterator):
        if False:
            print('Hello World!')
        self.initRandomGenerator(split)
        for obj in iterator:
            if self._lowerBound <= self.getUniformSample() < self._upperBound:
                yield obj

class RDDStratifiedSampler(RDDSamplerBase):

    def __init__(self, withReplacement, fractions, seed=None):
        if False:
            while True:
                i = 10
        RDDSamplerBase.__init__(self, withReplacement, seed)
        self._fractions = fractions

    def func(self, split, iterator):
        if False:
            print('Hello World!')
        self.initRandomGenerator(split)
        if self._withReplacement:
            for (key, val) in iterator:
                count = self.getPoissonSample(self._fractions[key])
                for _ in range(0, count):
                    yield (key, val)
        else:
            for (key, val) in iterator:
                if self.getUniformSample() < self._fractions[key]:
                    yield (key, val)