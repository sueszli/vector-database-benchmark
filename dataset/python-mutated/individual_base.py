from copy import deepcopy, copy
import names
from deap.base import Fitness
from conf import partitions

class FitnessMax(Fitness):
    weights = tuple([1 for _ in range(partitions)])

class Individual(list):
    mate = lambda *x: x
    mutate = lambda x: x

    @property
    def objective(self):
        if False:
            print('Hello World!')
        return sum(self)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{list(self.fitness.values)} {self.objective} {self.name}'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.name = names.get_full_name()
        self.fitness = FitnessMax()
        super(Individual, self).__init__(*args, **kwargs)

    def __deepcopy__(self, memodict={}):
        if False:
            i = 10
            return i + 15
        obj = copy(self)
        obj.fitness = deepcopy(self.fitness)
        obj.name = names.get_full_name()
        return obj

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        (child1, child2) = self.__class__.mate(deepcopy(self), deepcopy(other))
        if False:
            print(f'\nMating: with {self.__class__.mate}')
            print(self)
            print(other)
            print('Children:')
            print(child1)
            print(child2)
        del child1.fitness.values
        del child2.fitness.values
        return (child1, child2)

    def __invert__(self):
        if False:
            for i in range(10):
                print('nop')
        mutant = self.__class__.mutate(deepcopy(self))[0]
        if False:
            print(f'\nMutating: with {self.__class__.mutate}')
            print(self)
            print(mutant)
        del mutant.fitness.values
        return mutant

    def __hash__(self):
        if False:
            return 10
        return hash(tuple(self.fitness.values))