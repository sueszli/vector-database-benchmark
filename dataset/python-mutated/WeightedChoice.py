import random

class WeightedChoice:

    def __init__(self, listOfLists, weightIndex=0):
        if False:
            print('Hello World!')
        t = 0
        for i in listOfLists:
            t += i[weightIndex]
        self.total = t
        self.listOfLists = listOfLists
        self.weightIndex = weightIndex

    def choose(self, rng=random):
        if False:
            for i in range(10):
                print('nop')
        roll = rng.randrange(self.total)
        weight = self.weightIndex
        for i in self.listOfLists:
            roll -= i[weight]
            if roll <= 0:
                return i