from collections import OrderedDict

class IterableOrderedDict(OrderedDict):

    def __iter__(self):
        if False:
            print('Hello World!')
        yield from self.values()