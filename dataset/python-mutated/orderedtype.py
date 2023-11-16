from functools import total_ordering

@total_ordering
class OrderedType:
    creation_counter = 1

    def __init__(self, _creation_counter=None):
        if False:
            return 10
        self.creation_counter = _creation_counter or self.gen_counter()

    @staticmethod
    def gen_counter():
        if False:
            print('Hello World!')
        counter = OrderedType.creation_counter
        OrderedType.creation_counter += 1
        return counter

    def reset_counter(self):
        if False:
            i = 10
            return i + 15
        self.creation_counter = self.gen_counter()

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(self, type(other)):
            return self.creation_counter == other.creation_counter
        return NotImplemented

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, OrderedType):
            return self.creation_counter < other.creation_counter
        return NotImplemented

    def __gt__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, OrderedType):
            return self.creation_counter > other.creation_counter
        return NotImplemented

    def __hash__(self):
        if False:
            return 10
        return hash(self.creation_counter)