class Equality:

    def __init__(self, lhs, rhs):
        if False:
            print('Hello World!')
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.lhs} = {self.rhs}'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.lhs} = {self.rhs}'

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, Equality):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False