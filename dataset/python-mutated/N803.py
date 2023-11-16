def func(_, a, badAllowed):
    if False:
        for i in range(10):
            print('nop')
    return (_, a, badAllowed)

def func(_, a, stillBad):
    if False:
        for i in range(10):
            print('nop')
    return (_, a, stillBad)

class Class:

    def method(self, _, a, badAllowed):
        if False:
            print('Hello World!')
        return (_, a, badAllowed)

    def method(self, _, a, stillBad):
        if False:
            print('Hello World!')
        return (_, a, stillBad)