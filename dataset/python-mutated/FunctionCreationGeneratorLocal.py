import itertools
module_var = None

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')

    def empty():
        if False:
            i = 10
            return i + 15
        yield 1
    empty = 1
    return empty
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')