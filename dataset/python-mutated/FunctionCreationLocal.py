import itertools

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')

    def empty():
        if False:
            return 10
        pass
    empty = 1
    return empty
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')