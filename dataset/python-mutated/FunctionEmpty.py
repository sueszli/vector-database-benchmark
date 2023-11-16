import itertools

def empty():
    if False:
        for i in range(10):
            print('nop')
    pass
module_var = None

def calledRepeatedly(called):
    if False:
        return 10
    module_var
    called()
    return called
for x in itertools.repeat(None, 50000):
    calledRepeatedly(empty)
print('OK.')