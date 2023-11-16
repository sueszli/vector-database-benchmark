import itertools
module_var = None

def calledRepeatedly(x):
    if False:
        i = 10
        return i + 15
    module_var

    def empty():
        if False:
            return 10
        return x

    def empty():
        if False:
            return 10
        return module_var
    return (empty, x)
for x in itertools.repeat(None, 50000):
    calledRepeatedly(x)
print('OK.')