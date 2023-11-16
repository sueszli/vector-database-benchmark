import itertools
module_value1 = 1000

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')
    module_value1
    l = [x for x in range(7)]
    l = 1
    return (l, x)
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')