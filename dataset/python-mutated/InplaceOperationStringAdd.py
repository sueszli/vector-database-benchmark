import itertools
module_value1 = 5
additiv_global = '*' * 1000

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')
    module_value1
    s = '2'
    additiv = additiv_global
    s += additiv
    s += additiv
    s += additiv
    s += additiv
    s += additiv
    s += additiv
    s += additiv
    return s
for x in itertools.repeat(None, 10000):
    calledRepeatedly()
print('OK.')