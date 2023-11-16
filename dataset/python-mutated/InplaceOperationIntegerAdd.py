import itertools
module_value1 = 5
module_value2 = 3

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')
    module_value1
    s = 2
    local_value = module_value1
    s += module_value1
    s += 1000
    s += 1000
    s += 1000
    s += 1000
    s += 1000
    s += module_value2
    return s
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')