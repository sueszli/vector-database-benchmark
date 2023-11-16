import itertools
module_value1 = 5.0
module_value2 = 3.0

def calledRepeatedly():
    if False:
        return 10
    module_value1
    s = 2.0
    local_value = module_value1
    s += module_value1
    s += 1000.0
    s += 1000.0
    s += 1000.0
    s += 1000.0
    s += 1000.0
    s += module_value2
    return s
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')