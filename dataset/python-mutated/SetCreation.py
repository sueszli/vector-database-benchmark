import itertools
module_value1 = 1000

def calledRepeatedly(set_value1):
    if False:
        for i in range(10):
            print('nop')
    module_value1
    l = {set_value1, set_value1, set_value1, set_value1, set_value1}
    l = 1
    return (l, set_value1)
for x in itertools.repeat(None, 50000):
    calledRepeatedly(module_value1)
print('OK.')