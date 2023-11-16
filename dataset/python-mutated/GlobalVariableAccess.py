import itertools
module_value1 = 1000
module_value2 = 2000

def calledRepeatedly():
    if False:
        return 10
    module_value1
    return module_value2
    return None
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')