import itertools
module_value1 = -1
module_value2 = 17

def calledRepeatedly():
    if False:
        while True:
            i = 10
    module_value1
    local_value = module_value1
    s = module_value1
    t = module_value2
    t = s ** t
    return (s, t, local_value)
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')