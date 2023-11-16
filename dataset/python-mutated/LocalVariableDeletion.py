import itertools
module_value1 = 1000
module_value2 = None
module_value3 = None

def calledRepeatedly():
    if False:
        print('Hello World!')
    module_value1
    local_value = module_value1
    global module_value2, module_value3
    module_value2 = local_value
    if module_value2 is None:
        del local_value
        another_local_value = module_value3
    module_value2 = module_value1
    local_value = module_value3
    return local_value
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')