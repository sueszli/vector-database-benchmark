import itertools
module_value1 = 1000
module_value2 = None
module_value3 = None

def calledRepeatedly():
    if False:
        for i in range(10):
            print('nop')
    closure_value = module_value1

    def f():
        if False:
            for i in range(10):
                print('nop')
        module_value1
        global module_value2, module_value3
        module_value2 = closure_value
        module_value3 = closure_value
    f()
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')