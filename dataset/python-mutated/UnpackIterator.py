import itertools
module_var = 1
range_arg = 2

def calledRepeatedly():
    if False:
        print('Hello World!')
    global module_var
    c = module_var
    iterator = iter(range(range_arg))
    (a, b) = iterator
    a = c
    b = c
    return (a, b)
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')