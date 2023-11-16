import itertools
import sys
module_value1 = '1000'
module_value2 = '2000'
loop_count = 50000 if len(sys.argv) < 2 else int(sys.argv[1])

def calledRepeatedly(value1, value2):
    if False:
        for i in range(10):
            print('nop')
    module_value1
    if value1 == value2:
        return
    return (value1, value2)
for x in itertools.repeat(None, 50000):
    calledRepeatedly(module_value1, module_value2)
print('OK.')