import itertools
module_value1 = 5

class C:

    def __init__(self):
        if False:
            return 10
        self.s = '2' * 100000

    def increment(self):
        if False:
            i = 10
            return i + 15
        additiv = '*' * 1000
        self.s += additiv
        self.s += additiv
        self.s += additiv
        self.s += additiv
        self.s += additiv
        return additiv

def calledRepeatedly():
    if False:
        return 10
    module_value1
    local_value = C()
    local_value.increment()
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')