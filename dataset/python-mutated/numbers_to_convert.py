class MyObject:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def __int__(self):
        if False:
            return 10
        return 42 // self.value

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'MyObject'

def get_variables():
    if False:
        i = 10
        return i + 15
    return {'object': MyObject(1), 'object_failing': MyObject(0)}