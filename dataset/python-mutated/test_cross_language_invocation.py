import ray

@ray.remote
def py_return_input(v):
    if False:
        for i in range(10):
            print('nop')
    return v

@ray.remote
def py_return_val():
    if False:
        print('Hello World!')
    return 42

@ray.remote
class Counter(object):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = int(value)

    def increase(self, delta):
        if False:
            print('Hello World!')
        self.value += int(delta)
        return str(self.value)