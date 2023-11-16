import ray
ray.init()
outputs = []
for i in range(10):

    @ray.remote
    def double(i):
        if False:
            for i in range(10):
                print('nop')
        return i * 2
    outputs.append(double.remote(i))
outputs = ray.get(outputs)
assert outputs == [i * 2 for i in range(10)]

@ray.remote
def double(i):
    if False:
        while True:
            i = 10
    return i * 2
outputs = []
for i in range(10):
    outputs.append(double.remote(i))
outputs = ray.get(outputs)
assert outputs == [i * 2 for i in range(10)]