import ray
ray.init()

@ray.remote(num_cpus=1)
def f():
    if False:
        i = 10
        return i + 15
    pass
print('Hanging...')
ray.get(f.remote())
print('Success!')