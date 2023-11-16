import ray
ray.init()

@ray.remote
def process(file):
    if False:
        print('Hello World!')
    pass
NUM_FILES = 1000
result_refs = []
for i in range(NUM_FILES):
    result_refs.append(process.remote(f'{i}.csv'))
ray.get(result_refs)
result_refs = []
for i in range(NUM_FILES):
    result_refs.append(process.options(memory=2 * 1024 * 1024 * 1024).remote(f'{i}.csv'))
ray.get(result_refs)