import ray
import time
import itertools
ray.init()
numbers = list(range(10000))

def double(number):
    if False:
        i = 10
        return i + 15
    time.sleep(1e-05)
    return number * 2
start_time = time.time()
serial_doubled_numbers = [double(number) for number in numbers]
end_time = time.time()
print(f'Ordinary funciton call takes {end_time - start_time} seconds')

@ray.remote
def remote_double(number):
    if False:
        while True:
            i = 10
    return double(number)
start_time = time.time()
doubled_number_refs = [remote_double.remote(number) for number in numbers]
parallel_doubled_numbers = ray.get(doubled_number_refs)
end_time = time.time()
print(f'Parallelizing tasks takes {end_time - start_time} seconds')
assert serial_doubled_numbers == parallel_doubled_numbers

@ray.remote
def remote_double_batch(numbers):
    if False:
        i = 10
        return i + 15
    return [double(number) for number in numbers]
BATCH_SIZE = 1000
start_time = time.time()
doubled_batch_refs = []
for i in range(0, len(numbers), BATCH_SIZE):
    batch = numbers[i:i + BATCH_SIZE]
    doubled_batch_refs.append(remote_double_batch.remote(batch))
parallel_doubled_numbers_with_batching = list(itertools.chain(*ray.get(doubled_batch_refs)))
end_time = time.time()
print(f'Parallelizing tasks with batching takes {end_time - start_time} seconds')
assert serial_doubled_numbers == parallel_doubled_numbers_with_batching