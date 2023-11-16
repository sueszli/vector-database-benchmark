import ray
import time
from numpy import random

def partition(collection):
    if False:
        i = 10
        return i + 15
    pivot = collection.pop()
    (greater, lesser) = ([], [])
    for element in collection:
        if element > pivot:
            greater.append(element)
        else:
            lesser.append(element)
    return (lesser, pivot, greater)

def quick_sort(collection):
    if False:
        i = 10
        return i + 15
    if len(collection) <= 200000:
        return sorted(collection)
    else:
        (lesser, pivot, greater) = partition(collection)
        lesser = quick_sort(lesser)
        greater = quick_sort(greater)
    return lesser + [pivot] + greater

@ray.remote
def quick_sort_distributed(collection):
    if False:
        return 10
    if len(collection) <= 200000:
        return sorted(collection)
    else:
        (lesser, pivot, greater) = partition(collection)
        lesser = quick_sort_distributed.remote(lesser)
        greater = quick_sort_distributed.remote(greater)
        return ray.get(lesser) + [pivot] + ray.get(greater)
for size in [200000, 4000000, 8000000]:
    print(f'Array size: {size}')
    unsorted = random.randint(1000000, size=size).tolist()
    s = time.time()
    quick_sort(unsorted)
    print(f'Sequential execution: {time.time() - s:.3f}')
    s = time.time()
    ray.get(quick_sort_distributed.remote(unsorted))
    print(f'Distributed execution: {time.time() - s:.3f}')
    print('--' * 10)