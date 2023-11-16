import ray
import numpy as np

@ray.remote
def task_with_single_small_return_value_bad():
    if False:
        while True:
            i = 10
    small_return_value = 1
    small_return_value_ref = ray.put(small_return_value)
    return small_return_value_ref

@ray.remote
def task_with_single_small_return_value_good():
    if False:
        while True:
            i = 10
    small_return_value = 1
    return small_return_value
assert ray.get(ray.get(task_with_single_small_return_value_bad.remote())) == ray.get(task_with_single_small_return_value_good.remote())

@ray.remote
def task_with_single_large_return_value_bad():
    if False:
        return 10
    large_return_value = np.zeros(10 * 1024 * 1024)
    large_return_value_ref = ray.put(large_return_value)
    return large_return_value_ref

@ray.remote
def task_with_single_large_return_value_good():
    if False:
        print('Hello World!')
    large_return_value = np.zeros(10 * 1024 * 1024)
    return large_return_value
assert np.array_equal(ray.get(ray.get(task_with_single_large_return_value_bad.remote())), ray.get(task_with_single_large_return_value_good.remote()))

@ray.remote
class Actor:

    def task_with_single_return_value_bad(self):
        if False:
            return 10
        single_return_value = np.zeros(9 * 1024 * 1024)
        return ray.put(single_return_value)

    def task_with_single_return_value_good(self):
        if False:
            return 10
        return np.zeros(9 * 1024 * 1024)
actor = Actor.remote()
assert np.array_equal(ray.get(ray.get(actor.task_with_single_return_value_bad.remote())), ray.get(actor.task_with_single_return_value_good.remote()))

@ray.remote(num_returns=1)
def task_with_static_multiple_returns_bad1():
    if False:
        while True:
            i = 10
    return_value_1_ref = ray.put(1)
    return_value_2_ref = ray.put(2)
    return (return_value_1_ref, return_value_2_ref)

@ray.remote(num_returns=2)
def task_with_static_multiple_returns_bad2():
    if False:
        i = 10
        return i + 15
    return_value_1_ref = ray.put(1)
    return_value_2_ref = ray.put(2)
    return (return_value_1_ref, return_value_2_ref)

@ray.remote(num_returns=2)
def task_with_static_multiple_returns_good():
    if False:
        while True:
            i = 10
    return_value_1 = 1
    return_value_2 = 2
    return (return_value_1, return_value_2)
assert ray.get(ray.get(task_with_static_multiple_returns_bad1.remote())[0]) == ray.get(ray.get(task_with_static_multiple_returns_bad2.remote()[0])) == ray.get(task_with_static_multiple_returns_good.remote()[0])

@ray.remote
class Actor:

    @ray.method(num_returns=1)
    def task_with_static_multiple_returns_bad1(self):
        if False:
            for i in range(10):
                print('nop')
        return_value_1_ref = ray.put(1)
        return_value_2_ref = ray.put(2)
        return (return_value_1_ref, return_value_2_ref)

    @ray.method(num_returns=2)
    def task_with_static_multiple_returns_bad2(self):
        if False:
            return 10
        return_value_1_ref = ray.put(1)
        return_value_2_ref = ray.put(2)
        return (return_value_1_ref, return_value_2_ref)

    @ray.method(num_returns=2)
    def task_with_static_multiple_returns_good(self):
        if False:
            return 10
        return_value_1 = 1
        return_value_2 = 2
        return (return_value_1, return_value_2)
actor = Actor.remote()
assert ray.get(ray.get(actor.task_with_static_multiple_returns_bad1.remote())[0]) == ray.get(ray.get(actor.task_with_static_multiple_returns_bad2.remote()[0])) == ray.get(actor.task_with_static_multiple_returns_good.remote()[0])

@ray.remote(num_returns=1)
def task_with_dynamic_returns_bad(n):
    if False:
        print('Hello World!')
    return_value_refs = []
    for i in range(n):
        return_value_refs.append(ray.put(np.zeros(i * 1024 * 1024)))
    return return_value_refs

@ray.remote(num_returns='dynamic')
def task_with_dynamic_returns_good(n):
    if False:
        while True:
            i = 10
    for i in range(n):
        yield np.zeros(i * 1024 * 1024)
assert np.array_equal(ray.get(ray.get(task_with_dynamic_returns_bad.remote(2))[0]), ray.get(next(iter(ray.get(task_with_dynamic_returns_good.remote(2))))))