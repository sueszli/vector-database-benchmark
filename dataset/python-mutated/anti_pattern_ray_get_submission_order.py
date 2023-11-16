import random
import time
import ray
ray.init()

@ray.remote
def f(i):
    if False:
        for i in range(10):
            print('nop')
    time.sleep(random.random())
    return i
sum_in_submission_order = 0
refs = [f.remote(i) for i in range(100)]
for ref in refs:
    result = ray.get(ref)
    sum_in_submission_order = sum_in_submission_order + result
sum_in_completion_order = 0
refs = [f.remote(i) for i in range(100)]
unfinished = refs
while unfinished:
    (finished, unfinished) = ray.wait(unfinished, num_returns=1)
    result = ray.get(finished[0])
    sum_in_completion_order = sum_in_completion_order + result
assert sum_in_submission_order == sum_in_completion_order