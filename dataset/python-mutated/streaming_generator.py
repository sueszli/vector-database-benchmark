import ray
import time

@ray.remote(num_returns='streaming')
def task():
    if False:
        print('Hello World!')
    for i in range(5):
        time.sleep(5)
        yield i
gen = task.remote()
ref = next(gen)
ray.get(ref)
ref = next(gen)
ray.get(ref)
for ref in gen:
    print(ray.get(ref))

@ray.remote(num_returns='streaming')
def task():
    if False:
        return 10
    for i in range(5):
        time.sleep(1)
        if i == 1:
            raise ValueError
        yield i
gen = task.remote()
ray.get(next(gen))
try:
    ray.get(next(gen))
except ValueError as e:
    print(f'Exception is raised when i == 1 as expected {e}')

@ray.remote
class Actor:

    def f(self):
        if False:
            print('Hello World!')
        for i in range(5):
            yield i

@ray.remote
class AsyncActor:

    async def f(self):
        for i in range(5):
            yield i

@ray.remote(max_concurrency=5)
class ThreadedActor:

    def f(self):
        if False:
            return 10
        for i in range(5):
            yield i
actor = Actor.remote()
for ref in actor.f.options(num_returns='streaming').remote():
    print(ray.get(ref))
actor = AsyncActor.remote()
for ref in actor.f.options(num_returns='streaming').remote():
    print(ray.get(ref))
actor = ThreadedActor.remote()
for ref in actor.f.options(num_returns='streaming').remote():
    print(ray.get(ref))
import asyncio

@ray.remote(num_returns='streaming')
def task():
    if False:
        i = 10
        return i + 15
    for i in range(5):
        time.sleep(1)
        yield i

async def main():
    async for ref in task.remote():
        print(await ref)
asyncio.run(main())

@ray.remote(num_returns='streaming')
def task():
    if False:
        print('Hello World!')
    for i in range(5):
        time.sleep(1)
        yield i
gen = task.remote()
ref1 = next(gen)
del gen
import asyncio

@ray.remote(num_returns='streaming')
def task():
    if False:
        return 10
    for i in range(5):
        time.sleep(1)
        yield i

async def async_task():
    async for ref in task.remote():
        print(await ref)

async def main():
    t1 = async_task()
    t2 = async_task()
    await asyncio.gather(t1, t2)
asyncio.run(main())

@ray.remote(num_returns='streaming')
def task():
    if False:
        i = 10
        return i + 15
    for i in range(5):
        time.sleep(5)
        yield i
gen = task.remote()
(ready, unready) = ray.wait([gen], timeout=0)
print('timeout 0, nothing is ready.')
print(ready)
assert len(ready) == 0
assert len(unready) == 1
(ready, unready) = ray.wait([gen])
print('Wait for 5 seconds. The next item is ready.')
assert len(ready) == 1
assert len(unready) == 0
next(gen)
(ready, unready) = ray.wait([gen], timeout=0)
print('Wait for 0 seconds. The next item is not ready.')
print(ready, unready)
assert len(ready) == 0
assert len(unready) == 1
from ray._raylet import StreamingObjectRefGenerator

@ray.remote(num_returns='streaming')
def generator_task():
    if False:
        return 10
    for i in range(5):
        time.sleep(5)
        yield i

@ray.remote
def regular_task():
    if False:
        while True:
            i = 10
    for i in range(5):
        time.sleep(5)
    return
gen = [generator_task.remote()]
ref = [regular_task.remote()]
(ready, unready) = ([], [*gen, *ref])
result = []
while unready:
    (ready, unready) = ray.wait(unready)
    for r in ready:
        if isinstance(r, StreamingObjectRefGenerator):
            try:
                ref = next(r)
                result.append(ray.get(ref))
            except StopIteration:
                pass
            else:
                unready.append(r)
        else:
            result.append(ray.get(r))