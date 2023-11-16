import ray
from ray.util.queue import Queue, Empty
ray.init()
queue = Queue(maxsize=100)

@ray.remote
def consumer(id, queue):
    if False:
        i = 10
        return i + 15
    try:
        while True:
            next_item = queue.get(block=True, timeout=1)
            print(f'consumer {id} got work {next_item}')
    except Empty:
        pass
[queue.put(i) for i in range(10)]
print('Put work 1 - 10 to queue...')
consumers = [consumer.remote(id, queue) for id in range(2)]
ray.get(consumers)