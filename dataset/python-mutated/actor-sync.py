import asyncio
import ray

@ray.remote(num_cpus=0)
class SignalActor:

    def __init__(self):
        if False:
            return 10
        self.ready_event = asyncio.Event()

    def send(self, clear=False):
        if False:
            for i in range(10):
                print('nop')
        self.ready_event.set()
        if clear:
            self.ready_event.clear()

    async def wait(self, should_wait=True):
        if should_wait:
            await self.ready_event.wait()

@ray.remote
def wait_and_go(signal):
    if False:
        return 10
    ray.get(signal.wait.remote())
    print('go!')
signal = SignalActor.remote()
tasks = [wait_and_go.remote(signal) for _ in range(4)]
print('ready...')
print('set..')
ray.get(signal.send.remote())
ray.get(tasks)