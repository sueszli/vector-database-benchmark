import asyncio

def create_remote_signal_actor(ray):
    if False:
        while True:
            i = 10

    @ray.remote
    class SignalActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ready_event = asyncio.Event()

        def send(self, clear=False):
            if False:
                i = 10
                return i + 15
            self.ready_event.set()
            if clear:
                self.ready_event.clear()

        async def wait(self, should_wait=True):
            if should_wait:
                await self.ready_event.wait()
    return SignalActor

def run_wrapped_actor_creation():
    if False:
        for i in range(10):
            print('nop')
    import ray
    RemoteClass = ray.remote(SomeClass)
    handle = RemoteClass.remote()
    return ray.get(handle.ready.remote())

class SomeClass:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def ready(self):
        if False:
            while True:
                i = 10
        return 1