import signal
import gsm
import trio

class GracefulShutdownManager:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._shutting_down = False
        self._cancel_scopes = set()

    def start_shutdown(self):
        if False:
            print('Hello World!')
        self._shutting_down = True
        for cancel_scope in self._cancel_scopes:
            cancel_scope.cancel()

    def cancel_on_graceful_shutdown(self):
        if False:
            i = 10
            return i + 15
        cancel_scope = trio.CancelScope()
        self._cancel_scopes.add(cancel_scope)
        if self._shutting_down:
            cancel_scope.cancel()
        return cancel_scope

    @property
    def shutting_down(self):
        if False:
            return 10
        return self._shutting_down

async def stream_handler(stream):
    while True:
        with gsm.cancel_on_graceful_shutdown():
            data = await stream.receive_some()
            print(f'data = {data!r}')
        if gsm.shutting_down:
            break

async def listen_for_shutdown_signals():
    with trio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signal_aiter:
        async for _sig in signal_aiter:
            gsm.start_shutdown()
            break