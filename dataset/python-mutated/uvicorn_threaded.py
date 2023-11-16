import contextlib
import threading
import time
import uvicorn

def asyncio_setup() -> None:
    if False:
        return 10
    import sys
    if sys.version_info >= (3, 8) and sys.platform == 'win32':
        import asyncio
        import selectors
        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(selector)
        asyncio.set_event_loop(loop)

class UvicornServer(uvicorn.Server):
    """
    Multithreaded server - as found in https://github.com/encode/uvicorn/issues/742

    Removed install_signal_handlers() override based on changes from this commit:
        https://github.com/encode/uvicorn/commit/ce2ef45a9109df8eae038c0ec323eb63d644cbc6

    Cannot rely on asyncio.get_event_loop() to create new event loop because of this check:
        https://github.com/python/cpython/blob/4d7f11e05731f67fd2c07ec2972c6cb9861d52be/Lib/asyncio/events.py#L638

    Fix by overriding run() and forcing creation of new event loop if uvloop is available
    """

    def run(self, sockets=None):
        if False:
            print('Hello World!')
        import asyncio
        '\n        Parent implementation calls self.config.setup_event_loop(),\n            but we need to create uvloop event loop manually\n        '
        try:
            import uvloop
        except ImportError:
            asyncio_setup()
        else:
            asyncio.set_event_loop(uvloop.new_event_loop())
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        loop.run_until_complete(self.serve(sockets=sockets))

    @contextlib.contextmanager
    def run_in_thread(self):
        if False:
            return 10
        self.thread = threading.Thread(target=self.run, name='FTUvicorn')
        self.thread.start()
        while not self.started:
            time.sleep(0.001)

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        self.should_exit = True
        self.thread.join()