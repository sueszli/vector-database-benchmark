import asyncio

class AsyncTimer:
    """A non-blocking timer, that calls a function after a specified number of seconds:
    :param int interval: time interval in seconds
    :param callable callback: function to be called after the interval has elapsed
    """

    def __init__(self, interval, callback):
        if False:
            for i in range(10):
                print('nop')
        self._interval = interval
        self._callback = callback
        self._task = None

    def start(self):
        if False:
            print('Hello World!')
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._interval)
        await self._callback()

    def cancel(self):
        if False:
            print('Hello World!')
        if self._task is not None:
            self._task.cancel()
        self._task = None