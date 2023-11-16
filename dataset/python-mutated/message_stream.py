import asyncio
import time

class MessageStream:
    """
    A message stream for consumers to subscribe to,
    and for producers to publish to.
    """

    def __init__(self):
        if False:
            return 10
        self._loop = asyncio.get_running_loop()
        self._waiter = self._loop.create_future()

    def publish(self, message):
        if False:
            for i in range(10):
                print('nop')
        '\n        Publish a message to this MessageStream\n\n        :param message: The message to publish\n        '
        (waiter, self._waiter) = (self._waiter, self._loop.create_future())
        waiter.set_result((message, time.time(), self._waiter))

    async def __aiter__(self):
        """
        Iterate over the messages in the message stream
        """
        waiter = self._waiter
        while True:
            (message, ts, waiter) = await asyncio.shield(waiter)
            yield (message, ts)