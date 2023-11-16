import asyncio
from typing import Callable

def call_after(function: Callable, after: Callable):
    if False:
        print('Hello World!')
    "Run a callable after another has executed. Useful when trying to make sure that a function\n    did actually run, but just monkeypatching it doesn't work because this would break some other\n    functionality.\n\n    Example usage:\n\n    def test_stuff(self, bot, monkeypatch):\n\n        def after(arg):\n            # arg is the return value of `send_message`\n            self.received = arg\n\n        monkeypatch.setattr(bot, 'send_message', call_after(bot.send_message, after)\n\n    "
    if asyncio.iscoroutinefunction(function):

        async def wrapped(*args, **kwargs):
            out = await function(*args, **kwargs)
            if asyncio.iscoroutinefunction(after):
                await after(out)
            else:
                after(out)
            return out
    else:

        def wrapped(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            out = function(*args, **kwargs)
            after(out)
            return out
    return wrapped