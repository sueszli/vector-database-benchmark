from typing import Callable
import pyrogram

class OnDisconnect:

    def on_disconnect(self=None) -> Callable:
        if False:
            i = 10
            return i + 15
        'Decorator for handling disconnections.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.DisconnectHandler`.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                print('Hello World!')
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.DisconnectHandler(func))
            else:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.DisconnectHandler(func), 0))
            return func
        return decorator