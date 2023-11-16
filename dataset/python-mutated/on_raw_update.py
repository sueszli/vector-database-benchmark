from typing import Callable
import pyrogram

class OnRawUpdate:

    def on_raw_update(self=None, group: int=0) -> Callable:
        if False:
            i = 10
            return i + 15
        'Decorator for handling raw updates.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.RawUpdateHandler`.\n\n        Parameters:\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.RawUpdateHandler(func), group)
            else:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.RawUpdateHandler(func), group))
            return func
        return decorator