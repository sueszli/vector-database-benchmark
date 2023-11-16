from typing import Callable
import pyrogram
from pyrogram.filters import Filter

class OnCallbackQuery:

    def on_callback_query(self=None, filters=None, group: int=0) -> Callable:
        if False:
            while True:
                i = 10
        'Decorator for handling callback queries.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.CallbackQueryHandler`.\n\n        Parameters:\n            filters (:obj:`~pyrogram.filters`, *optional*):\n                Pass one or more filters to allow only a subset of callback queries to be passed\n                in your function.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                i = 10
                return i + 15
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.CallbackQueryHandler(func, filters), group)
            elif isinstance(self, Filter) or self is None:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.CallbackQueryHandler(func, self), group if filters is None else filters))
            return func
        return decorator