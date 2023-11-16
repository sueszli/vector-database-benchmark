from typing import Callable
import pyrogram
from pyrogram.filters import Filter

class OnDeletedMessages:

    def on_deleted_messages(self=None, filters=None, group: int=0) -> Callable:
        if False:
            return 10
        'Decorator for handling deleted messages.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.DeletedMessagesHandler`.\n\n        Parameters:\n            filters (:obj:`~pyrogram.filters`, *optional*):\n                Pass one or more filters to allow only a subset of messages to be passed\n                in your function.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                i = 10
                return i + 15
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.DeletedMessagesHandler(func, filters), group)
            elif isinstance(self, Filter) or self is None:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.DeletedMessagesHandler(func, self), group if filters is None else filters))
            return func
        return decorator