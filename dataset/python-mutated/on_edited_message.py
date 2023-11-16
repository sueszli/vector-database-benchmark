from typing import Callable
import pyrogram
from pyrogram.filters import Filter

class OnEditedMessage:

    def on_edited_message(self=None, filters=None, group: int=0) -> Callable:
        if False:
            while True:
                i = 10
        'Decorator for handling edited messages.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.EditedMessageHandler`.\n\n        Parameters:\n            filters (:obj:`~pyrogram.filters`, *optional*):\n                Pass one or more filters to allow only a subset of messages to be passed\n                in your function.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                while True:
                    i = 10
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.EditedMessageHandler(func, filters), group)
            elif isinstance(self, Filter) or self is None:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.EditedMessageHandler(func, self), group if filters is None else filters))
            return func
        return decorator