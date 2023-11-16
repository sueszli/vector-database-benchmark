from typing import Callable
import pyrogram
from pyrogram.filters import Filter

class OnChatMemberUpdated:

    def on_chat_member_updated(self=None, filters=None, group: int=0) -> Callable:
        if False:
            i = 10
            return i + 15
        'Decorator for handling event changes on chat members.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.ChatMemberUpdatedHandler`.\n\n        Parameters:\n            filters (:obj:`~pyrogram.filters`, *optional*):\n                Pass one or more filters to allow only a subset of updates to be passed in your function.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.ChatMemberUpdatedHandler(func, filters), group)
            elif isinstance(self, Filter) or self is None:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.ChatMemberUpdatedHandler(func, self), group if filters is None else filters))
            return func
        return decorator