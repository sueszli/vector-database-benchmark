from typing import Callable
import pyrogram
from pyrogram.filters import Filter

class OnChosenInlineResult:

    def on_chosen_inline_result(self=None, filters=None, group: int=0) -> Callable:
        if False:
            i = 10
            return i + 15
        'Decorator for handling chosen inline results.\n\n        This does the same thing as :meth:`~pyrogram.Client.add_handler` using the\n        :obj:`~pyrogram.handlers.ChosenInlineResultHandler`.\n\n        Parameters:\n            filters (:obj:`~pyrogram.filters`, *optional*):\n                Pass one or more filters to allow only a subset of chosen inline results to be passed\n                in your function.\n\n            group (``int``, *optional*):\n                The group identifier, defaults to 0.\n        '

        def decorator(func: Callable) -> Callable:
            if False:
                print('Hello World!')
            if isinstance(self, pyrogram.Client):
                self.add_handler(pyrogram.handlers.ChosenInlineResultHandler(func, filters), group)
            elif isinstance(self, Filter) or self is None:
                if not hasattr(func, 'handlers'):
                    func.handlers = []
                func.handlers.append((pyrogram.handlers.ChosenInlineResultHandler(func, self), group if filters is None else filters))
            return func
        return decorator