from typing import Callable
from .handler import Handler

class PollHandler(Handler):
    """The Poll handler class. Used to handle polls updates.

    It is intended to be used with :meth:`~pyrogram.Client.add_handler`

    For a nicer way to register this handler, have a look at the
    :meth:`~pyrogram.Client.on_poll` decorator.

    Parameters:
        callback (``Callable``):
            Pass a function that will be called when a new poll update arrives. It takes *(client, poll)*
            as positional arguments (look at the section below for a detailed description).

        filters (:obj:`Filters`):
            Pass one or more filters to allow only a subset of polls to be passed
            in your callback function.

    Other parameters:
        client (:obj:`~pyrogram.Client`):
            The Client itself, useful when you want to call other API methods inside the poll handler.

        poll (:obj:`~pyrogram.types.Poll`):
            The received poll.
    """

    def __init__(self, callback: Callable, filters=None):
        if False:
            while True:
                i = 10
        super().__init__(callback, filters)