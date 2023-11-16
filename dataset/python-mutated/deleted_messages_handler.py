from typing import List, Callable
import pyrogram
from pyrogram.filters import Filter
from pyrogram.types import Message
from .handler import Handler

class DeletedMessagesHandler(Handler):
    """The deleted messages handler class. Used to handle deleted messages coming from any chat
    (private, group, channel). It is intended to be used with :meth:`~pyrogram.Client.add_handler`

    For a nicer way to register this handler, have a look at the
    :meth:`~pyrogram.Client.on_deleted_messages` decorator.

    Parameters:
        callback (``Callable``):
            Pass a function that will be called when one or more messages have been deleted.
            It takes *(client, messages)* as positional arguments (look at the section below for a detailed description).

        filters (:obj:`Filters`):
            Pass one or more filters to allow only a subset of messages to be passed
            in your callback function.

    Other parameters:
        client (:obj:`~pyrogram.Client`):
            The Client itself, useful when you want to call other API methods inside the message handler.

        messages (List of :obj:`~pyrogram.types.Message`):
            The deleted messages, as list.
    """

    def __init__(self, callback: Callable, filters: Filter=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(callback, filters)

    async def check(self, client: 'pyrogram.Client', messages: List[Message]):
        for message in messages:
            if await super().check(client, message):
                return True
        else:
            return False