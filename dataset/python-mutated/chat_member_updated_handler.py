from typing import Callable
from .handler import Handler

class ChatMemberUpdatedHandler(Handler):
    """The ChatMemberUpdated handler class. Used to handle changes in the status of a chat member.
    It is intended to be used with :meth:`~pyrogram.Client.add_handler`.

    For a nicer way to register this handler, have a look at the
    :meth:`~pyrogram.Client.on_chat_member_updated` decorator.

    Parameters:
        callback (``Callable``):
            Pass a function that will be called when a new ChatMemberUpdated event arrives. It takes
            *(client, chat_member_updated)* as positional arguments (look at the section below for a detailed
            description).

        filters (:obj:`Filters`):
            Pass one or more filters to allow only a subset of updates to be passed in your callback function.

    Other parameters:
        client (:obj:`~pyrogram.Client`):
            The Client itself, useful when you want to call other API methods inside the handler.

        chat_member_updated (:obj:`~pyrogram.types.ChatMemberUpdated`):
            The received chat member update.
    """

    def __init__(self, callback: Callable, filters=None):
        if False:
            while True:
                i = 10
        super().__init__(callback, filters)