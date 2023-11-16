from typing import Callable
from .handler import Handler

class EditedMessageHandler(Handler):
    """The EditedMessage handler class. Used to handle edited messages.
     It is intended to be used with :meth:`~pyrogram.Client.add_handler`

    For a nicer way to register this handler, have a look at the
    :meth:`~pyrogram.Client.on_edited_message` decorator.

    Parameters:
        callback (``Callable``):
            Pass a function that will be called when a new edited message arrives. It takes *(client, message)*
            as positional arguments (look at the section below for a detailed description).

        filters (:obj:`Filters`):
            Pass one or more filters to allow only a subset of messages to be passed
            in your callback function.

    Other parameters:
        client (:obj:`~pyrogram.Client`):
            The Client itself, useful when you want to call other API methods inside the message handler.

        edited_message (:obj:`~pyrogram.types.Message`):
            The received edited message.
    """

    def __init__(self, callback: Callable, filters=None):
        if False:
            i = 10
            return i + 15
        super().__init__(callback, filters)