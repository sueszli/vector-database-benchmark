from typing import Callable
from .handler import Handler

class RawUpdateHandler(Handler):
    """The Raw Update handler class. Used to handle raw updates. It is intended to be used with
    :meth:`~pyrogram.Client.add_handler`

    For a nicer way to register this handler, have a look at the
    :meth:`~pyrogram.Client.on_raw_update` decorator.

    Parameters:
        callback (``Callable``):
            A function that will be called when a new update is received from the server. It takes
            *(client, update, users, chats)* as positional arguments (look at the section below for
            a detailed description).

    Other Parameters:
        client (:obj:`~pyrogram.Client`):
            The Client itself, useful when you want to call other API methods inside the update handler.

        update (``Update``):
            The received update, which can be one of the many single Updates listed in the
            :obj:`~pyrogram.raw.base.Update` base type.

        users (``dict``):
            Dictionary of all :obj:`~pyrogram.types.User` mentioned in the update.
            You can access extra info about the user (such as *first_name*, *last_name*, etc...) by using
            the IDs you find in the *update* argument (e.g.: *users[1768841572]*).

        chats (``dict``):
            Dictionary of all :obj:`~pyrogram.types.Chat` and
            :obj:`~pyrogram.raw.types.Channel` mentioned in the update.
            You can access extra info about the chat (such as *title*, *participants_count*, etc...)
            by using the IDs you find in the *update* argument (e.g.: *chats[1701277281]*).

    Note:
        The following Empty or Forbidden types may exist inside the *users* and *chats* dictionaries.
        They mean you have been blocked by the user or banned from the group/channel.

        - :obj:`~pyrogram.raw.types.UserEmpty`
        - :obj:`~pyrogram.raw.types.ChatEmpty`
        - :obj:`~pyrogram.raw.types.ChatForbidden`
        - :obj:`~pyrogram.raw.types.ChannelForbidden`
    """

    def __init__(self, callback: Callable):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(callback)