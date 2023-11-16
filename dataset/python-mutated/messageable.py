"""Provide the MessageableMixin class."""
from __future__ import annotations
from typing import TYPE_CHECKING
from ....const import API_PATH
from ....util import _deprecate_args
if TYPE_CHECKING:
    import praw

class MessageableMixin:
    """Interface for classes that can be messaged."""

    @_deprecate_args('subject', 'message', 'from_subreddit')
    def message(self, *, from_subreddit: praw.models.Subreddit | str | None=None, message: str, subject: str):
        if False:
            for i in range(10):
                print('nop')
        'Send a message to a :class:`.Redditor` or a :class:`.Subreddit`\'s moderators (modmail).\n\n        :param from_subreddit: A :class:`.Subreddit` instance or string to send the\n            message from. When provided, messages are sent from the subreddit rather\n            than from the authenticated user.\n\n            .. note::\n\n                The authenticated user must be a moderator of the subreddit and have the\n                ``mail`` moderator permission.\n\n        :param message: The message content.\n        :param subject: The subject of the message.\n\n        For example, to send a private message to u/spez, try:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").message(subject="TEST", message="test message from PRAW")\n\n        To send a message to u/spez from the moderators of r/test try:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").message(\n                subject="TEST", message="test message from r/test", from_subreddit="test"\n            )\n\n        To send a message to the moderators of r/test, try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").message(subject="TEST", message="test PM from PRAW")\n\n        '
        data = {'subject': subject, 'text': message, 'to': f"{getattr(self.__class__, 'MESSAGE_PREFIX', '')}{self}"}
        if from_subreddit:
            data['from_sr'] = str(from_subreddit)
        self._reddit.post(API_PATH['compose'], data=data)