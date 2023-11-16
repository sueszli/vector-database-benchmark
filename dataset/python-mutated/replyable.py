"""Provide the ReplyableMixin class."""
from __future__ import annotations
from typing import TYPE_CHECKING
from ....const import API_PATH
if TYPE_CHECKING:
    import praw.models

class ReplyableMixin:
    """Interface for :class:`.RedditBase` classes that can be replied to."""

    def reply(self, body: str) -> praw.models.Comment | praw.models.Message | None:
        if False:
            i = 10
            return i + 15
        'Reply to the object.\n\n        :param body: The Markdown formatted content for a comment.\n\n        :returns: A :class:`.Comment` or :class:`.Message` object for the newly created\n            comment or message or ``None`` if Reddit doesn\'t provide one.\n\n        :raises: ``prawcore.exceptions.Forbidden`` when attempting to reply to some\n            items, such as locked submissions/comments or non-replyable messages.\n\n        A ``None`` value can be returned if the target is a comment or submission in a\n        quarantined subreddit and the authenticated user has not opt-ed into viewing the\n        content. When this happens the comment will be successfully created on Reddit\n        and can be retried by drawing the comment from the user\'s comment history.\n\n        Example usage:\n\n        .. code-block:: python\n\n            submission = reddit.submission("5or86n")\n            submission.reply("reply")\n\n            comment = reddit.comment("dxolpyc")\n            comment.reply("reply")\n\n        '
        data = {'text': body, 'thing_id': self.fullname}
        comments = self._reddit.post(API_PATH['comment'], data=data)
        try:
            return comments[0]
        except IndexError:
            return None