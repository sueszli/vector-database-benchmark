"""Provide the InboxableMixin class."""
from ....const import API_PATH

class InboxableMixin:
    """Interface for :class:`.RedditBase` subclasses that originate from the inbox."""

    def block(self):
        if False:
            while True:
                i = 10
        'Block the user who sent the item.\n\n        .. note::\n\n            This method pertains only to objects which were retrieved via the inbox.\n\n        Example usage:\n\n        .. code-block:: python\n\n            comment = reddit.comment("dkk4qjd")\n            comment.block()\n\n            # or, identically:\n\n            comment.author.block()\n\n        '
        self._reddit.post(API_PATH['block'], data={'id': self.fullname})

    def collapse(self):
        if False:
            while True:
                i = 10
        'Mark the item as collapsed.\n\n        .. note::\n\n            This method pertains only to objects which were retrieved via the inbox.\n\n        Example usage:\n\n        .. code-block:: python\n\n            inbox = reddit.inbox()\n\n            # select first inbox item and collapse it message = next(inbox)\n            message.collapse()\n\n        .. seealso::\n\n            :meth:`.uncollapse`\n\n        '
        self._reddit.inbox.collapse([self])

    def mark_read(self):
        if False:
            return 10
        'Mark a single inbox item as read.\n\n        .. note::\n\n            This method pertains only to objects which were retrieved via the inbox.\n\n        Example usage:\n\n        .. code-block:: python\n\n            inbox = reddit.inbox.unread()\n\n            for message in inbox:\n                # process unread messages\n                ...\n\n        .. seealso::\n\n            :meth:`.mark_unread`\n\n        To mark the whole inbox as read with a single network request, use\n        :meth:`.Inbox.mark_all_read`\n\n        '
        self._reddit.inbox.mark_read([self])

    def mark_unread(self):
        if False:
            return 10
        'Mark the item as unread.\n\n        .. note::\n\n            This method pertains only to objects which were retrieved via the inbox.\n\n        Example usage:\n\n        .. code-block:: python\n\n            inbox = reddit.inbox(limit=10)\n\n            for message in inbox:\n                # process messages\n                ...\n\n        .. seealso::\n\n            :meth:`.mark_read`\n\n        '
        self._reddit.inbox.mark_unread([self])

    def unblock_subreddit(self):
        if False:
            for i in range(10):
                print('nop')
        'Unblock a subreddit.\n\n        .. note::\n\n            This method pertains only to objects which were retrieved via the inbox.\n\n        For example, to unblock all blocked subreddits that you can find by going\n        through your inbox:\n\n        .. code-block:: python\n\n            from praw.models import SubredditMessage\n\n            subs = set()\n            for item in reddit.inbox.messages(limit=None):\n                if isinstance(item, SubredditMessage):\n                    if (\n                        item.subject == "[message from blocked subreddit]"\n                        and str(item.subreddit) not in subs\n                    ):\n                        item.unblock_subreddit()\n                        subs.add(str(item.subreddit))\n\n        '
        self._reddit.post(API_PATH['unblock_subreddit'], data={'id': self.fullname})

    def uncollapse(self):
        if False:
            for i in range(10):
                print('nop')
        'Mark the item as uncollapsed.\n\n        .. note::\n\n            This method pertains only to objects which were retrieved via the inbox.\n\n        Example usage:\n\n        .. code-block:: python\n\n            inbox = reddit.inbox()\n\n            # select first inbox item and uncollapse it\n            message = next(inbox)\n            message.uncollapse()\n\n        .. seealso::\n\n            :meth:`.collapse`\n\n        '
        self._reddit.inbox.uncollapse([self])