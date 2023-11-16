"""Provide the InboxToggleableMixin class."""
from ....const import API_PATH

class InboxToggleableMixin:
    """Interface for classes that can optionally receive inbox replies."""

    def disable_inbox_replies(self):
        if False:
            for i in range(10):
                print('nop')
        'Disable inbox replies for the item.\n\n        .. note::\n\n            This can only apply to items created by the authenticated user.\n\n        Example usage:\n\n        .. code-block:: python\n\n            comment = reddit.comment("dkk4qjd")\n            comment.disable_inbox_replies()\n\n            submission = reddit.submission("8dmv8z")\n            submission.disable_inbox_replies()\n\n        .. seealso::\n\n            :meth:`.enable_inbox_replies`\n\n        '
        self._reddit.post(API_PATH['sendreplies'], data={'id': self.fullname, 'state': False})

    def enable_inbox_replies(self):
        if False:
            return 10
        'Enable inbox replies for the item.\n\n        .. note::\n\n            This can only apply to items created by the authenticated user.\n\n        Example usage:\n\n        .. code-block:: python\n\n            comment = reddit.comment("dkk4qjd")\n            comment.enable_inbox_replies()\n\n            submission = reddit.submission("8dmv8z")\n            submission.enable_inbox_replies()\n\n        .. seealso::\n\n            :meth:`.disable_inbox_replies`\n\n        '
        self._reddit.post(API_PATH['sendreplies'], data={'id': self.fullname, 'state': True})