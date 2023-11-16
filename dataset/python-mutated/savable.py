"""Provide the SavableMixin class."""
from __future__ import annotations
from ....const import API_PATH
from ....util import _deprecate_args

class SavableMixin:
    """Interface for :class:`.RedditBase` classes that can be saved."""

    @_deprecate_args('category')
    def save(self, *, category: str | None=None):
        if False:
            print('Hello World!')
        'Save the object.\n\n        :param category: The category to save to. If the authenticated user does not\n            have Reddit Premium this value is ignored by Reddit (default: ``None``).\n\n        Example usage:\n\n        .. code-block:: python\n\n            submission = reddit.submission("5or86n")\n            submission.save(category="view later")\n\n            comment = reddit.comment("dxolpyc")\n            comment.save()\n\n        .. seealso::\n\n            :meth:`.unsave`\n\n        '
        self._reddit.post(API_PATH['save'], data={'category': category, 'id': self.fullname})

    def unsave(self):
        if False:
            for i in range(10):
                print('nop')
        'Unsave the object.\n\n        Example usage:\n\n        .. code-block:: python\n\n            submission = reddit.submission("5or86n")\n            submission.unsave()\n\n            comment = reddit.comment("dxolpyc")\n            comment.unsave()\n\n        .. seealso::\n\n            :meth:`.save`\n\n        '
        self._reddit.post(API_PATH['unsave'], data={'id': self.fullname})