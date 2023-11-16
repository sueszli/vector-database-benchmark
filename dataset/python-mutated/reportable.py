"""Provide the ReportableMixin class."""
from ....const import API_PATH

class ReportableMixin:
    """Interface for :class:`.RedditBase` classes that can be reported."""

    def report(self, reason: str):
        if False:
            for i in range(10):
                print('nop')
        'Report this object to the moderators of its subreddit.\n\n        :param reason: The reason for reporting.\n\n        :raises: :class:`.RedditAPIException` if ``reason`` is longer than 100\n            characters.\n\n        Example usage:\n\n        .. code-block:: python\n\n            submission = reddit.submission("5or86n")\n            submission.report("report reason")\n\n            comment = reddit.comment("dxolpyc")\n            comment.report("report reason")\n\n        '
        self._reddit.post(API_PATH['report'], data={'id': self.fullname, 'reason': reason})