"""Provide CommentForest for submission comments."""
from __future__ import annotations
from heapq import heappop, heappush
from typing import TYPE_CHECKING
from ..exceptions import DuplicateReplaceException
from ..util import _deprecate_args
from .reddit.more import MoreComments
if TYPE_CHECKING:
    import praw.models

class CommentForest:
    """A forest of comments starts with multiple top-level comments.

    Each of these comments can be a tree of replies.

    """

    @staticmethod
    def _gather_more_comments(tree: list[praw.models.MoreComments], *, parent_tree: list[praw.models.MoreComments] | None=None) -> list[MoreComments]:
        if False:
            for i in range(10):
                print('nop')
        'Return a list of :class:`.MoreComments` objects obtained from tree.'
        more_comments = []
        queue = [(None, x) for x in tree]
        while queue:
            (parent, comment) = queue.pop(0)
            if isinstance(comment, MoreComments):
                heappush(more_comments, comment)
                if parent:
                    comment._remove_from = parent.replies._comments
                else:
                    comment._remove_from = parent_tree or tree
            else:
                for item in comment.replies:
                    queue.append((comment, item))
        return more_comments

    def __getitem__(self, index: int) -> praw.models.Comment:
        if False:
            print('Hello World!')
        'Return the comment at position ``index`` in the list.\n\n        This method is to be used like an array access, such as:\n\n        .. code-block:: python\n\n            first_comment = submission.comments[0]\n\n        Alternatively, the presence of this method enables one to iterate over all top\n        level comments, like so:\n\n        .. code-block:: python\n\n            for comment in submission.comments:\n                print(comment.body)\n\n        '
        return self._comments[index]

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of top-level comments in the forest.'
        return len(self._comments)

    def _insert_comment(self, comment: praw.models.Comment):
        if False:
            print('Hello World!')
        if comment.name in self._submission._comments_by_id:
            raise DuplicateReplaceException
        comment.submission = self._submission
        if isinstance(comment, MoreComments) or comment.is_root:
            self._comments.append(comment)
        else:
            assert comment.parent_id in self._submission._comments_by_id, 'PRAW Error occurred. Please file a bug report and include the code that caused the error.'
            parent = self._submission._comments_by_id[comment.parent_id]
            parent.replies._comments.append(comment)

    def _update(self, comments: list[praw.models.Comment]):
        if False:
            while True:
                i = 10
        self._comments = comments
        for comment in comments:
            comment.submission = self._submission

    def list(self) -> list[praw.models.Comment | praw.models.MoreComments]:
        if False:
            return 10
        'Return a flattened list of all comments.\n\n        This list may contain :class:`.MoreComments` instances if :meth:`.replace_more`\n        was not called first.\n\n        '
        comments = []
        queue = list(self)
        while queue:
            comment = queue.pop(0)
            comments.append(comment)
            if not isinstance(comment, MoreComments):
                queue.extend(comment.replies)
        return comments

    def __init__(self, submission: praw.models.Submission, comments: list[praw.models.Comment] | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a :class:`.CommentForest` instance.\n\n        :param submission: An instance of :class:`.Submission` that is the parent of the\n            comments.\n        :param comments: Initialize the forest with a list of comments (default:\n            ``None``).\n\n        '
        self._comments = comments
        self._submission = submission

    @_deprecate_args('limit', 'threshold')
    def replace_more(self, *, limit: int | None=32, threshold: int=0) -> list[praw.models.MoreComments]:
        if False:
            i = 10
            return i + 15
        'Update the comment forest by resolving instances of :class:`.MoreComments`.\n\n        :param limit: The maximum number of :class:`.MoreComments` instances to replace.\n            Each replacement requires 1 API request. Set to ``None`` to have no limit,\n            or to ``0`` to remove all :class:`.MoreComments` instances without\n            additional requests (default: ``32``).\n        :param threshold: The minimum number of children comments a\n            :class:`.MoreComments` instance must have in order to be replaced.\n            :class:`.MoreComments` instances that represent "continue this thread" links\n            unfortunately appear to have 0 children (default: ``0``).\n\n        :returns: A list of :class:`.MoreComments` instances that were not replaced.\n\n        :raises: ``prawcore.TooManyRequests`` when used concurrently.\n\n        For example, to replace up to 32 :class:`.MoreComments` instances of a\n        submission try:\n\n        .. code-block:: python\n\n            submission = reddit.submission("3hahrw")\n            submission.comments.replace_more()\n\n        Alternatively, to replace :class:`.MoreComments` instances within the replies of\n        a single comment try:\n\n        .. code-block:: python\n\n            comment = reddit.comment("d8r4im1")\n            comment.refresh()\n            comment.replies.replace_more()\n\n        .. note::\n\n            This method can take a long time as each replacement will discover at most\n            100 new :class:`.Comment` instances. As a result, consider looping and\n            handling exceptions until the method returns successfully. For example:\n\n            .. code-block:: python\n\n                while True:\n                    try:\n                        submission.comments.replace_more()\n                        break\n                    except PossibleExceptions:\n                        print("Handling replace_more exception")\n                        sleep(1)\n\n        .. warning::\n\n            If this method is called, and the comments are refreshed, calling this\n            method again will result in a :class:`.DuplicateReplaceException`.\n\n        '
        remaining = limit
        more_comments = self._gather_more_comments(self._comments)
        skipped = []
        while more_comments:
            item = heappop(more_comments)
            if remaining is not None and remaining <= 0 or item.count < threshold:
                skipped.append(item)
                item._remove_from.remove(item)
                continue
            new_comments = item.comments(update=False)
            if remaining is not None:
                remaining -= 1
            for more in self._gather_more_comments(new_comments, parent_tree=self._comments):
                more.submission = self._submission
                heappush(more_comments, more)
            for comment in new_comments:
                self._insert_comment(comment)
            item._remove_from.remove(item)
        return more_comments + skipped