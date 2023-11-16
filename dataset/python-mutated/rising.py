"""Provide the RisingListingMixin class."""
from __future__ import annotations
from typing import TYPE_CHECKING, Iterator
from urllib.parse import urljoin
from ...base import PRAWBase
from ..generator import ListingGenerator
if TYPE_CHECKING:
    import praw.models

class RisingListingMixin(PRAWBase):
    """Mixes in the rising methods."""

    def random_rising(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Submission]:
        if False:
            print('Hello World!')
        'Return a :class:`.ListingGenerator` for random rising submissions.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        For example, to get random rising submissions for r/test:\n\n        .. code-block:: python\n\n            for submission in reddit.subreddit("test").random_rising():\n                print(submission.title)\n\n        '
        return ListingGenerator(self._reddit, urljoin(self._path, 'randomrising'), **generator_kwargs)

    def rising(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Submission]:
        if False:
            print('Hello World!')
        'Return a :class:`.ListingGenerator` for rising submissions.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        For example, to get rising submissions for r/test:\n\n        .. code-block:: python\n\n            for submission in reddit.subreddit("test").rising():\n                print(submission.title)\n\n        '
        return ListingGenerator(self._reddit, urljoin(self._path, 'rising'), **generator_kwargs)