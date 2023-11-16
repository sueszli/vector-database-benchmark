"""Provide the Front class."""
from __future__ import annotations
from typing import TYPE_CHECKING, Iterator
from urllib.parse import urljoin
from .listing.generator import ListingGenerator
from .listing.mixins import SubredditListingMixin
if TYPE_CHECKING:
    import praw.models

class Front(SubredditListingMixin):
    """Front is a Listing class that represents the front page."""

    def __init__(self, reddit: praw.Reddit):
        if False:
            print('Hello World!')
        'Initialize a :class:`.Front` instance.'
        super().__init__(reddit, _data=None)
        self._path = '/'

    def best(self, **generator_kwargs: str | int) -> Iterator[praw.models.Submission]:
        if False:
            i = 10
            return i + 15
        'Return a :class:`.ListingGenerator` for best items.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        '
        return ListingGenerator(self._reddit, urljoin(self._path, 'best'), **generator_kwargs)