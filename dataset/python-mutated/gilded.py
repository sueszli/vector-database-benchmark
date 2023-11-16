"""Provide the GildedListingMixin class."""
from __future__ import annotations
from typing import Any, Iterator
from urllib.parse import urljoin
from ...base import PRAWBase
from ..generator import ListingGenerator

class GildedListingMixin(PRAWBase):
    """Mixes in the gilded method."""

    def gilded(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[Any]:
        if False:
            print('Hello World!')
        'Return a :class:`.ListingGenerator` for gilded items.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        For example, to get gilded items in r/test:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("test").gilded():\n                print(item.id)\n\n        '
        return ListingGenerator(self._reddit, urljoin(self._path, 'gilded'), **generator_kwargs)