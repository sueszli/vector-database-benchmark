"""Provide the Subreddits class."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterator
from warnings import warn
from ..const import API_PATH
from ..util import _deprecate_args
from . import Subreddit
from .base import PRAWBase
from .listing.generator import ListingGenerator
from .util import stream_generator
if TYPE_CHECKING:
    import praw.models

class Subreddits(PRAWBase):
    """Subreddits is a Listing class that provides various subreddit lists."""

    @staticmethod
    def _to_list(subreddit_list: list[str | praw.models.Subreddit]) -> str:
        if False:
            return 10
        return ','.join([str(x) for x in subreddit_list])

    def default(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Subreddit]:
        if False:
            print('Hello World!')
        'Return a :class:`.ListingGenerator` for default subreddits.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        '
        return ListingGenerator(self._reddit, API_PATH['subreddits_default'], **generator_kwargs)

    def gold(self, **generator_kwargs: Any) -> Iterator[praw.models.Subreddit]:
        if False:
            for i in range(10):
                print('nop')
        'Alias for :meth:`.premium` to maintain backwards compatibility.'
        warn("'subreddits.gold' has be renamed to 'subreddits.premium'.", category=DeprecationWarning, stacklevel=2)
        return self.premium(**generator_kwargs)

    def new(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Subreddit]:
        if False:
            print('Hello World!')
        'Return a :class:`.ListingGenerator` for new subreddits.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        '
        return ListingGenerator(self._reddit, API_PATH['subreddits_new'], **generator_kwargs)

    def popular(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Subreddit]:
        if False:
            while True:
                i = 10
        'Return a :class:`.ListingGenerator` for popular subreddits.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        '
        return ListingGenerator(self._reddit, API_PATH['subreddits_popular'], **generator_kwargs)

    def premium(self, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Subreddit]:
        if False:
            print('Hello World!')
        'Return a :class:`.ListingGenerator` for premium subreddits.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        '
        return ListingGenerator(self._reddit, API_PATH['subreddits_gold'], **generator_kwargs)

    def recommended(self, subreddits: list[str | praw.models.Subreddit], omit_subreddits: list[str | praw.models.Subreddit] | None=None) -> list[praw.models.Subreddit]:
        if False:
            i = 10
            return i + 15
        "Return subreddits recommended for the given list of subreddits.\n\n        :param subreddits: A list of :class:`.Subreddit` instances and/or subreddit\n            names.\n        :param omit_subreddits: A list of :class:`.Subreddit` instances and/or subreddit\n            names to exclude from the results (Reddit's end may not work as expected).\n\n        "
        if not isinstance(subreddits, list):
            msg = 'subreddits must be a list'
            raise TypeError(msg)
        if omit_subreddits is not None and (not isinstance(omit_subreddits, list)):
            msg = 'omit_subreddits must be a list or None'
            raise TypeError(msg)
        params = {'omit': self._to_list(omit_subreddits or [])}
        url = API_PATH['sub_recommended'].format(subreddits=self._to_list(subreddits))
        return [Subreddit(self._reddit, sub['sr_name']) for sub in self._reddit.get(url, params=params)]

    def search(self, query: str, **generator_kwargs: str | int | dict[str, str]) -> Iterator[praw.models.Subreddit]:
        if False:
            return 10
        'Return a :class:`.ListingGenerator` of subreddits matching ``query``.\n\n        Subreddits are searched by both their title and description.\n\n        :param query: The query string to filter subreddits by.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        .. seealso::\n\n            :meth:`.search_by_name` to search by subreddit names\n\n        '
        self._safely_add_arguments(arguments=generator_kwargs, key='params', q=query)
        return ListingGenerator(self._reddit, API_PATH['subreddits_search'], **generator_kwargs)

    @_deprecate_args('query', 'include_nsfw', 'exact')
    def search_by_name(self, query: str, *, include_nsfw: bool=True, exact: bool=False) -> list[praw.models.Subreddit]:
        if False:
            i = 10
            return i + 15
        'Return list of :class:`.Subreddit`\\ s whose names begin with ``query``.\n\n        :param query: Search for subreddits beginning with this string.\n        :param exact: Return only exact matches to ``query`` (default: ``False``).\n        :param include_nsfw: Include subreddits labeled NSFW (default: ``True``).\n\n        '
        result = self._reddit.post(API_PATH['subreddits_name_search'], data={'include_over_18': include_nsfw, 'exact': exact, 'query': query})
        return [self._reddit.subreddit(x) for x in result['names']]

    def search_by_topic(self, query: str) -> list[praw.models.Subreddit]:
        if False:
            while True:
                i = 10
        'Return list of Subreddits whose topics match ``query``.\n\n        :param query: Search for subreddits relevant to the search topic.\n\n        .. note::\n\n            As of 09/01/2020, this endpoint always returns 404.\n\n        '
        result = self._reddit.get(API_PATH['subreddits_by_topic'], params={'query': query})
        return [self._reddit.subreddit(x['name']) for x in result if x.get('name')]

    def stream(self, **stream_options: str | int | dict[str, str]) -> Iterator[praw.models.Subreddit]:
        if False:
            while True:
                i = 10
        'Yield new subreddits as they are created.\n\n        Subreddits are yielded oldest first. Up to 100 historical subreddits will\n        initially be returned.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        '
        return stream_generator(self.new, **stream_options)