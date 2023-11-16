"""Provide the WikiPage class."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator, Iterator
from ...const import API_PATH
from ...util import _deprecate_args
from ...util.cache import cachedproperty
from ..listing.generator import ListingGenerator
from .base import RedditBase
from .redditor import Redditor
if TYPE_CHECKING:
    import praw.models

class WikiPageModeration:
    """Provides a set of moderation functions for a :class:`.WikiPage`.

    For example, to add u/spez as an editor on the wikipage ``"praw_test"`` try:

    .. code-block:: python

        reddit.subreddit("test").wiki["praw_test"].mod.add("spez")

    """

    def __init__(self, wikipage: WikiPage):
        if False:
            while True:
                i = 10
        'Initialize a :class:`.WikiPageModeration` instance.\n\n        :param wikipage: The wikipage to moderate.\n\n        '
        self.wikipage = wikipage

    def add(self, redditor: praw.models.Redditor):
        if False:
            print('Hello World!')
        'Add an editor to this :class:`.WikiPage`.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n\n        To add u/spez as an editor on the wikipage ``"praw_test"`` try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").wiki["praw_test"].mod.add("spez")\n\n        '
        data = {'page': self.wikipage.name, 'username': str(redditor)}
        url = API_PATH['wiki_page_editor'].format(subreddit=self.wikipage.subreddit, method='add')
        self.wikipage._reddit.post(url, data=data)

    def remove(self, redditor: praw.models.Redditor):
        if False:
            i = 10
            return i + 15
        'Remove an editor from this :class:`.WikiPage`.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n\n        To remove u/spez as an editor on the wikipage ``"praw_test"`` try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").wiki["praw_test"].mod.remove("spez")\n\n        '
        data = {'page': self.wikipage.name, 'username': str(redditor)}
        url = API_PATH['wiki_page_editor'].format(subreddit=self.wikipage.subreddit, method='del')
        self.wikipage._reddit.post(url, data=data)

    def revert(self):
        if False:
            while True:
                i = 10
        'Revert a wikipage back to a specific revision.\n\n        To revert the page ``"praw_test"`` in r/test to revision ``"1234abc"``, try\n\n        .. code-block:: python\n\n            reddit.subreddit("test").wiki["praw_test"].revision("1234abc").mod.revert()\n\n        .. note::\n\n            When you attempt to revert the page ``config/stylesheet``, Reddit checks to\n            see if the revision being reverted to passes the CSS filter. If the check\n            fails, then the revision attempt will also fail, and a\n            ``prawcore.Forbidden`` exception will be raised. For example, you can\'t\n            revert to a revision that contains a link to ``url(%%PRAW%%)`` if there is\n            no image named ``PRAW`` on the current stylesheet.\n\n            Here is an example of how to look for this type of error:\n\n            .. code-block:: python\n\n                from prawcore.exceptions import Forbidden\n\n                try:\n                    reddit.subreddit("test").wiki["config/stylesheet"].revision("1234abc").mod.revert()\n                except Forbidden as exception:\n                    try:\n                        exception.response.json()\n                    except ValueError:\n                        exception.response.text\n\n            If the error occurs, the output will look something like\n\n            .. code-block:: python\n\n                {"reason": "INVALID_CSS", "message": "Forbidden", "explanation": "%(css_error)s"}\n\n        '
        self.wikipage._reddit.post(API_PATH['wiki_revert'].format(subreddit=self.wikipage.subreddit), data={'page': self.wikipage.name, 'revision': self.wikipage._revision})

    def settings(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Return the settings for this :class:`.WikiPage`.'
        url = API_PATH['wiki_page_settings'].format(subreddit=self.wikipage.subreddit, page=self.wikipage.name)
        return self.wikipage._reddit.get(url)['data']

    @_deprecate_args('listed', 'permlevel')
    def update(self, *, listed: bool, permlevel: int, **other_settings: Any) -> dict[str, Any]:
        if False:
            return 10
        'Update the settings for this :class:`.WikiPage`.\n\n        :param listed: Show this page on page list.\n        :param permlevel: Who can edit this page? ``0`` use subreddit wiki permissions,\n            ``1`` only approved wiki contributors for this page may edit (see\n            :meth:`.WikiPageModeration.add`), ``2`` only mods may edit and view.\n        :param other_settings: Additional keyword arguments to pass.\n\n        :returns: The updated WikiPage settings.\n\n        To set the wikipage ``"praw_test"`` in r/test to mod only and disable it from\n        showing in the page list, try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").wiki["praw_test"].mod.update(listed=False, permlevel=2)\n\n        '
        other_settings.update({'listed': listed, 'permlevel': permlevel})
        url = API_PATH['wiki_page_settings'].format(subreddit=self.wikipage.subreddit, page=self.wikipage.name)
        return self.wikipage._reddit.post(url, data=other_settings)['data']

class WikiPage(RedditBase):
    """An individual :class:`.WikiPage` object.

    .. include:: ../../typical_attributes.rst

    ================= =================================================================
    Attribute         Description
    ================= =================================================================
    ``content_html``  The contents of the wiki page, as HTML.
    ``content_md``    The contents of the wiki page, as Markdown.
    ``may_revise``    A ``bool`` representing whether or not the authenticated user may
                      edit the wiki page.
    ``name``          The name of the wiki page.
    ``revision_by``   The :class:`.Redditor` who authored this revision of the wiki
                      page.
    ``revision_date`` The time of this revision, in `Unix Time`_.
    ``subreddit``     The :class:`.Subreddit` this wiki page belongs to.
    ================= =================================================================

    .. _unix time: https://en.wikipedia.org/wiki/Unix_time

    """
    __hash__ = RedditBase.__hash__

    @staticmethod
    def _revision_generator(*, generator_kwargs: dict[str, Any], subreddit: praw.models.Subreddit, url: str) -> Generator[dict[str, Redditor | WikiPage | str | int | bool | None], None, None]:
        if False:
            return 10
        for revision in ListingGenerator(subreddit._reddit, url, **generator_kwargs):
            if revision['author'] is not None:
                revision['author'] = Redditor(subreddit._reddit, _data=revision['author']['data'])
            revision['page'] = WikiPage(subreddit._reddit, subreddit, revision['page'], revision['id'])
            yield revision

    @cachedproperty
    def mod(self) -> WikiPageModeration:
        if False:
            while True:
                i = 10
        'Provide an instance of :class:`.WikiPageModeration`.\n\n        For example, to add u/spez as an editor on the wikipage ``"praw_test"`` try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").wiki["praw_test"].mod.add("spez")\n\n        '
        return WikiPageModeration(self)

    def __init__(self, reddit: praw.Reddit, subreddit: praw.models.Subreddit, name: str, revision: str | None=None, _data: dict[str, Any] | None=None):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.WikiPage` instance.\n\n        :param revision: A specific revision ID to fetch. By default, fetches the most\n            recent revision.\n\n        '
        self.name = name
        self._revision = revision
        self.subreddit = subreddit
        super().__init__(reddit, _data=_data, _str_field=False)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return an object initialization representation of the instance.'
        return f'{self.__class__.__name__}(subreddit={self.subreddit!r}, name={self.name!r})'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return a string representation of the instance.'
        return f'{self.subreddit}/{self.name}'

    def _fetch(self):
        if False:
            while True:
                i = 10
        data = self._fetch_data()
        data = data['data']
        if data['revision_by'] is not None:
            data['revision_by'] = Redditor(self._reddit, _data=data['revision_by']['data'])
        self.__dict__.update(data)
        super()._fetch()

    def _fetch_info(self):
        if False:
            while True:
                i = 10
        return ('wiki_page', {'subreddit': self.subreddit, 'page': self.name}, {'v': self._revision} if self._revision else None)

    def discussions(self, **generator_kwargs: Any) -> Iterator[praw.models.Submission]:
        if False:
            while True:
                i = 10
        'Return a :class:`.ListingGenerator` for discussions of a wiki page.\n\n        Discussions are site-wide links to a wiki page.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To view the titles of discussions of the page ``"praw_test"`` in r/test, try:\n\n        .. code-block:: python\n\n            for submission in reddit.subreddit("test").wiki["praw_test"].discussions():\n                print(submission.title)\n\n        '
        return ListingGenerator(self._reddit, API_PATH['wiki_discussions'].format(subreddit=self.subreddit, page=self.name), **generator_kwargs)

    @_deprecate_args('content', 'reason')
    def edit(self, *, content: str, reason: str | None=None, **other_settings: Any):
        if False:
            return 10
        'Edit this wiki page\'s contents.\n\n        :param content: The updated Markdown content of the page.\n        :param reason: The reason for the revision.\n        :param other_settings: Additional keyword arguments to pass.\n\n        For example, to replace the first wiki page of r/test with the phrase ``"test\n        wiki page"``:\n\n        .. code-block:: python\n\n            page = next(iter(reddit.subreddit("test").wiki))\n            page.edit(content="test wiki page")\n\n        '
        other_settings.update({'content': content, 'page': self.name, 'reason': reason})
        self._reddit.post(API_PATH['wiki_edit'].format(subreddit=self.subreddit), data=other_settings)

    def revision(self, revision: str) -> WikiPage:
        if False:
            for i in range(10):
                print('nop')
        'Return a specific version of this page by revision ID.\n\n        To view revision ``"1234abc"`` of ``"praw_test"`` in r/test:\n\n        .. code-block:: python\n\n            page = reddit.subreddit("test").wiki["praw_test"].revision("1234abc")\n\n        '
        return WikiPage(self.subreddit._reddit, self.subreddit, self.name, revision)

    def revisions(self, **generator_kwargs: str | int | dict[str, str]) -> Generator[WikiPage, None, None]:
        if False:
            while True:
                i = 10
        'Return a :class:`.ListingGenerator` for page revisions.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To view the wiki revisions for ``"praw_test"`` in r/test try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("test").wiki["praw_test"].revisions():\n                print(item)\n\n        To get :class:`.WikiPage` objects for each revision:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("test").wiki["praw_test"].revisions():\n                print(item["page"])\n\n        '
        url = API_PATH['wiki_page_revisions'].format(subreddit=self.subreddit, page=self.name)
        return self._revision_generator(generator_kwargs=generator_kwargs, subreddit=self.subreddit, url=url)