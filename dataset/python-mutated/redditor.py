"""Provide the Redditor class."""
from __future__ import annotations
from json import dumps
from typing import TYPE_CHECKING, Any, Generator
from ...const import API_PATH
from ...util import _deprecate_args
from ...util.cache import cachedproperty
from ..listing.mixins import RedditorListingMixin
from ..util import stream_generator
from .base import RedditBase
from .mixins import FullnameMixin, MessageableMixin
if TYPE_CHECKING:
    import praw.models

class Redditor(MessageableMixin, RedditorListingMixin, FullnameMixin, RedditBase):
    """A class representing the users of Reddit.

    .. include:: ../../typical_attributes.rst

    .. note::

        Shadowbanned accounts are treated the same as non-existent accounts, meaning
        that they will not have any attributes.

    .. note::

        Suspended/banned accounts will only return the ``name`` and ``is_suspended``
        attributes.

    =================================== ================================================
    Attribute                           Description
    =================================== ================================================
    ``comment_karma``                   The comment karma for the :class:`.Redditor`.
    ``comments``                        Provide an instance of :class:`.SubListing` for
                                        comment access.
    ``submissions``                     Provide an instance of :class:`.SubListing` for
                                        submission access.
    ``created_utc``                     Time the account was created, represented in
                                        `Unix Time`_.
    ``has_verified_email``              Whether or not the :class:`.Redditor` has
                                        verified their email.
    ``icon_img``                        The url of the Redditors' avatar.
    ``id``                              The ID of the :class:`.Redditor`.
    ``is_employee``                     Whether or not the :class:`.Redditor` is a
                                        Reddit employee.
    ``is_friend``                       Whether or not the :class:`.Redditor` is friends
                                        with the authenticated user.
    ``is_mod``                          Whether or not the :class:`.Redditor` mods any
                                        subreddits.
    ``is_gold``                         Whether or not the :class:`.Redditor` has active
                                        Reddit Premium status.
    ``is_suspended``                    Whether or not the :class:`.Redditor` is
                                        currently suspended.
    ``link_karma``                      The link karma for the :class:`.Redditor`.
    ``name``                            The Redditor's username.
    ``subreddit``                       If the :class:`.Redditor` has created a
                                        user-subreddit, provides a dictionary of
                                        additional attributes. See below.
    ``subreddit["banner_img"]``         The URL of the user-subreddit banner.
    ``subreddit["name"]``               The fullname of the user-subreddit.
    ``subreddit["over_18"]``            Whether or not the user-subreddit is NSFW.
    ``subreddit["public_description"]`` The public description of the user-subreddit.
    ``subreddit["subscribers"]``        The number of users subscribed to the
                                        user-subreddit.
    ``subreddit["title"]``              The title of the user-subreddit.
    =================================== ================================================

    .. _unix time: https://en.wikipedia.org/wiki/Unix_time

    """
    STR_FIELD = 'name'

    @classmethod
    def from_data(cls, reddit: praw.Reddit, data: dict[str, Any]) -> Redditor | None:
        if False:
            i = 10
            return i + 15
        'Return an instance of :class:`.Redditor`, or ``None`` from ``data``.'
        if data == '[deleted]':
            return None
        return cls(reddit, data)

    @cachedproperty
    def notes(self) -> praw.models.RedditorModNotes:
        if False:
            return 10
        'Provide an instance of :class:`.RedditorModNotes`.\n\n        This provides an interface for managing moderator notes for a redditor.\n\n        .. note::\n\n            The authenticated user must be a moderator of the provided subreddit(s).\n\n        For example, all the notes for u/spez in r/test can be iterated through like so:\n\n        .. code-block:: python\n\n            redditor = reddit.redditor("spez")\n\n            for note in redditor.notes.subreddits("test"):\n                print(f"{note.label}: {note.note}")\n\n        '
        from praw.models.mod_notes import RedditorModNotes
        return RedditorModNotes(self._reddit, self)

    @cachedproperty
    def stream(self) -> praw.models.reddit.redditor.RedditorStream:
        if False:
            return 10
        'Provide an instance of :class:`.RedditorStream`.\n\n        Streams can be used to indefinitely retrieve new comments made by a redditor,\n        like:\n\n        .. code-block:: python\n\n            for comment in reddit.redditor("spez").stream.comments():\n                print(comment)\n\n        Additionally, new submissions can be retrieved via the stream. In the following\n        example all submissions are fetched via the redditor u/spez:\n\n        .. code-block:: python\n\n            for submission in reddit.redditor("spez").stream.submissions():\n                print(submission)\n\n        '
        return RedditorStream(self)

    @property
    def _kind(self) -> str:
        if False:
            i = 10
            return i + 15
        "Return the class's kind."
        return self._reddit.config.kinds['redditor']

    @property
    def _path(self) -> str:
        if False:
            i = 10
            return i + 15
        return API_PATH['user'].format(user=self)

    def __init__(self, reddit: praw.Reddit, name: str | None=None, fullname: str | None=None, _data: dict[str, Any] | None=None):
        if False:
            print('Hello World!')
        'Initialize a :class:`.Redditor` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n        :param name: The name of the redditor.\n        :param fullname: The fullname of the redditor, starting with ``t2_``.\n\n        Exactly one of ``name``, ``fullname`` or ``_data`` must be provided.\n\n        '
        if (name, fullname, _data).count(None) != 2:
            msg = "Exactly one of 'name', 'fullname', or '_data' must be provided."
            raise TypeError(msg)
        if _data:
            assert isinstance(_data, dict) and 'name' in _data, 'Please file a bug with PRAW.'
        self._listing_use_sort = True
        if name:
            self.name = name
        elif fullname:
            self._fullname = fullname
        super().__init__(reddit, _data=_data, _extra_attribute_to_check='_fullname')

    def __setattr__(self, name: str, value: Any):
        if False:
            while True:
                i = 10
        'Objectify the subreddit attribute.'
        if name == 'subreddit' and value:
            from .user_subreddit import UserSubreddit
            value = UserSubreddit(reddit=self._reddit, _data=value)
        super().__setattr__(name, value)

    def _fetch(self):
        if False:
            return 10
        data = self._fetch_data()
        data = data['data']
        other = type(self)(self._reddit, _data=data)
        self.__dict__.update(other.__dict__)
        super()._fetch()

    def _fetch_info(self):
        if False:
            while True:
                i = 10
        if hasattr(self, '_fullname'):
            self.name = self._fetch_username(self._fullname)
        return ('user_about', {'user': self.name}, None)

    def _fetch_username(self, fullname: str):
        if False:
            while True:
                i = 10
        return self._reddit.get(API_PATH['user_by_fullname'], params={'ids': fullname})[fullname]['name']

    def _friend(self, *, data: dict[str, Any], method: str):
        if False:
            print('Hello World!')
        url = API_PATH['friend_v1'].format(user=self)
        self._reddit.request(data=dumps(data), method=method, path=url)

    def block(self):
        if False:
            while True:
                i = 10
        'Block the :class:`.Redditor`.\n\n        For example, to block :class:`.Redditor` u/spez:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").block()\n\n        .. note::\n\n            Blocking a trusted user will remove that user from your trusted list.\n\n        .. seealso::\n\n            :meth:`.trust`\n\n        '
        self._reddit.post(API_PATH['block_user'], params={'name': self.name})

    def distrust(self):
        if False:
            while True:
                i = 10
        'Remove the :class:`.Redditor` from your whitelist of trusted users.\n\n        For example, to remove :class:`.Redditor` u/spez from your whitelist:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").distrust()\n\n        .. seealso::\n\n            :meth:`.trust`\n\n        '
        self._reddit.post(API_PATH['remove_whitelisted'], data={'name': self.name})

    @_deprecate_args('note')
    def friend(self, *, note: str=None):
        if False:
            i = 10
            return i + 15
        'Friend the :class:`.Redditor`.\n\n        :param note: A note to save along with the relationship. Requires Reddit Premium\n            (default: ``None``).\n\n        Calling this method subsequent times will update the note.\n\n        For example, to friend u/spez:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").friend()\n\n        To add a note to the friendship (requires Reddit Premium):\n\n        .. code-block:: python\n\n            reddit.redditor("spez").friend(note="My favorite admin")\n\n        '
        self._friend(data={'note': note} if note else {}, method='PUT')

    def friend_info(self) -> praw.models.Redditor:
        if False:
            print('Hello World!')
        'Return a :class:`.Redditor` instance with specific friend-related attributes.\n\n        :returns: A :class:`.Redditor` instance with fields ``date``, ``id``, and\n            possibly ``note`` if the authenticated user has Reddit Premium.\n\n        For example, to get the friendship information of :class:`.Redditor` u/spez:\n\n        .. code-block:: python\n\n            info = reddit.redditor("spez").friend_info\n            friend_data = info.date\n\n        '
        return self._reddit.get(API_PATH['friend_v1'].format(user=self))

    @_deprecate_args('months')
    def gild(self, *, months: int=1):
        if False:
            return 10
        'Gild the :class:`.Redditor`.\n\n        :param months: Specifies the number of months to gild up to 36 (default: ``1``).\n\n        For example, to gild :class:`.Redditor` u/spez for 1 month:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").gild(months=1)\n\n        '
        if months < 1 or months > 36:
            msg = 'months must be between 1 and 36'
            raise TypeError(msg)
        self._reddit.post(API_PATH['gild_user'].format(username=self), data={'months': months})

    def moderated(self) -> list[praw.models.Subreddit]:
        if False:
            while True:
                i = 10
        'Return a list of the redditor\'s moderated subreddits.\n\n        :returns: A list of :class:`.Subreddit` objects. Return ``[]`` if the redditor\n            has no moderated subreddits.\n\n        :raises: ``prawcore.ServerError`` in certain circumstances. See the note below.\n\n        .. note::\n\n            The redditor\'s own user profile subreddit will not be returned, but other\n            user profile subreddits they moderate will be returned.\n\n        Usage:\n\n        .. code-block:: python\n\n            for subreddit in reddit.redditor("spez").moderated():\n                print(subreddit.display_name)\n                print(subreddit.title)\n\n        .. note::\n\n            A ``prawcore.ServerError`` exception may be raised if the redditor moderates\n            a large number of subreddits. If that happens, try switching to\n            :ref:`read-only mode <read_only_application>`. For example,\n\n            .. code-block:: python\n\n                reddit.read_only = True\n                for subreddit in reddit.redditor("reddit").moderated():\n                    print(str(subreddit))\n\n            It is possible that requests made in read-only mode will also raise a\n            ``prawcore.ServerError`` exception.\n\n            When used in read-only mode, this method does not retrieve information about\n            subreddits that require certain special permissions to access, e.g., private\n            subreddits and premium-only subreddits.\n\n        .. seealso::\n\n            :meth:`.User.moderator_subreddits`\n\n        '
        return self._reddit.get(API_PATH['moderated'].format(user=self)) or []

    def multireddits(self) -> list[praw.models.Multireddit]:
        if False:
            print('Hello World!')
        'Return a list of the redditor\'s public multireddits.\n\n        For example, to to get :class:`.Redditor` u/spez\'s multireddits:\n\n        .. code-block:: python\n\n            multireddits = reddit.redditor("spez").multireddits()\n\n        '
        return self._reddit.get(API_PATH['multireddit_user'].format(user=self))

    def trophies(self) -> list[praw.models.Trophy]:
        if False:
            return 10
        'Return a list of the redditor\'s trophies.\n\n        :returns: A list of :class:`.Trophy` objects. Return ``[]`` if the redditor has\n            no trophies.\n\n        :raises: :class:`.RedditAPIException` if the redditor doesn\'t exist.\n\n        Usage:\n\n        .. code-block:: python\n\n            for trophy in reddit.redditor("spez").trophies():\n                print(trophy.name)\n                print(trophy.description)\n\n        '
        return list(self._reddit.get(API_PATH['trophies'].format(user=self)))

    def trust(self):
        if False:
            return 10
        'Add the :class:`.Redditor` to your whitelist of trusted users.\n\n        Trusted users will always be able to send you PMs.\n\n        Example usage:\n\n        .. code-block:: python\n\n            reddit.redditor("AaronSw").trust()\n\n        Use the ``accept_pms`` parameter of :meth:`.Preferences.update` to toggle your\n        ``accept_pms`` setting between ``"everyone"`` and ``"whitelisted"``. For\n        example:\n\n        .. code-block:: python\n\n            # Accept private messages from everyone:\n            reddit.user.preferences.update(accept_pms="everyone")\n            # Only accept private messages from trusted users:\n            reddit.user.preferences.update(accept_pms="whitelisted")\n\n        You may trust a user even if your ``accept_pms`` setting is switched to\n        ``"everyone"``.\n\n        .. note::\n\n            You are allowed to have a user on your blocked list and your friends list at\n            the same time. However, you cannot trust a user who is on your blocked list.\n\n        .. seealso::\n\n            - :meth:`.distrust`\n            - :meth:`.Preferences.update`\n            - :meth:`.trusted`\n\n        '
        self._reddit.post(API_PATH['add_whitelisted'], data={'name': self.name})

    def unblock(self):
        if False:
            print('Hello World!')
        'Unblock the :class:`.Redditor`.\n\n        For example, to unblock :class:`.Redditor` u/spez:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").unblock()\n\n        '
        data = {'container': self._reddit.user.me().fullname, 'name': str(self), 'type': 'enemy'}
        url = API_PATH['unfriend'].format(subreddit='all')
        self._reddit.post(url, data=data)

    def unfriend(self):
        if False:
            return 10
        'Unfriend the :class:`.Redditor`.\n\n        For example, to unfriend :class:`.Redditor` u/spez:\n\n        .. code-block:: python\n\n            reddit.redditor("spez").unfriend()\n\n        '
        self._friend(data={'id': str(self)}, method='DELETE')

class RedditorStream:
    """Provides submission and comment streams."""

    def __init__(self, redditor: praw.models.Redditor):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.RedditorStream` instance.\n\n        :param redditor: The redditor associated with the streams.\n\n        '
        self.redditor = redditor

    def comments(self, **stream_options: str | int | dict[str, str]) -> Generator[praw.models.Comment, None, None]:
        if False:
            while True:
                i = 10
        'Yield new comments as they become available.\n\n        Comments are yielded oldest first. Up to 100 historical comments will initially\n        be returned.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        For example, to retrieve all new comments made by redditor u/spez, try:\n\n        .. code-block:: python\n\n            for comment in reddit.redditor("spez").stream.comments():\n                print(comment)\n\n        '
        return stream_generator(self.redditor.comments.new, **stream_options)

    def submissions(self, **stream_options: str | int | dict[str, str]) -> Generator[praw.models.Submission, None, None]:
        if False:
            print('Hello World!')
        'Yield new submissions as they become available.\n\n        Submissions are yielded oldest first. Up to 100 historical submissions will\n        initially be returned.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        For example, to retrieve all new submissions made by redditor u/spez, try:\n\n        .. code-block:: python\n\n            for submission in reddit.redditor("spez").stream.submissions():\n                print(submission)\n\n        '
        return stream_generator(self.redditor.submissions.new, **stream_options)