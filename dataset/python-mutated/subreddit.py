"""Provide the Subreddit class."""
from __future__ import annotations
import contextlib
from copy import deepcopy
from csv import writer
from io import StringIO
from json import dumps, loads
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterator
from urllib.parse import urljoin
from warnings import warn
from xml.etree.ElementTree import XML
import websocket
from prawcore import Redirect
from prawcore.exceptions import ServerError
from requests.exceptions import HTTPError
from ...const import API_PATH, JPEG_HEADER
from ...exceptions import ClientException, InvalidFlairTemplateID, MediaPostFailed, RedditAPIException, TooLargeMediaException, WebSocketException
from ...util import _deprecate_args, cachedproperty
from ..listing.generator import ListingGenerator
from ..listing.mixins import SubredditListingMixin
from ..util import permissions_string, stream_generator
from .base import RedditBase
from .emoji import SubredditEmoji
from .mixins import FullnameMixin, MessageableMixin
from .modmail import ModmailConversation
from .removal_reasons import SubredditRemovalReasons
from .rules import SubredditRules
from .widgets import SubredditWidgets, WidgetEncoder
from .wikipage import WikiPage
if TYPE_CHECKING:
    from requests import Response
    import praw.models

class Modmail:
    """Provides modmail functions for a :class:`.Subreddit`.

    For example, to send a new modmail from r/test to u/spez with the subject ``"test"``
    along with a message body of ``"hello"``:

    .. code-block:: python

        reddit.subreddit("test").modmail.create(subject="test", body="hello", recipient="spez")

    """

    def __call__(self, id: str | None=None, mark_read: bool=False) -> ModmailConversation:
        if False:
            for i in range(10):
                print('nop')
        'Return an individual conversation.\n\n        :param id: A reddit base36 conversation ID, e.g., ``"2gmz"``.\n        :param mark_read: If ``True``, conversation is marked as read (default:\n            ``False``).\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz", mark_read=True)\n\n        To print all messages from a conversation as Markdown source:\n\n        .. code-block:: python\n\n            conversation = reddit.subreddit("test").modmail("2gmz", mark_read=True)\n            for message in conversation.messages:\n                print(message.body_markdown)\n\n        ``ModmailConversation.user`` is a special instance of :class:`.Redditor` with\n        extra attributes describing the non-moderator user\'s recent posts, comments, and\n        modmail messages within the subreddit, as well as information on active bans and\n        mutes. This attribute does not exist on internal moderator discussions.\n\n        For example, to print the user\'s ban status:\n\n        .. code-block:: python\n\n            conversation = reddit.subreddit("test").modmail("2gmz", mark_read=True)\n            print(conversation.user.ban_status)\n\n        To print a list of recent submissions by the user:\n\n        .. code-block:: python\n\n            conversation = reddit.subreddit("test").modmail("2gmz", mark_read=True)\n            print(conversation.user.recent_posts)\n\n        '
        return ModmailConversation(self.subreddit._reddit, id=id, mark_read=mark_read)

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            print('Hello World!')
        'Initialize a :class:`.Modmail` instance.'
        self.subreddit = subreddit

    def _build_subreddit_list(self, other_subreddits: list[praw.models.Subreddit] | None):
        if False:
            return 10
        'Return a comma-separated list of subreddit display names.'
        subreddits = [self.subreddit] + (other_subreddits or [])
        return ','.join((str(subreddit) for subreddit in subreddits))

    @_deprecate_args('other_subreddits', 'state')
    def bulk_read(self, *, other_subreddits: list[praw.models.Subreddit | str] | None=None, state: str | None=None) -> list[ModmailConversation]:
        if False:
            for i in range(10):
                print('nop')
        'Mark conversations for subreddit(s) as read.\n\n        .. note::\n\n            Due to server-side restrictions, r/all is not a valid subreddit for this\n            method. Instead, use :meth:`~.Modmail.subreddits` to get a list of\n            subreddits using the new modmail.\n\n        :param other_subreddits: A list of :class:`.Subreddit` instances for which to\n            mark conversations (default: ``None``).\n        :param state: Can be one of: ``"all"``, ``"archived"``, or ``"highlighted"``,\n            ``"inprogress"``, ``"join_requests"``, ``"mod"``, ``"new"``,\n            ``"notifications"``, or ``"appeals"`` (default: ``"all"``). ``"all"`` does\n            not include internal, archived, or appeals conversations.\n\n        :returns: A list of :class:`.ModmailConversation` instances that were marked\n            read.\n\n        For example, to mark all notifications for a subreddit as read:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            subreddit.modmail.bulk_read(state="notifications")\n\n        '
        params = {'entity': self._build_subreddit_list(other_subreddits)}
        if state:
            params['state'] = state
        response = self.subreddit._reddit.post(API_PATH['modmail_bulk_read'], params=params)
        return [self(conversation_id) for conversation_id in response['conversation_ids']]

    @_deprecate_args('after', 'other_subreddits', 'sort', 'state')
    def conversations(self, *, after: str | None=None, other_subreddits: list[praw.models.Subreddit] | None=None, sort: str | None=None, state: str | None=None, **generator_kwargs: Any) -> Iterator[ModmailConversation]:
        if False:
            i = 10
            return i + 15
        'Generate :class:`.ModmailConversation` objects for subreddit(s).\n\n        :param after: A base36 modmail conversation id. When provided, the listing\n            begins after this conversation (default: ``None``).\n\n            .. deprecated:: 7.4.0\n\n                This parameter is deprecated and will be removed in PRAW 8.0. This\n                method will automatically fetch the next batch. Please pass it in the\n                ``params`` argument like so:\n\n                .. code-block:: python\n\n                    for convo in subreddit.modmail.conversations(params={"after": "qovbn"}):\n                        # process conversation\n                        ...\n\n        :param other_subreddits: A list of :class:`.Subreddit` instances for which to\n            fetch conversations (default: ``None``).\n        :param sort: Can be one of: ``"mod"``, ``"recent"``, ``"unread"``, or ``"user"``\n            (default: ``"recent"``).\n        :param state: Can be one of: ``"all"``, ``"archived"``, ``"highlighted"``,\n            ``"inprogress"``, ``"join_requests"``, ``"mod"``, ``"new"``,\n            ``"notifications"``, or ``"appeals"`` (default: ``"all"``). ``"all"`` does\n            not include internal, archived, or appeals conversations.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        For example:\n\n        .. code-block:: python\n\n            conversations = reddit.subreddit("all").modmail.conversations(state="mod")\n\n        '
        params = {}
        if after:
            warn("The 'after' argument is deprecated and should be moved to the 'params' dictionary argument.", category=DeprecationWarning, stacklevel=3)
            params['after'] = after
        if self.subreddit != 'all':
            params['entity'] = self._build_subreddit_list(other_subreddits)
        Subreddit._safely_add_arguments(arguments=generator_kwargs, key='params', sort=sort, state=state, **params)
        return ListingGenerator(self.subreddit._reddit, API_PATH['modmail_conversations'], **generator_kwargs)

    @_deprecate_args('subject', 'body', 'recipient', 'author_hidden')
    def create(self, *, author_hidden: bool=False, body: str, recipient: str | praw.models.Redditor, subject: str) -> ModmailConversation:
        if False:
            for i in range(10):
                print('nop')
        'Create a new :class:`.ModmailConversation`.\n\n        :param author_hidden: When ``True``, author is hidden from non-moderators\n            (default: ``False``).\n        :param body: The message body. Cannot be empty.\n        :param recipient: The recipient; a username or an instance of\n            :class:`.Redditor`.\n        :param subject: The message subject. Cannot be empty.\n\n        :returns: A :class:`.ModmailConversation` object for the newly created\n            conversation.\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            redditor = reddit.redditor("bboe")\n            subreddit.modmail.create(subject="Subject", body="Body", recipient=redditor)\n\n        '
        data = {'body': body, 'isAuthorHidden': author_hidden, 'srName': self.subreddit, 'subject': subject, 'to': recipient}
        return self.subreddit._reddit.post(API_PATH['modmail_conversations'], data=data)

    def subreddits(self) -> Generator[praw.models.Subreddit, None, None]:
        if False:
            i = 10
            return i + 15
        'Yield subreddits using the new modmail that the user moderates.\n\n        For example:\n\n        .. code-block:: python\n\n            subreddits = reddit.subreddit("all").modmail.subreddits()\n\n        '
        response = self.subreddit._reddit.get(API_PATH['modmail_subreddits'])
        for value in response['subreddits'].values():
            subreddit = self.subreddit._reddit.subreddit(value['display_name'])
            subreddit.last_updated = value['lastUpdated']
            yield subreddit

    def unread_count(self) -> dict[str, int]:
        if False:
            for i in range(10):
                print('nop')
        'Return unread conversation count by conversation state.\n\n        At time of writing, possible states are: ``"archived"``, ``"highlighted"``,\n        ``"inprogress"``, ``"join_requests"``, ``"mod"``, ``"new"``,\n        ``"notifications"``, or ``"appeals"``.\n\n        :returns: A dict mapping conversation states to unread counts.\n\n        For example, to print the count of unread moderator discussions:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            unread_counts = subreddit.modmail.unread_count()\n            print(unread_counts["mod"])\n\n        '
        return self.subreddit._reddit.get(API_PATH['modmail_unread_count'])

class SubredditFilters:
    """Provide functions to interact with the special :class:`.Subreddit`'s filters.

    Members of this class should be utilized via :meth:`.Subreddit.filters`. For
    example, to add a filter, run:

    .. code-block:: python

        reddit.subreddit("all").filters.add("test")

    """

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            while True:
                i = 10
        'Initialize a :class:`.SubredditFilters` instance.\n\n        :param subreddit: The special subreddit whose filters to work with.\n\n        As of this writing filters can only be used with the special subreddits ``all``\n        and ``mod``.\n\n        '
        self.subreddit = subreddit

    def __iter__(self) -> Generator[praw.models.Subreddit, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Iterate through the special :class:`.Subreddit`\'s filters.\n\n        This method should be invoked as:\n\n        .. code-block:: python\n\n            for subreddit in reddit.subreddit("test").filters:\n                ...\n\n        '
        url = API_PATH['subreddit_filter_list'].format(special=self.subreddit, user=self.subreddit._reddit.user.me())
        params = {'unique': self.subreddit._reddit._next_unique}
        response_data = self.subreddit._reddit.get(url, params=params)
        yield from response_data.subreddits

    def add(self, subreddit: praw.models.Subreddit | str):
        if False:
            for i in range(10):
                print('nop')
        'Add ``subreddit`` to the list of filtered subreddits.\n\n        :param subreddit: The subreddit to add to the filter list.\n\n        Items from subreddits added to the filtered list will no longer be included when\n        obtaining listings for r/all.\n\n        Alternatively, you can filter a subreddit temporarily from a special listing in\n        a manner like so:\n\n        .. code-block:: python\n\n            reddit.subreddit("all-redditdev-learnpython")\n\n        :raises: ``prawcore.NotFound`` when calling on a non-special subreddit.\n\n        '
        url = API_PATH['subreddit_filter'].format(special=self.subreddit, user=self.subreddit._reddit.user.me(), subreddit=subreddit)
        self.subreddit._reddit.put(url, data={'model': dumps({'name': str(subreddit)})})

    def remove(self, subreddit: praw.models.Subreddit | str):
        if False:
            print('Hello World!')
        'Remove ``subreddit`` from the list of filtered subreddits.\n\n        :param subreddit: The subreddit to remove from the filter list.\n\n        :raises: ``prawcore.NotFound`` when calling on a non-special subreddit.\n\n        '
        url = API_PATH['subreddit_filter'].format(special=self.subreddit, user=self.subreddit._reddit.user.me(), subreddit=str(subreddit))
        self.subreddit._reddit.delete(url)

class SubredditFlair:
    """Provide a set of functions to interact with a :class:`.Subreddit`'s flair."""

    @cachedproperty
    def link_templates(self) -> praw.models.reddit.subreddit.SubredditLinkFlairTemplates:
        if False:
            i = 10
            return i + 15
        'Provide an instance of :class:`.SubredditLinkFlairTemplates`.\n\n        Use this attribute for interacting with a :class:`.Subreddit`\'s link flair\n        templates. For example to list all the link flair templates for a subreddit\n        which you have the ``flair`` moderator permission on try:\n\n        .. code-block:: python\n\n            for template in reddit.subreddit("test").flair.link_templates:\n                print(template)\n\n        '
        return SubredditLinkFlairTemplates(self.subreddit)

    @cachedproperty
    def templates(self) -> praw.models.reddit.subreddit.SubredditRedditorFlairTemplates:
        if False:
            i = 10
            return i + 15
        'Provide an instance of :class:`.SubredditRedditorFlairTemplates`.\n\n        Use this attribute for interacting with a :class:`.Subreddit`\'s flair templates.\n        For example to list all the flair templates for a subreddit which you have the\n        ``flair`` moderator permission on try:\n\n        .. code-block:: python\n\n            for template in reddit.subreddit("test").flair.templates:\n                print(template)\n\n        '
        return SubredditRedditorFlairTemplates(self.subreddit)

    def __call__(self, redditor: praw.models.Redditor | str | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Redditor]:
        if False:
            for i in range(10):
                print('nop')
        'Return a :class:`.ListingGenerator` for Redditors and their flairs.\n\n        :param redditor: When provided, yield at most a single :class:`.Redditor`\n            instance (default: ``None``).\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        Usage:\n\n        .. code-block:: python\n\n            for flair in reddit.subreddit("test").flair(limit=None):\n                print(flair)\n\n        '
        Subreddit._safely_add_arguments(arguments=generator_kwargs, key='params', name=redditor)
        generator_kwargs.setdefault('limit', None)
        url = API_PATH['flairlist'].format(subreddit=self.subreddit)
        return ListingGenerator(self.subreddit._reddit, url, **generator_kwargs)

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            print('Hello World!')
        'Initialize a :class:`.SubredditFlair` instance.\n\n        :param subreddit: The subreddit whose flair to work with.\n\n        '
        self.subreddit = subreddit

    @_deprecate_args('position', 'self_assign', 'link_position', 'link_self_assign')
    def configure(self, *, link_position: str='left', link_self_assign: bool=False, position: str='right', self_assign: bool=False, **settings: Any):
        if False:
            print('Hello World!')
        'Update the :class:`.Subreddit`\'s flair configuration.\n\n        :param link_position: One of ``"left"``, ``"right"``, or ``False`` to disable\n            (default: ``"left"``).\n        :param link_self_assign: Permit self assignment of link flair (default:\n            ``False``).\n        :param position: One of ``"left"``, ``"right"``, or ``False`` to disable\n            (default: ``"right"``).\n        :param self_assign: Permit self assignment of user flair (default: ``False``).\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            subreddit.flair.configure(link_position="right", self_assign=True)\n\n        Additional keyword arguments can be provided to handle new settings as Reddit\n        introduces them.\n\n        '
        data = {'flair_enabled': bool(position), 'flair_position': position or 'right', 'flair_self_assign_enabled': self_assign, 'link_flair_position': link_position or '', 'link_flair_self_assign_enabled': link_self_assign}
        data.update(settings)
        url = API_PATH['flairconfig'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

    def delete(self, redditor: praw.models.Redditor | str):
        if False:
            return 10
        'Delete flair for a :class:`.Redditor`.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n\n        .. seealso::\n\n            :meth:`~.SubredditFlair.update` to delete the flair of many redditors at\n            once.\n\n        '
        url = API_PATH['deleteflair'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data={'name': str(redditor)})

    def delete_all(self) -> list[dict[str, str | bool | dict[str, str]]]:
        if False:
            print('Hello World!')
        'Delete all :class:`.Redditor` flair in the :class:`.Subreddit`.\n\n        :returns: List of dictionaries indicating the success or failure of each delete.\n\n        '
        return self.update((x['user'] for x in self()))

    @_deprecate_args('redditor', 'text', 'css_class', 'flair_template_id')
    def set(self, redditor: praw.models.Redditor | str, *, css_class: str='', flair_template_id: str | None=None, text: str=''):
        if False:
            i = 10
            return i + 15
        'Set flair for a :class:`.Redditor`.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n        :param text: The flair text to associate with the :class:`.Redditor` or\n            :class:`.Submission` (default: ``""``).\n        :param css_class: The css class to associate with the flair html (default:\n            ``""``). Use either this or ``flair_template_id``.\n        :param flair_template_id: The ID of the flair template to be used (default:\n            ``None``). Use either this or ``css_class``.\n\n        This method can only be used by an authenticated user who is a moderator of the\n        associated :class:`.Subreddit`.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").flair.set("bboe", text="PRAW author", css_class="mods")\n            template = "6bd28436-1aa7-11e9-9902-0e05ab0fad46"\n            reddit.subreddit("test").flair.set(\n                "spez", text="Reddit CEO", flair_template_id=template\n            )\n\n        '
        if css_class and flair_template_id is not None:
            msg = "Parameter 'css_class' cannot be used in conjunction with 'flair_template_id'."
            raise TypeError(msg)
        data = {'name': str(redditor), 'text': text}
        if flair_template_id is not None:
            data['flair_template_id'] = flair_template_id
            url = API_PATH['select_flair'].format(subreddit=self.subreddit)
        else:
            data['css_class'] = css_class
            url = API_PATH['flair'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

    @_deprecate_args('flair_list', 'text', 'css_class')
    def update(self, flair_list: Iterator[str | praw.models.Redditor | dict[str, str | praw.models.Redditor]], *, text: str='', css_class: str='') -> list[dict[str, str | bool | dict[str, str]]]:
        if False:
            print('Hello World!')
        'Set or clear the flair for many redditors at once.\n\n        :param flair_list: Each item in this list should be either:\n\n            - The name of a redditor.\n            - An instance of :class:`.Redditor`.\n            - A dictionary mapping keys ``"user"``, ``"flair_text"``, and\n              ``"flair_css_class"`` to their respective values. The ``"user"`` key\n              should map to a redditor, as described above. When a dictionary isn\'t\n              provided, or the dictionary is missing either ``"flair_text"`` or\n              ``"flair_css_class"`` keys, the default values will come from the other\n              arguments.\n        :param css_class: The css class to use when not explicitly provided in\n            ``flair_list`` (default: ``""``).\n        :param text: The flair text to use when not explicitly provided in\n            ``flair_list`` (default: ``""``).\n\n        :returns: List of dictionaries indicating the success or failure of each update.\n\n        For example, to clear the flair text, and set the ``"praw"`` flair css class on\n        a few users try:\n\n        .. code-block:: python\n\n            subreddit.flair.update(["bboe", "spez", "spladug"], css_class="praw")\n\n        '
        temp_lines = StringIO()
        for item in flair_list:
            if isinstance(item, dict):
                writer(temp_lines).writerow([str(item['user']), item.get('flair_text', text), item.get('flair_css_class', css_class)])
            else:
                writer(temp_lines).writerow([str(item), text, css_class])
        lines = temp_lines.getvalue().splitlines()
        temp_lines.close()
        response = []
        url = API_PATH['flaircsv'].format(subreddit=self.subreddit)
        while lines:
            data = {'flair_csv': '\n'.join(lines[:100])}
            response.extend(self.subreddit._reddit.post(url, data=data))
            lines = lines[100:]
        return response

class SubredditFlairTemplates:
    """Provide functions to interact with a :class:`.Subreddit`'s flair templates."""

    @staticmethod
    def flair_type(is_link: bool) -> str:
        if False:
            print('Hello World!')
        'Return ``"LINK_FLAIR"`` or ``"USER_FLAIR"`` depending on ``is_link`` value.'
        return 'LINK_FLAIR' if is_link else 'USER_FLAIR'

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a :class:`.SubredditFlairTemplates` instance.\n\n        :param subreddit: The subreddit whose flair templates to work with.\n\n        .. note::\n\n            This class should not be initialized directly. Instead, obtain an instance\n            via: ``reddit.subreddit("test").flair.templates`` or\n            ``reddit.subreddit("test").flair.link_templates``.\n\n        '
        self.subreddit = subreddit

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Abstract method to return flair templates.'
        raise NotImplementedError

    def _add(self, *, allowable_content: str | None=None, background_color: str | None=None, css_class: str='', is_link: bool | None=None, max_emojis: int | None=None, mod_only: bool | None=None, text: str, text_color: str | None=None, text_editable: bool=False):
        if False:
            while True:
                i = 10
        url = API_PATH['flairtemplate_v2'].format(subreddit=self.subreddit)
        data = {'allowable_content': allowable_content, 'background_color': background_color, 'css_class': css_class, 'flair_type': self.flair_type(is_link), 'max_emojis': max_emojis, 'mod_only': bool(mod_only), 'text': text, 'text_color': text_color, 'text_editable': bool(text_editable)}
        self.subreddit._reddit.post(url, data=data)

    def _clear(self, *, is_link: bool | None=None):
        if False:
            i = 10
            return i + 15
        url = API_PATH['flairtemplateclear'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data={'flair_type': self.flair_type(is_link)})

    def _reorder(self, flair_list: list, *, is_link: bool | None=None):
        if False:
            print('Hello World!')
        url = API_PATH['flairtemplatereorder'].format(subreddit=self.subreddit)
        self.subreddit._reddit.patch(url, params={'flair_type': self.flair_type(is_link), 'subreddit': self.subreddit.display_name}, json=flair_list)

    def delete(self, template_id: str):
        if False:
            print('Hello World!')
        'Remove a flair template provided by ``template_id``.\n\n        For example, to delete the first :class:`.Redditor` flair template listed, try:\n\n        .. code-block:: python\n\n            template_info = list(subreddit.flair.templates)[0]\n            subreddit.flair.templates.delete(template_info["id"])\n\n        '
        url = API_PATH['flairtemplatedelete'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data={'flair_template_id': template_id})

    @_deprecate_args('template_id', 'text', 'css_class', 'text_editable', 'background_color', 'text_color', 'mod_only', 'allowable_content', 'max_emojis', 'fetch')
    def update(self, template_id: str, *, allowable_content: str | None=None, background_color: str | None=None, css_class: str | None=None, fetch: bool=True, max_emojis: int | None=None, mod_only: bool | None=None, text: str | None=None, text_color: str | None=None, text_editable: bool | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Update the flair template provided by ``template_id``.\n\n        :param template_id: The flair template to update. If not valid then an exception\n            will be thrown.\n        :param allowable_content: If specified, most be one of ``"all"``, ``"emoji"``,\n            or ``"text"`` to restrict content to that type. If set to ``"emoji"`` then\n            the ``"text"`` param must be a valid emoji string, for example,\n            ``":snoo:"``.\n        :param background_color: The flair template\'s new background color, as a hex\n            color.\n        :param css_class: The flair template\'s new css_class (default: ``""``).\n        :param fetch: Whether PRAW will fetch existing information on the existing flair\n            before updating (default: ``True``).\n        :param max_emojis: Maximum emojis in the flair (Reddit defaults this value to\n            ``10``).\n        :param mod_only: Indicate if the flair can only be used by moderators.\n        :param text: The flair template\'s new text.\n        :param text_color: The flair template\'s new text color, either ``"light"`` or\n            ``"dark"``.\n        :param text_editable: Indicate if the flair text can be modified for each\n            redditor that sets it (default: ``False``).\n\n        .. warning::\n\n            If parameter ``fetch`` is set to ``False``, all parameters not provided will\n            be reset to their default (``None`` or ``False``) values.\n\n        For example, to make a user flair template text editable, try:\n\n        .. code-block:: python\n\n            template_info = list(subreddit.flair.templates)[0]\n            subreddit.flair.templates.update(\n                template_info["id"],\n                text=template_info["flair_text"],\n                text_editable=True,\n            )\n\n        '
        url = API_PATH['flairtemplate_v2'].format(subreddit=self.subreddit)
        data = {'allowable_content': allowable_content, 'background_color': background_color, 'css_class': css_class, 'flair_template_id': template_id, 'max_emojis': max_emojis, 'mod_only': mod_only, 'text': text, 'text_color': text_color, 'text_editable': text_editable}
        if fetch:
            _existing_data = [template for template in iter(self) if template['id'] == template_id]
            if len(_existing_data) != 1:
                raise InvalidFlairTemplateID(template_id)
            existing_data = _existing_data[0]
            for (key, value) in existing_data.items():
                if data.get(key) is None:
                    data[key] = value
        self.subreddit._reddit.post(url, data=data)

class SubredditModeration:
    """Provides a set of moderation functions to a :class:`.Subreddit`.

    For example, to accept a moderation invite from r/test:

    .. code-block:: python

        reddit.subreddit("test").mod.accept_invite()

    """

    @staticmethod
    def _handle_only(*, generator_kwargs: dict[str, Any], only: str | None):
        if False:
            for i in range(10):
                print('nop')
        if only is not None:
            if only == 'submissions':
                only = 'links'
            RedditBase._safely_add_arguments(arguments=generator_kwargs, key='params', only=only)

    @cachedproperty
    def notes(self) -> praw.models.SubredditModNotes:
        if False:
            while True:
                i = 10
        'Provide an instance of :class:`.SubredditModNotes`.\n\n        This provides an interface for managing moderator notes for this subreddit.\n\n        For example, all the notes for u/spez in r/test can be iterated through like so:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n\n            for note in subreddit.mod.notes.redditors("spez"):\n                print(f"{note.label}: {note.note}")\n\n        '
        from praw.models.mod_notes import SubredditModNotes
        return SubredditModNotes(self.subreddit._reddit, subreddit=self.subreddit)

    @cachedproperty
    def removal_reasons(self) -> SubredditRemovalReasons:
        if False:
            while True:
                i = 10
        'Provide an instance of :class:`.SubredditRemovalReasons`.\n\n        Use this attribute for interacting with a :class:`.Subreddit`\'s removal reasons.\n        For example to list all the removal reasons for a subreddit which you have the\n        ``posts`` moderator permission on, try:\n\n        .. code-block:: python\n\n            for removal_reason in reddit.subreddit("test").mod.removal_reasons:\n                print(removal_reason)\n\n        A single removal reason can be lazily retrieved via:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").mod.removal_reasons["reason_id"]\n\n        .. note::\n\n            Attempting to access attributes of an nonexistent removal reason will result\n            in a :class:`.ClientException`.\n\n        '
        return SubredditRemovalReasons(self.subreddit)

    @cachedproperty
    def stream(self) -> praw.models.reddit.subreddit.SubredditModerationStream:
        if False:
            while True:
                i = 10
        'Provide an instance of :class:`.SubredditModerationStream`.\n\n        Streams can be used to indefinitely retrieve Moderator only items from\n        :class:`.SubredditModeration` made to moderated subreddits, like:\n\n        .. code-block:: python\n\n            for log in reddit.subreddit("mod").mod.stream.log():\n                print(f"Mod: {log.mod}, Subreddit: {log.subreddit}")\n\n        '
        return SubredditModerationStream(self.subreddit)

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.SubredditModeration` instance.\n\n        :param subreddit: The subreddit to moderate.\n\n        '
        self.subreddit = subreddit
        self._stream = None

    def accept_invite(self):
        if False:
            print('Hello World!')
        'Accept an invitation as a moderator of the community.'
        url = API_PATH['accept_mod_invite'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url)

    @_deprecate_args('only')
    def edited(self, *, only: str | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Comment | praw.models.Submission]:
        if False:
            return 10
        'Return a :class:`.ListingGenerator` for edited comments and submissions.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To print all items in the edited queue try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.edited(limit=None):\n                print(item)\n\n        '
        self._handle_only(generator_kwargs=generator_kwargs, only=only)
        return ListingGenerator(self.subreddit._reddit, API_PATH['about_edited'].format(subreddit=self.subreddit), **generator_kwargs)

    def inbox(self, **generator_kwargs: Any) -> Iterator[praw.models.SubredditMessage]:
        if False:
            i = 10
            return i + 15
        'Return a :class:`.ListingGenerator` for moderator messages.\n\n        .. warning::\n\n            Legacy modmail is being deprecated in June 2021. Please see\n            https://www.reddit.com/r/modnews/comments/mar9ha/even_more_modmail_improvements/\n            for more info.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        .. seealso::\n\n            :meth:`.unread` for unread moderator messages.\n\n        To print the last 5 moderator mail messages and their replies, try:\n\n        .. code-block:: python\n\n            for message in reddit.subreddit("mod").mod.inbox(limit=5):\n                print(f"From: {message.author}, Body: {message.body}")\n                for reply in message.replies:\n                    print(f"From: {reply.author}, Body: {reply.body}")\n\n        '
        warn('Legacy modmail is being deprecated in June 2021. Please see https://www.reddit.com/r/modnews/comments/mar9ha/even_more_modmail_improvements/ for more info.', category=DeprecationWarning, stacklevel=3)
        return ListingGenerator(self.subreddit._reddit, API_PATH['moderator_messages'].format(subreddit=self.subreddit), **generator_kwargs)

    @_deprecate_args('action', 'mod')
    def log(self, *, action: str | None=None, mod: praw.models.Redditor | str | None=None, **generator_kwargs: Any) -> Iterator[praw.models.ModAction]:
        if False:
            return 10
        'Return a :class:`.ListingGenerator` for moderator log entries.\n\n        :param action: If given, only return log entries for the specified action.\n        :param mod: If given, only return log entries for actions made by the passed in\n            redditor.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To print the moderator and subreddit of the last 5 modlog entries try:\n\n        .. code-block:: python\n\n            for log in reddit.subreddit("mod").mod.log(limit=5):\n                print(f"Mod: {log.mod}, Subreddit: {log.subreddit}")\n\n        '
        params = {'mod': str(mod) if mod else mod, 'type': action}
        Subreddit._safely_add_arguments(arguments=generator_kwargs, key='params', **params)
        return ListingGenerator(self.subreddit._reddit, API_PATH['about_log'].format(subreddit=self.subreddit), **generator_kwargs)

    @_deprecate_args('only')
    def modqueue(self, *, only: str | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Submission | praw.models.Comment]:
        if False:
            return 10
        'Return a :class:`.ListingGenerator` for modqueue items.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To print all modqueue items try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.modqueue(limit=None):\n                print(item)\n\n        '
        self._handle_only(generator_kwargs=generator_kwargs, only=only)
        return ListingGenerator(self.subreddit._reddit, API_PATH['about_modqueue'].format(subreddit=self.subreddit), **generator_kwargs)

    @_deprecate_args('only')
    def reports(self, *, only: str | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Submission | praw.models.Comment]:
        if False:
            for i in range(10):
                print('nop')
        'Return a :class:`.ListingGenerator` for reported comments and submissions.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To print the user and mod report reasons in the report queue try:\n\n        .. code-block:: python\n\n            for reported_item in reddit.subreddit("mod").mod.reports():\n                print(f"User Reports: {reported_item.user_reports}")\n                print(f"Mod Reports: {reported_item.mod_reports}")\n\n        '
        self._handle_only(generator_kwargs=generator_kwargs, only=only)
        return ListingGenerator(self.subreddit._reddit, API_PATH['about_reports'].format(subreddit=self.subreddit), **generator_kwargs)

    def settings(self) -> dict[str, str | int | bool]:
        if False:
            print('Hello World!')
        "Return a dictionary of the :class:`.Subreddit`'s current settings."
        url = API_PATH['subreddit_settings'].format(subreddit=self.subreddit)
        return self.subreddit._reddit.get(url)['data']

    @_deprecate_args('only')
    def spam(self, *, only: str | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Submission | praw.models.Comment]:
        if False:
            while True:
                i = 10
        'Return a :class:`.ListingGenerator` for spam comments and submissions.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To print the items in the spam queue try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.spam():\n                print(item)\n\n        '
        self._handle_only(generator_kwargs=generator_kwargs, only=only)
        return ListingGenerator(self.subreddit._reddit, API_PATH['about_spam'].format(subreddit=self.subreddit), **generator_kwargs)

    def unmoderated(self, **generator_kwargs: Any) -> Iterator[praw.models.Submission]:
        if False:
            i = 10
            return i + 15
        'Return a :class:`.ListingGenerator` for unmoderated submissions.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To print the items in the unmoderated queue try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.unmoderated():\n                print(item)\n\n        '
        return ListingGenerator(self.subreddit._reddit, API_PATH['about_unmoderated'].format(subreddit=self.subreddit), **generator_kwargs)

    def unread(self, **generator_kwargs: Any) -> Iterator[praw.models.SubredditMessage]:
        if False:
            return 10
        'Return a :class:`.ListingGenerator` for unread moderator messages.\n\n        .. warning::\n\n            Legacy modmail is being deprecated in June 2021. Please see\n            https://www.reddit.com/r/modnews/comments/mar9ha/even_more_modmail_improvements/\n            for more info.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        .. seealso::\n\n            :meth:`.inbox` for all messages.\n\n        To print the mail in the unread modmail queue try:\n\n        .. code-block:: python\n\n            for message in reddit.subreddit("mod").mod.unread():\n                print(f"From: {message.author}, To: {message.dest}")\n\n        '
        warn('Legacy modmail is being deprecated in June 2021. Please see https://www.reddit.com/r/modnews/comments/mar9ha/even_more_modmail_improvements/ for more info.', category=DeprecationWarning, stacklevel=3)
        return ListingGenerator(self.subreddit._reddit, API_PATH['moderator_unread'].format(subreddit=self.subreddit), **generator_kwargs)

    def update(self, **settings: str | int | bool) -> dict[str, str | int | bool]:
        if False:
            for i in range(10):
                print('nop')
        'Update the :class:`.Subreddit`\'s settings.\n\n        See https://www.reddit.com/dev/api#POST_api_site_admin for the full list.\n\n        :param all_original_content: Mandate all submissions to be original content\n            only.\n        :param allow_chat_post_creation: Allow users to create chat submissions.\n        :param allow_images: Allow users to upload images using the native image\n            hosting.\n        :param allow_polls: Allow users to post polls to the subreddit.\n        :param allow_post_crossposts: Allow users to crosspost submissions from other\n            subreddits.\n        :param allow_videos: Allow users to upload videos using the native image\n            hosting.\n        :param collapse_deleted_comments: Collapse deleted and removed comments on\n            comments pages by default.\n        :param comment_score_hide_mins: The number of minutes to hide comment scores.\n        :param content_options: The types of submissions users can make. One of\n            ``"any"``, ``"link"``, or ``"self"``.\n        :param crowd_control_chat_level: Controls the crowd control level for chat\n            rooms. Goes from 0-3.\n        :param crowd_control_level: Controls the crowd control level for submissions.\n            Goes from 0-3.\n        :param crowd_control_mode: Enables/disables crowd control.\n        :param default_set: Allow the subreddit to appear on r/all as well as the\n            default and trending lists.\n        :param disable_contributor_requests: Specifies whether redditors may send\n            automated modmail messages requesting approval as a submitter.\n        :param exclude_banned_modqueue: Exclude posts by site-wide banned users from\n            modqueue/unmoderated.\n        :param free_form_reports: Allow users to specify custom reasons in the report\n            menu.\n        :param header_hover_text: The text seen when hovering over the snoo.\n        :param hide_ads: Don\'t show ads within this subreddit. Only applies to\n            Premium-user only subreddits.\n        :param key_color: A 6-digit rgb hex color (e.g., ``"#AABBCC"``), used as a\n            thematic color for your subreddit on mobile.\n        :param language: A valid IETF language tag (underscore separated).\n        :param original_content_tag_enabled: Enables the use of the ``original content``\n            label for submissions.\n        :param over_18: Viewers must be over 18 years old (i.e., NSFW).\n        :param public_description: Public description blurb. Appears in search results\n            and on the landing page for private subreddits.\n        :param restrict_commenting: Specifies whether approved users have the ability to\n            comment.\n        :param restrict_posting: Specifies whether approved users have the ability to\n            submit posts.\n        :param show_media: Show thumbnails on submissions.\n        :param show_media_preview: Expand media previews on comments pages.\n        :param spam_comments: Spam filter strength for comments. One of ``"all"``,\n            ``"low"``, or ``"high"``.\n        :param spam_links: Spam filter strength for links. One of ``"all"``, ``"low"``,\n            or ``"high"``.\n        :param spam_selfposts: Spam filter strength for selfposts. One of ``"all"``,\n            ``"low"``, or ``"high"``.\n        :param spoilers_enabled: Enable marking posts as containing spoilers.\n        :param submit_link_label: Custom label for submit link button (``None`` for\n            default).\n        :param submit_text: Text to show on submission page.\n        :param submit_text_label: Custom label for submit text post button (``None`` for\n            default).\n        :param subreddit_type: One of ``"archived"``, ``"employees_only"``,\n            ``"gold_only"``, ``gold_restricted``, ``"private"``, ``"public"``, or\n            ``"restricted"``.\n        :param suggested_comment_sort: All comment threads will use this sorting method\n            by default. Leave ``None``, or choose one of ``"confidence"``,\n            ``"controversial"``, ``"live"``, ``"new"``, ``"old"``, ``"qa"``,\n            ``"random"``, or ``"top"``.\n        :param title: The title of the subreddit.\n        :param welcome_message_enabled: Enables the subreddit welcome message.\n        :param welcome_message_text: The text to be used as a welcome message. A welcome\n            message is sent to all new subscribers by a Reddit bot.\n        :param wiki_edit_age: Account age, in days, required to edit and create wiki\n            pages.\n        :param wiki_edit_karma: Subreddit karma required to edit and create wiki pages.\n        :param wikimode: One of ``"anyone"``, ``"disabled"``, or ``"modonly"``.\n\n        .. note::\n\n            Updating the subreddit sidebar on old reddit (``description``) is no longer\n            supported using this method. You can update the sidebar by editing the\n            ``"config/sidebar"`` wiki page. For example:\n\n            .. code-block:: python\n\n                sidebar = reddit.subreddit("test").wiki["config/sidebar"]\n                sidebar.edit(content="new sidebar content")\n\n        Additional keyword arguments can be provided to handle new settings as Reddit\n        introduces them.\n\n        Settings that are documented here and aren\'t explicitly set by you in a call to\n        :meth:`.SubredditModeration.update` should retain their current value. If they\n        do not, please file a bug.\n\n        '
        remap = {'content_options': 'link_type', 'default_set': 'allow_top', 'header_hover_text': 'header_title', 'language': 'lang', 'subreddit_type': 'type'}
        settings = {remap.get(key, key): value for (key, value) in settings.items()}
        settings['sr'] = self.subreddit.fullname
        return self.subreddit._reddit.patch(API_PATH['update_settings'], json=settings)

class SubredditModerationStream:
    """Provides moderator streams."""

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.SubredditModerationStream` instance.\n\n        :param subreddit: The moderated subreddit associated with the streams.\n\n        '
        self.subreddit = subreddit

    @_deprecate_args('only')
    def edited(self, *, only: str | None=None, **stream_options: Any) -> Generator[praw.models.Comment | praw.models.Submission, None, None]:
        if False:
            while True:
                i = 10
        'Yield edited comments and submissions as they become available.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        For example, to retrieve all new edited submissions/comments made to all\n        moderated subreddits, try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.stream.edited():\n                print(item)\n\n        '
        return stream_generator(self.subreddit.mod.edited, only=only, **stream_options)

    @_deprecate_args('action', 'mod')
    def log(self, *, action: str | None=None, mod: str | praw.models.Redditor | None=None, **stream_options: Any) -> Generator[praw.models.ModAction, None, None]:
        if False:
            print('Hello World!')
        'Yield moderator log entries as they become available.\n\n        :param action: If given, only return log entries for the specified action.\n        :param mod: If given, only return log entries for actions made by the passed in\n            redditor.\n\n        For example, to retrieve all new mod actions made to all moderated subreddits,\n        try:\n\n        .. code-block:: python\n\n            for log in reddit.subreddit("mod").mod.stream.log():\n                print(f"Mod: {log.mod}, Subreddit: {log.subreddit}")\n\n        '
        return stream_generator(self.subreddit.mod.log, attribute_name='id', action=action, mod=mod, **stream_options)

    @_deprecate_args('other_subreddits', 'sort', 'state')
    def modmail_conversations(self, *, other_subreddits: list[praw.models.Subreddit] | None=None, sort: str | None=None, state: str | None=None, **stream_options: Any) -> Generator[ModmailConversation, None, None]:
        if False:
            print('Hello World!')
        'Yield new-modmail conversations as they become available.\n\n        :param other_subreddits: A list of :class:`.Subreddit` instances for which to\n            fetch conversations (default: ``None``).\n        :param sort: Can be one of: ``"mod"``, ``"recent"``, ``"unread"``, or ``"user"``\n            (default: ``"recent"``).\n        :param state: Can be one of: ``"all"``, ``"appeals"``, ``"archived"``,\n            ``"default"``, ``"highlighted"``, ``"inbox"``, ``"inprogress"``,\n            ``"join_requests"``, ``"mod"``, ``"new"``, or ``"notifications"`` (default:\n            ``"all"``). ``"all"`` does not include mod or archived conversations.\n            ``"inbox"`` does not include appeals conversations.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        To print new mail in the unread modmail queue try:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("all")\n            for message in subreddit.mod.stream.modmail_conversations():\n                print(f"From: {message.owner}, To: {message.participant}")\n\n        '
        if self.subreddit == 'mod':
            self.subreddit = self.subreddit._reddit.subreddit('all')
        return stream_generator(self.subreddit.modmail.conversations, attribute_name='id', exclude_before=True, other_subreddits=other_subreddits, sort=sort, state=state, **stream_options)

    @_deprecate_args('only')
    def modqueue(self, *, only: str | None=None, **stream_options: Any) -> Generator[praw.models.Comment | praw.models.Submission, None, None]:
        if False:
            while True:
                i = 10
        'Yield :class:`.Comment`\\ s and :class:`.Submission`\\ s in the modqueue as they become available.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        To print all new modqueue items try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.stream.modqueue():\n                print(item)\n\n        '
        return stream_generator(self.subreddit.mod.modqueue, only=only, **stream_options)

    @_deprecate_args('only')
    def reports(self, *, only: str | None=None, **stream_options: Any) -> Generator[praw.models.Comment | praw.models.Submission, None, None]:
        if False:
            i = 10
            return i + 15
        'Yield reported :class:`.Comment`\\ s and :class:`.Submission`\\ s as they become available.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        To print new user and mod report reasons in the report queue try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.stream.reports():\n                print(item)\n\n        '
        return stream_generator(self.subreddit.mod.reports, only=only, **stream_options)

    @_deprecate_args('only')
    def spam(self, *, only: str | None=None, **stream_options: Any) -> Generator[praw.models.Comment | praw.models.Submission, None, None]:
        if False:
            while True:
                i = 10
        'Yield spam :class:`.Comment`\\ s and :class:`.Submission`\\ s as they become available.\n\n        :param only: If specified, one of ``"comments"`` or ``"submissions"`` to yield\n            only results of that type.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        To print new items in the spam queue try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.stream.spam():\n                print(item)\n\n        '
        return stream_generator(self.subreddit.mod.spam, only=only, **stream_options)

    def unmoderated(self, **stream_options: Any) -> Generator[praw.models.Submission, None, None]:
        if False:
            print('Hello World!')
        'Yield unmoderated :class:`.Submission`\\ s as they become available.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        To print new items in the unmoderated queue try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("mod").mod.stream.unmoderated():\n                print(item)\n\n        '
        return stream_generator(self.subreddit.mod.unmoderated, **stream_options)

    def unread(self, **stream_options: Any) -> Generator[praw.models.SubredditMessage, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Yield unread old modmail messages as they become available.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        .. seealso::\n\n            :meth:`.SubredditModeration.inbox` for all messages.\n\n        To print new mail in the unread modmail queue try:\n\n        .. code-block:: python\n\n            for message in reddit.subreddit("mod").mod.stream.unread():\n                print(f"From: {message.author}, To: {message.dest}")\n\n        '
        return stream_generator(self.subreddit.mod.unread, **stream_options)

class SubredditQuarantine:
    """Provides subreddit quarantine related methods.

    To opt-in into a quarantined subreddit:

    .. code-block:: python

        reddit.subreddit("test").quaran.opt_in()

    """

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            print('Hello World!')
        'Initialize a :class:`.SubredditQuarantine` instance.\n\n        :param subreddit: The :class:`.Subreddit` associated with the quarantine.\n\n        '
        self.subreddit = subreddit

    def opt_in(self):
        if False:
            for i in range(10):
                print('nop')
        'Permit your user access to the quarantined subreddit.\n\n        Usage:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("QUESTIONABLE")\n            next(subreddit.hot())  # Raises prawcore.Forbidden\n\n            subreddit.quaran.opt_in()\n            next(subreddit.hot())  # Returns Submission\n\n        '
        data = {'sr_name': self.subreddit}
        with contextlib.suppress(Redirect):
            self.subreddit._reddit.post(API_PATH['quarantine_opt_in'], data=data)

    def opt_out(self):
        if False:
            return 10
        'Remove access to the quarantined subreddit.\n\n        Usage:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("QUESTIONABLE")\n            next(subreddit.hot())  # Returns Submission\n\n            subreddit.quaran.opt_out()\n            next(subreddit.hot())  # Raises prawcore.Forbidden\n\n        '
        data = {'sr_name': self.subreddit}
        with contextlib.suppress(Redirect):
            self.subreddit._reddit.post(API_PATH['quarantine_opt_out'], data=data)

class SubredditRelationship:
    """Represents a relationship between a :class:`.Redditor` and :class:`.Subreddit`.

    Instances of this class can be iterated through in order to discover the Redditors
    that make up the relationship.

    For example, banned users of a subreddit can be iterated through like so:

    .. code-block:: python

        for ban in reddit.subreddit("test").banned():
            print(f"{ban}: {ban.note}")

    """

    def __call__(self, redditor: str | praw.models.Redditor | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Redditor]:
        if False:
            for i in range(10):
                print('nop')
        'Return a :class:`.ListingGenerator` for :class:`.Redditor`\\ s in the relationship.\n\n        :param redditor: When provided, yield at most a single :class:`.Redditor`\n            instance. This is useful to confirm if a relationship exists, or to fetch\n            the metadata associated with a particular relationship (default: ``None``).\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        '
        Subreddit._safely_add_arguments(arguments=generator_kwargs, key='params', user=redditor)
        url = API_PATH[f'list_{self.relationship}'].format(subreddit=self.subreddit)
        return ListingGenerator(self.subreddit._reddit, url, **generator_kwargs)

    def __init__(self, subreddit: praw.models.Subreddit, relationship: str):
        if False:
            while True:
                i = 10
        'Initialize a :class:`.SubredditRelationship` instance.\n\n        :param subreddit: The :class:`.Subreddit` for the relationship.\n        :param relationship: The name of the relationship.\n\n        '
        self.relationship = relationship
        self.subreddit = subreddit

    def add(self, redditor: str | praw.models.Redditor, **other_settings: Any):
        if False:
            return 10
        'Add ``redditor`` to this relationship.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n\n        '
        data = {'name': str(redditor), 'type': self.relationship}
        data.update(other_settings)
        url = API_PATH['friend'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

    def remove(self, redditor: str | praw.models.Redditor):
        if False:
            i = 10
            return i + 15
        'Remove ``redditor`` from this relationship.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n\n        '
        data = {'name': str(redditor), 'type': self.relationship}
        url = API_PATH['unfriend'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

class SubredditStream:
    """Provides submission and comment streams."""

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a :class:`.SubredditStream` instance.\n\n        :param subreddit: The subreddit associated with the streams.\n\n        '
        self.subreddit = subreddit

    def comments(self, **stream_options: Any) -> Generator[praw.models.Comment, None, None]:
        if False:
            return 10
        'Yield new comments as they become available.\n\n        Comments are yielded oldest first. Up to 100 historical comments will initially\n        be returned.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        .. note::\n\n            While PRAW tries to catch all new comments, some high-volume streams,\n            especially the r/all stream, may drop some comments.\n\n        For example, to retrieve all new comments made to r/test, try:\n\n        .. code-block:: python\n\n            for comment in reddit.subreddit("test").stream.comments():\n                print(comment)\n\n        To only retrieve new submissions starting when the stream is created, pass\n        ``skip_existing=True``:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            for comment in subreddit.stream.comments(skip_existing=True):\n                print(comment)\n\n        '
        return stream_generator(self.subreddit.comments, **stream_options)

    def submissions(self, **stream_options: Any) -> Generator[praw.models.Submission, None, None]:
        if False:
            print('Hello World!')
        'Yield new :class:`.Submission`\\ s as they become available.\n\n        Submissions are yielded oldest first. Up to 100 historical submissions will\n        initially be returned.\n\n        Keyword arguments are passed to :func:`.stream_generator`.\n\n        .. note::\n\n            While PRAW tries to catch all new submissions, some high-volume streams,\n            especially the r/all stream, may drop some submissions.\n\n        For example, to retrieve all new submissions made to all of Reddit, try:\n\n        .. code-block:: python\n\n            for submission in reddit.subreddit("all").stream.submissions():\n                print(submission)\n\n        '
        return stream_generator(self.subreddit.new, **stream_options)

class SubredditStylesheet:
    """Provides a set of stylesheet functions to a :class:`.Subreddit`.

    For example, to add the css data ``.test{color:blue}`` to the existing stylesheet:

    .. code-block:: python

        subreddit = reddit.subreddit("test")
        stylesheet = subreddit.stylesheet()
        stylesheet.stylesheet += ".test{color:blue}"
        subreddit.stylesheet.update(stylesheet.stylesheet)

    """

    def __call__(self) -> praw.models.Stylesheet:
        if False:
            print('Hello World!')
        'Return the :class:`.Subreddit`\'s stylesheet.\n\n        To be used as:\n\n        .. code-block:: python\n\n            stylesheet = reddit.subreddit("test").stylesheet()\n\n        '
        url = API_PATH['about_stylesheet'].format(subreddit=self.subreddit)
        return self.subreddit._reddit.get(url)

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.SubredditStylesheet` instance.\n\n        :param subreddit: The :class:`.Subreddit` associated with the stylesheet.\n\n        An instance of this class is provided as:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet\n\n        '
        self.subreddit = subreddit

    def _update_structured_styles(self, style_data: dict[str, str | Any]):
        if False:
            print('Hello World!')
        url = API_PATH['structured_styles'].format(subreddit=self.subreddit)
        self.subreddit._reddit.patch(url, data=style_data)

    def _upload_image(self, *, data: dict[str, str | Any], image_path: str) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        with Path(image_path).open('rb') as image:
            header = image.read(len(JPEG_HEADER))
            image.seek(0)
            data['img_type'] = 'jpg' if header == JPEG_HEADER else 'png'
            url = API_PATH['upload_image'].format(subreddit=self.subreddit)
            response = self.subreddit._reddit.post(url, data=data, files={'file': image})
            if response['errors']:
                error_type = response['errors'][0]
                error_value = response.get('errors_values', [''])[0]
                assert error_type in ['BAD_CSS_NAME', 'IMAGE_ERROR'], 'Please file a bug with PRAW.'
                raise RedditAPIException([[error_type, error_value, None]])
            return response

    def _upload_style_asset(self, *, image_path: str, image_type: str) -> str:
        if False:
            print('Hello World!')
        file = Path(image_path)
        data = {'imagetype': image_type, 'filepath': file.name}
        data['mimetype'] = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            data['mimetype'] = 'image/png'
        url = API_PATH['style_asset_lease'].format(subreddit=self.subreddit)
        upload_lease = self.subreddit._reddit.post(url, data=data)['s3UploadLease']
        upload_data = {item['name']: item['value'] for item in upload_lease['fields']}
        upload_url = f"https:{upload_lease['action']}"
        with file.open('rb') as image:
            response = self.subreddit._reddit._core._requestor._http.post(upload_url, data=upload_data, files={'file': image})
        response.raise_for_status()
        return f"{upload_url}/{upload_data['key']}"

    def delete_banner(self):
        if False:
            print('Hello World!')
        'Remove the current :class:`.Subreddit` (redesign) banner image.\n\n        Succeeds even if there is no banner image.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_banner()\n\n        '
        data = {'bannerBackgroundImage': ''}
        self._update_structured_styles(data)

    def delete_banner_additional_image(self):
        if False:
            print('Hello World!')
        'Remove the current :class:`.Subreddit` (redesign) banner additional image.\n\n        Succeeds even if there is no additional image. Will also delete any configured\n        hover image.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_banner_additional_image()\n\n        '
        data = {'bannerPositionedImage': '', 'secondaryBannerPositionedImage': ''}
        self._update_structured_styles(data)

    def delete_banner_hover_image(self):
        if False:
            return 10
        'Remove the current :class:`.Subreddit` (redesign) banner hover image.\n\n        Succeeds even if there is no hover image.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_banner_hover_image()\n\n        '
        data = {'secondaryBannerPositionedImage': ''}
        self._update_structured_styles(data)

    def delete_header(self):
        if False:
            return 10
        'Remove the current :class:`.Subreddit` header image.\n\n        Succeeds even if there is no header image.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_header()\n\n        '
        url = API_PATH['delete_sr_header'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url)

    def delete_image(self, name: str):
        if False:
            while True:
                i = 10
        'Remove the named image from the :class:`.Subreddit`.\n\n        Succeeds even if the named image does not exist.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_image("smile")\n\n        '
        url = API_PATH['delete_sr_image'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data={'img_name': name})

    def delete_mobile_banner(self):
        if False:
            i = 10
            return i + 15
        'Remove the current :class:`.Subreddit` (redesign) mobile banner.\n\n        Succeeds even if there is no mobile banner.\n\n        For example:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            subreddit.stylesheet.delete_banner_hover_image()\n\n        '
        data = {'mobileBannerImage': ''}
        self._update_structured_styles(data)

    def delete_mobile_header(self):
        if False:
            i = 10
            return i + 15
        'Remove the current :class:`.Subreddit` mobile header.\n\n        Succeeds even if there is no mobile header.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_mobile_header()\n\n        '
        url = API_PATH['delete_sr_header'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url)

    def delete_mobile_icon(self):
        if False:
            print('Hello World!')
        'Remove the current :class:`.Subreddit` mobile icon.\n\n        Succeeds even if there is no mobile icon.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.delete_mobile_icon()\n\n        '
        url = API_PATH['delete_sr_icon'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url)

    @_deprecate_args('stylesheet', 'reason')
    def update(self, stylesheet: str, *, reason: str | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Update the :class:`.Subreddit`\'s stylesheet.\n\n        :param stylesheet: The CSS for the new stylesheet.\n        :param reason: The reason for updating the stylesheet.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.update(\n                "p { color: green; }", reason="color text green"\n            )\n\n        '
        data = {'op': 'save', 'reason': reason, 'stylesheet_contents': stylesheet}
        url = API_PATH['subreddit_stylesheet'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

    @_deprecate_args('name', 'image_path')
    def upload(self, *, image_path: str, name: str) -> dict[str, str]:
        if False:
            return 10
        'Upload an image to the :class:`.Subreddit`.\n\n        :param image_path: A path to a jpeg or png image.\n        :param name: The name to use for the image. If an image already exists with the\n            same name, it will be replaced.\n\n        :returns: A dictionary containing a link to the uploaded image under the key\n            ``img_src``.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.upload(name="smile", image_path="img.png")\n\n        '
        return self._upload_image(data={'name': name, 'upload_type': 'img'}, image_path=image_path)

    def upload_banner(self, image_path: str):
        if False:
            return 10
        'Upload an image for the :class:`.Subreddit`\'s (redesign) banner image.\n\n        :param image_path: A path to a jpeg or png image.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.upload_banner("banner.png")\n\n        '
        image_type = 'bannerBackgroundImage'
        image_url = self._upload_style_asset(image_path=image_path, image_type=image_type)
        self._update_structured_styles({image_type: image_url})

    @_deprecate_args('image_path', 'align')
    def upload_banner_additional_image(self, image_path: str, *, align: str | None=None):
        if False:
            return 10
        'Upload an image for the :class:`.Subreddit`\'s (redesign) additional image.\n\n        :param image_path: A path to a jpeg or png image.\n        :param align: Either ``"left"``, ``"centered"``, or ``"right"``. (default:\n            ``"left"``).\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            subreddit.stylesheet.upload_banner_additional_image("banner.png")\n\n        '
        alignment = {}
        if align is not None:
            if align not in {'left', 'centered', 'right'}:
                msg = "'align' argument must be either 'left', 'centered', or 'right'"
                raise ValueError(msg)
            alignment['bannerPositionedImagePosition'] = align
        image_type = 'bannerPositionedImage'
        image_url = self._upload_style_asset(image_path=image_path, image_type=image_type)
        style_data = {image_type: image_url}
        if alignment:
            style_data.update(alignment)
        self._update_structured_styles(style_data)

    def upload_banner_hover_image(self, image_path: str):
        if False:
            while True:
                i = 10
        'Upload an image for the :class:`.Subreddit`\'s (redesign) additional image.\n\n        :param image_path: A path to a jpeg or png image.\n\n        Fails if the :class:`.Subreddit` does not have an additional image defined.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            subreddit.stylesheet.upload_banner_hover_image("banner.png")\n\n        '
        image_type = 'secondaryBannerPositionedImage'
        image_url = self._upload_style_asset(image_path=image_path, image_type=image_type)
        self._update_structured_styles({image_type: image_url})

    def upload_header(self, image_path: str) -> dict[str, str]:
        if False:
            i = 10
            return i + 15
        'Upload an image to be used as the :class:`.Subreddit`\'s header image.\n\n        :param image_path: A path to a jpeg or png image.\n\n        :returns: A dictionary containing a link to the uploaded image under the key\n            ``img_src``.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.upload_header("header.png")\n\n        '
        return self._upload_image(data={'upload_type': 'header'}, image_path=image_path)

    def upload_mobile_banner(self, image_path: str):
        if False:
            print('Hello World!')
        'Upload an image for the :class:`.Subreddit`\'s (redesign) mobile banner.\n\n        :param image_path: A path to a JPEG or PNG image.\n\n        For example:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            subreddit.stylesheet.upload_mobile_banner("banner.png")\n\n        Fails if the :class:`.Subreddit` does not have an additional image defined.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        '
        image_type = 'mobileBannerImage'
        image_url = self._upload_style_asset(image_path=image_path, image_type=image_type)
        self._update_structured_styles({image_type: image_url})

    def upload_mobile_header(self, image_path: str) -> dict[str, str]:
        if False:
            return 10
        'Upload an image to be used as the :class:`.Subreddit`\'s mobile header.\n\n        :param image_path: A path to a jpeg or png image.\n\n        :returns: A dictionary containing a link to the uploaded image under the key\n            ``img_src``.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.upload_mobile_header("header.png")\n\n        '
        return self._upload_image(data={'upload_type': 'banner'}, image_path=image_path)

    def upload_mobile_icon(self, image_path: str) -> dict[str, str]:
        if False:
            return 10
        'Upload an image to be used as the :class:`.Subreddit`\'s mobile icon.\n\n        :param image_path: A path to a jpeg or png image.\n\n        :returns: A dictionary containing a link to the uploaded image under the key\n            ``img_src``.\n\n        :raises: ``prawcore.TooLarge`` if the overall request body is too large.\n\n        :raises: :class:`.RedditAPIException` if there are other issues with the\n            uploaded image. Unfortunately the exception info might not be very specific,\n            so try through the website with the same image to see what the problem\n            actually might be.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").stylesheet.upload_mobile_icon("icon.png")\n\n        '
        return self._upload_image(data={'upload_type': 'icon'}, image_path=image_path)

class SubredditWiki:
    """Provides a set of wiki functions to a :class:`.Subreddit`."""

    def __getitem__(self, page_name: str) -> WikiPage:
        if False:
            return 10
        'Lazily return the :class:`.WikiPage` for the :class:`.Subreddit` named ``page_name``.\n\n        This method is to be used to fetch a specific wikipage, like so:\n\n        .. code-block:: python\n\n            wikipage = reddit.subreddit("test").wiki["proof"]\n            print(wikipage.content_md)\n\n        '
        return WikiPage(self.subreddit._reddit, self.subreddit, page_name.lower())

    def __init__(self, subreddit: praw.models.Subreddit):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.SubredditWiki` instance.\n\n        :param subreddit: The subreddit whose wiki to work with.\n\n        '
        self.banned = SubredditRelationship(subreddit, 'wikibanned')
        self.contributor = SubredditRelationship(subreddit, 'wikicontributor')
        self.subreddit = subreddit

    def __iter__(self) -> Generator[WikiPage, None, None]:
        if False:
            return 10
        'Iterate through the pages of the wiki.\n\n        This method is to be used to discover all wikipages for a subreddit:\n\n        .. code-block:: python\n\n            for wikipage in reddit.subreddit("test").wiki:\n                print(wikipage)\n\n        '
        response = self.subreddit._reddit.get(API_PATH['wiki_pages'].format(subreddit=self.subreddit), params={'unique': self.subreddit._reddit._next_unique})
        for page_name in response['data']:
            yield WikiPage(self.subreddit._reddit, self.subreddit, page_name)

    @_deprecate_args('name', 'content', 'reason')
    def create(self, *, content: str, name: str, reason: str | None=None, **other_settings: Any) -> WikiPage:
        if False:
            print('Hello World!')
        'Create a new :class:`.WikiPage`.\n\n        :param name: The name of the new :class:`.WikiPage`. This name will be\n            normalized.\n        :param content: The content of the new :class:`.WikiPage`.\n        :param reason: The reason for the creation.\n        :param other_settings: Additional keyword arguments to pass.\n\n        To create the wiki page ``"praw_test"`` in r/test try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").wiki.create(\n                name="praw_test", content="wiki body text", reason="PRAW Test Creation"\n            )\n\n        '
        name = name.replace(' ', '_').lower()
        new = WikiPage(self.subreddit._reddit, self.subreddit, name)
        new.edit(content=content, reason=reason, **other_settings)
        return new

    def revisions(self, **generator_kwargs: Any) -> Generator[dict[str, praw.models.Redditor | WikiPage | str | int | bool | None], None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Return a :class:`.ListingGenerator` for recent wiki revisions.\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        To view the wiki revisions for ``"praw_test"`` in r/test try:\n\n        .. code-block:: python\n\n            for item in reddit.subreddit("test").wiki["praw_test"].revisions():\n                print(item)\n\n        '
        url = API_PATH['wiki_revisions'].format(subreddit=self.subreddit)
        return WikiPage._revision_generator(generator_kwargs=generator_kwargs, subreddit=self.subreddit, url=url)

class ContributorRelationship(SubredditRelationship):
    """Provides methods to interact with a :class:`.Subreddit`'s contributors.

    Contributors are also known as approved submitters.

    Contributors of a subreddit can be iterated through like so:

    .. code-block:: python

        for contributor in reddit.subreddit("test").contributor():
            print(contributor)

    """

    def leave(self):
        if False:
            return 10
        'Abdicate the contributor position.'
        self.subreddit._reddit.post(API_PATH['leavecontributor'], data={'id': self.subreddit.fullname})

class ModeratorRelationship(SubredditRelationship):
    """Provides methods to interact with a :class:`.Subreddit`'s moderators.

    Moderators of a subreddit can be iterated through like so:

    .. code-block:: python

        for moderator in reddit.subreddit("test").moderator():
            print(moderator)

    """
    PERMISSIONS = {'access', 'chat_config', 'chat_operator', 'config', 'flair', 'mail', 'posts', 'wiki'}

    @staticmethod
    def _handle_permissions(*, other_settings: dict | None=None, permissions: list[str] | None=None) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        other_settings = deepcopy(other_settings) if other_settings else {}
        other_settings['permissions'] = permissions_string(known_permissions=ModeratorRelationship.PERMISSIONS, permissions=permissions)
        return other_settings

    def __call__(self, redditor: str | praw.models.Redditor | None=None) -> list[praw.models.Redditor]:
        if False:
            return 10
        'Return a list of :class:`.Redditor`\\ s who are moderators.\n\n        :param redditor: When provided, return a list containing at most one\n            :class:`.Redditor` instance. This is useful to confirm if a relationship\n            exists, or to fetch the metadata associated with a particular relationship\n            (default: ``None``).\n\n        .. note::\n\n            To help mitigate targeted moderator harassment, this call requires the\n            :class:`.Reddit` instance to be authenticated i.e., :attr:`.read_only` must\n            return ``False``. This call, however, only makes use of the ``read`` scope.\n            For more information on why the moderator list is hidden can be found here:\n            https://reddit.zendesk.com/hc/en-us/articles/360049499032-Why-is-the-moderator-list-hidden-\n\n        .. note::\n\n            Unlike other relationship callables, this relationship is not paginated.\n            Thus, it simply returns the full list, rather than an iterator for the\n            results.\n\n        To be used like:\n\n        .. code-block:: python\n\n            moderators = reddit.subreddit("test").moderator()\n\n        For example, to list the moderators along with their permissions try:\n\n        .. code-block:: python\n\n            for moderator in reddit.subreddit("test").moderator():\n                print(f"{moderator}: {moderator.mod_permissions}")\n\n        '
        params = {} if redditor is None else {'user': redditor}
        url = API_PATH[f'list_{self.relationship}'].format(subreddit=self.subreddit)
        return self.subreddit._reddit.get(url, params=params)

    @_deprecate_args('redditor', 'permissions')
    def add(self, redditor: str | praw.models.Redditor, *, permissions: list[str] | None=None, **other_settings: Any):
        if False:
            while True:
                i = 10
        'Add or invite ``redditor`` to be a moderator of the :class:`.Subreddit`.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n        :param permissions: When provided (not ``None``), permissions should be a list\n            of strings specifying which subset of permissions to grant. An empty list\n            ``[]`` indicates no permissions, and when not provided ``None``, indicates\n            full permissions (default: ``None``).\n\n        An invite will be sent unless the user making this call is an admin user.\n\n        For example, to invite u/spez with ``"posts"`` and ``"mail"`` permissions to\n        r/test, try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").moderator.add("spez", permissions=["posts", "mail"])\n\n        '
        other_settings = self._handle_permissions(other_settings=other_settings, permissions=permissions)
        super().add(redditor, **other_settings)

    @_deprecate_args('redditor', 'permissions')
    def invite(self, redditor: str | praw.models.Redditor, *, permissions: list[str] | None=None, **other_settings: Any):
        if False:
            i = 10
            return i + 15
        'Invite ``redditor`` to be a moderator of the :class:`.Subreddit`.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n        :param permissions: When provided (not ``None``), permissions should be a list\n            of strings specifying which subset of permissions to grant. An empty list\n            ``[]`` indicates no permissions, and when not provided ``None``, indicates\n            full permissions (default: ``None``).\n\n        For example, to invite u/spez with ``"posts"`` and ``"mail"`` permissions to\n        r/test, try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").moderator.invite("spez", permissions=["posts", "mail"])\n\n        '
        data = self._handle_permissions(other_settings=other_settings, permissions=permissions)
        data.update({'name': str(redditor), 'type': 'moderator_invite'})
        url = API_PATH['friend'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

    @_deprecate_args('redditor')
    def invited(self, *, redditor: str | praw.models.Redditor | None=None, **generator_kwargs: Any) -> Iterator[praw.models.Redditor]:
        if False:
            i = 10
            return i + 15
        'Return a :class:`.ListingGenerator` for :class:`.Redditor`\\ s invited to be moderators.\n\n        :param redditor: When provided, return a list containing at most one\n            :class:`.Redditor` instance. This is useful to confirm if a relationship\n            exists, or to fetch the metadata associated with a particular relationship\n            (default: ``None``).\n\n        Additional keyword arguments are passed in the initialization of\n        :class:`.ListingGenerator`.\n\n        .. note::\n\n            Unlike other usages of :class:`.ListingGenerator`, ``limit`` has no effect\n            in the quantity returned. This endpoint always returns moderators in batches\n            of 25 at a time regardless of what ``limit`` is set to.\n\n        Usage:\n\n        .. code-block:: python\n\n            for invited_mod in reddit.subreddit("test").moderator.invited():\n                print(invited_mod)\n\n        '
        generator_kwargs['params'] = {'username': redditor} if redditor else None
        url = API_PATH['list_invited_moderator'].format(subreddit=self.subreddit)
        return ListingGenerator(self.subreddit._reddit, url, **generator_kwargs)

    def leave(self):
        if False:
            for i in range(10):
                print('nop')
        'Abdicate the moderator position (use with care).\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").moderator.leave()\n\n        '
        self.remove(self.subreddit._reddit.config.username or self.subreddit._reddit.user.me())

    def remove_invite(self, redditor: str | praw.models.Redditor):
        if False:
            print('Hello World!')
        'Remove the moderator invite for ``redditor``.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").moderator.remove_invite("spez")\n\n        '
        data = {'name': str(redditor), 'type': 'moderator_invite'}
        url = API_PATH['unfriend'].format(subreddit=self.subreddit)
        self.subreddit._reddit.post(url, data=data)

    @_deprecate_args('redditor', 'permissions')
    def update(self, redditor: str | praw.models.Redditor, *, permissions: list[str] | None=None):
        if False:
            i = 10
            return i + 15
        'Update the moderator permissions for ``redditor``.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n        :param permissions: When provided (not ``None``), permissions should be a list\n            of strings specifying which subset of permissions to grant. An empty list\n            ``[]`` indicates no permissions, and when not provided, ``None``, indicates\n            full permissions (default: ``None``).\n\n        For example, to add all permissions to the moderator, try:\n\n        .. code-block:: python\n\n            subreddit.moderator.update("spez")\n\n        To remove all permissions from the moderator, try:\n\n        .. code-block:: python\n\n            subreddit.moderator.update("spez", permissions=[])\n\n        '
        url = API_PATH['setpermissions'].format(subreddit=self.subreddit)
        data = self._handle_permissions(other_settings={'name': str(redditor), 'type': 'moderator'}, permissions=permissions)
        self.subreddit._reddit.post(url, data=data)

    @_deprecate_args('redditor', 'permissions')
    def update_invite(self, redditor: str | praw.models.Redditor, *, permissions: list[str] | None=None):
        if False:
            while True:
                i = 10
        'Update the moderator invite permissions for ``redditor``.\n\n        :param redditor: A redditor name or :class:`.Redditor` instance.\n        :param permissions: When provided (not ``None``), permissions should be a list\n            of strings specifying which subset of permissions to grant. An empty list\n            ``[]`` indicates no permissions, and when not provided, ``None``, indicates\n            full permissions (default: ``None``).\n\n        For example, to grant the ``"flair"`` and ``"mail"`` permissions to the\n        moderator invite, try:\n\n        .. code-block:: python\n\n            subreddit.moderator.update_invite("spez", permissions=["flair", "mail"])\n\n        '
        url = API_PATH['setpermissions'].format(subreddit=self.subreddit)
        data = self._handle_permissions(other_settings={'name': str(redditor), 'type': 'moderator_invite'}, permissions=permissions)
        self.subreddit._reddit.post(url, data=data)

class Subreddit(MessageableMixin, SubredditListingMixin, FullnameMixin, RedditBase):
    """A class for Subreddits.

    To obtain an instance of this class for r/test execute:

    .. code-block:: python

        subreddit = reddit.subreddit("test")

    While r/all is not a real subreddit, it can still be treated like one. The following
    outputs the titles of the 25 hottest submissions in r/all:

    .. code-block:: python

        for submission in reddit.subreddit("all").hot(limit=25):
            print(submission.title)

    Multiple subreddits can be combined with a ``+`` like so:

    .. code-block:: python

        for submission in reddit.subreddit("redditdev+learnpython").top(time_filter="all"):
            print(submission)

    Subreddits can be filtered from combined listings as follows.

    .. note::

        These filters are ignored by certain methods, including :attr:`.comments`,
        :meth:`.gilded`, and :meth:`.SubredditStream.comments`.

    .. code-block:: python

        for submission in reddit.subreddit("all-redditdev").new():
            print(submission)

    .. include:: ../../typical_attributes.rst

    ========================= ==========================================================
    Attribute                 Description
    ========================= ==========================================================
    ``can_assign_link_flair`` Whether users can assign their own link flair.
    ``can_assign_user_flair`` Whether users can assign their own user flair.
    ``created_utc``           Time the subreddit was created, represented in `Unix
                              Time`_.
    ``description``           Subreddit description, in Markdown.
    ``description_html``      Subreddit description, in HTML.
    ``display_name``          Name of the subreddit.
    ``icon_img``              The URL of the subreddit icon image.
    ``id``                    ID of the subreddit.
    ``name``                  Fullname of the subreddit.
    ``over18``                Whether the subreddit is NSFW.
    ``public_description``    Description of the subreddit, shown in searches and on the
                              "You must be invited to visit this community" page (if
                              applicable).
    ``spoilers_enabled``      Whether the spoiler tag feature is enabled.
    ``subscribers``           Count of subscribers.
    ``user_is_banned``        Whether the authenticated user is banned.
    ``user_is_moderator``     Whether the authenticated user is a moderator.
    ``user_is_subscriber``    Whether the authenticated user is subscribed.
    ========================= ==========================================================

    .. note::

        Trying to retrieve attributes of quarantined or private subreddits will result
        in a 403 error. Trying to retrieve attributes of a banned subreddit will result
        in a 404 error.

    .. _unix time: https://en.wikipedia.org/wiki/Unix_time

    """
    STR_FIELD = 'display_name'
    MESSAGE_PREFIX = '#'

    @staticmethod
    def _create_or_update(*, _reddit: praw.Reddit, allow_images: bool | None=None, allow_post_crossposts: bool | None=None, allow_top: bool | None=None, collapse_deleted_comments: bool | None=None, comment_score_hide_mins: int | None=None, description: str | None=None, domain: str | None=None, exclude_banned_modqueue: bool | None=None, header_hover_text: str | None=None, hide_ads: bool | None=None, lang: str | None=None, key_color: str | None=None, link_type: str | None=None, name: str | None=None, over_18: bool | None=None, public_description: str | None=None, public_traffic: bool | None=None, show_media: bool | None=None, show_media_preview: bool | None=None, spam_comments: bool | None=None, spam_links: bool | None=None, spam_selfposts: bool | None=None, spoilers_enabled: bool | None=None, sr: str | None=None, submit_link_label: str | None=None, submit_text: str | None=None, submit_text_label: str | None=None, subreddit_type: str | None=None, suggested_comment_sort: str | None=None, title: str | None=None, wiki_edit_age: int | None=None, wiki_edit_karma: int | None=None, wikimode: str | None=None, **other_settings: Any):
        if False:
            while True:
                i = 10
        model = {'allow_images': allow_images, 'allow_post_crossposts': allow_post_crossposts, 'allow_top': allow_top, 'collapse_deleted_comments': collapse_deleted_comments, 'comment_score_hide_mins': comment_score_hide_mins, 'description': description, 'domain': domain, 'exclude_banned_modqueue': exclude_banned_modqueue, 'header-title': header_hover_text, 'hide_ads': hide_ads, 'key_color': key_color, 'lang': lang, 'link_type': link_type, 'name': name, 'over_18': over_18, 'public_description': public_description, 'public_traffic': public_traffic, 'show_media': show_media, 'show_media_preview': show_media_preview, 'spam_comments': spam_comments, 'spam_links': spam_links, 'spam_selfposts': spam_selfposts, 'spoilers_enabled': spoilers_enabled, 'sr': sr, 'submit_link_label': submit_link_label, 'submit_text': submit_text, 'submit_text_label': submit_text_label, 'suggested_comment_sort': suggested_comment_sort, 'title': title, 'type': subreddit_type, 'wiki_edit_age': wiki_edit_age, 'wiki_edit_karma': wiki_edit_karma, 'wikimode': wikimode}
        model.update(other_settings)
        _reddit.post(API_PATH['site_admin'], data=model)

    @staticmethod
    def _subreddit_list(*, other_subreddits: list[str | praw.models.Subreddit], subreddit: praw.models.Subreddit) -> str:
        if False:
            print('Hello World!')
        if other_subreddits:
            return ','.join([str(subreddit)] + [str(x) for x in other_subreddits])
        return str(subreddit)

    @staticmethod
    def _validate_gallery(images: list[dict[str, str]]):
        if False:
            while True:
                i = 10
        for image in images:
            image_path = image.get('image_path', '')
            if image_path:
                if not Path(image_path).is_file():
                    msg = f'{image_path!r} is not a valid image path.'
                    raise TypeError(msg)
            else:
                msg = "'image_path' is required."
                raise TypeError(msg)
            if not len(image.get('caption', '')) <= 180:
                msg = 'Caption must be 180 characters or less.'
                raise TypeError(msg)

    @staticmethod
    def _validate_inline_media(inline_media: praw.models.InlineMedia):
        if False:
            i = 10
            return i + 15
        if not Path(inline_media.path).is_file():
            msg = f'{inline_media.path!r} is not a valid file path.'
            raise ValueError(msg)

    @cachedproperty
    def banned(self) -> praw.models.reddit.subreddit.SubredditRelationship:
        if False:
            return 10
        'Provide an instance of :class:`.SubredditRelationship`.\n\n        For example, to ban a user try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").banned.add("spez", ban_reason="...")\n\n        To list the banned users along with any notes, try:\n\n        .. code-block:: python\n\n            for ban in reddit.subreddit("test").banned():\n                print(f"{ban}: {ban.note}")\n\n        '
        return SubredditRelationship(self, 'banned')

    @cachedproperty
    def collections(self) -> praw.models.reddit.collections.SubredditCollections:
        if False:
            return 10
        'Provide an instance of :class:`.SubredditCollections`.\n\n        To see the permalinks of all :class:`.Collection`\\ s that belong to a subreddit,\n        try:\n\n        .. code-block:: python\n\n            for collection in reddit.subreddit("test").collections:\n                print(collection.permalink)\n\n        To get a specific :class:`.Collection` by its UUID or permalink, use one of the\n        following:\n\n        .. code-block:: python\n\n            collection = reddit.subreddit("test").collections("some_uuid")\n            collection = reddit.subreddit("test").collections(\n                permalink="https://reddit.com/r/SUBREDDIT/collection/some_uuid"\n            )\n\n        '
        return self._subreddit_collections_class(self._reddit, self)

    @cachedproperty
    def contributor(self) -> praw.models.reddit.subreddit.ContributorRelationship:
        if False:
            return 10
        'Provide an instance of :class:`.ContributorRelationship`.\n\n        Contributors are also known as approved submitters.\n\n        To add a contributor try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").contributor.add("spez")\n\n        '
        return ContributorRelationship(self, 'contributor')

    @cachedproperty
    def emoji(self) -> SubredditEmoji:
        if False:
            while True:
                i = 10
        'Provide an instance of :class:`.SubredditEmoji`.\n\n        This attribute can be used to discover all emoji for a subreddit:\n\n        .. code-block:: python\n\n            for emoji in reddit.subreddit("test").emoji:\n                print(emoji)\n\n        A single emoji can be lazily retrieved via:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").emoji["emoji_name"]\n\n        .. note::\n\n            Attempting to access attributes of a nonexistent emoji will result in a\n            :class:`.ClientException`.\n\n        '
        return SubredditEmoji(self)

    @cachedproperty
    def filters(self) -> praw.models.reddit.subreddit.SubredditFilters:
        if False:
            print('Hello World!')
        'Provide an instance of :class:`.SubredditFilters`.\n\n        For example, to add a filter, run:\n\n        .. code-block:: python\n\n            reddit.subreddit("all").filters.add("test")\n\n        '
        return SubredditFilters(self)

    @cachedproperty
    def flair(self) -> praw.models.reddit.subreddit.SubredditFlair:
        if False:
            return 10
        'Provide an instance of :class:`.SubredditFlair`.\n\n        Use this attribute for interacting with a :class:`.Subreddit`\'s flair. For\n        example, to list all the flair for a subreddit which you have the ``flair``\n        moderator permission on try:\n\n        .. code-block:: python\n\n            for flair in reddit.subreddit("test").flair():\n                print(flair)\n\n        Flair templates can be interacted with through this attribute via:\n\n        .. code-block:: python\n\n            for template in reddit.subreddit("test").flair.templates:\n                print(template)\n\n        '
        return SubredditFlair(self)

    @cachedproperty
    def mod(self) -> SubredditModeration:
        if False:
            return 10
        'Provide an instance of :class:`.SubredditModeration`.\n\n        For example, to accept a moderation invite from r/test:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").mod.accept_invite()\n\n        '
        return SubredditModeration(self)

    @cachedproperty
    def moderator(self) -> praw.models.reddit.subreddit.ModeratorRelationship:
        if False:
            for i in range(10):
                print('nop')
        'Provide an instance of :class:`.ModeratorRelationship`.\n\n        For example, to add a moderator try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").moderator.add("spez")\n\n        To list the moderators along with their permissions try:\n\n        .. code-block:: python\n\n            for moderator in reddit.subreddit("test").moderator():\n                print(f"{moderator}: {moderator.mod_permissions}")\n\n        '
        return ModeratorRelationship(self, 'moderator')

    @cachedproperty
    def modmail(self) -> praw.models.reddit.subreddit.Modmail:
        if False:
            for i in range(10):
                print('nop')
        'Provide an instance of :class:`.Modmail`.\n\n        For example, to send a new modmail from r/test to u/spez with the subject\n        ``"test"`` along with a message body of ``"hello"``:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail.create(subject="test", body="hello", recipient="spez")\n\n        '
        return Modmail(self)

    @cachedproperty
    def muted(self) -> praw.models.reddit.subreddit.SubredditRelationship:
        if False:
            for i in range(10):
                print('nop')
        'Provide an instance of :class:`.SubredditRelationship`.\n\n        For example, muted users can be iterated through like so:\n\n        .. code-block:: python\n\n            for mute in reddit.subreddit("test").muted():\n                print(f"{mute}: {mute.date}")\n\n        '
        return SubredditRelationship(self, 'muted')

    @cachedproperty
    def quaran(self) -> praw.models.reddit.subreddit.SubredditQuarantine:
        if False:
            return 10
        'Provide an instance of :class:`.SubredditQuarantine`.\n\n        This property is named ``quaran`` because ``quarantine`` is a subreddit\n        attribute returned by Reddit to indicate whether or not a subreddit is\n        quarantined.\n\n        To opt-in into a quarantined subreddit:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").quaran.opt_in()\n\n        '
        return SubredditQuarantine(self)

    @cachedproperty
    def rules(self) -> SubredditRules:
        if False:
            for i in range(10):
                print('nop')
        'Provide an instance of :class:`.SubredditRules`.\n\n        Use this attribute for interacting with a :class:`.Subreddit`\'s rules.\n\n        For example, to list all the rules for a subreddit:\n\n        .. code-block:: python\n\n            for rule in reddit.subreddit("test").rules:\n                print(rule)\n\n        Moderators can also add rules to the subreddit. For example, to make a rule\n        called ``"No spam"`` in r/test:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").rules.mod.add(\n                short_name="No spam", kind="all", description="Do not spam. Spam bad"\n            )\n\n        '
        return SubredditRules(self)

    @cachedproperty
    def stream(self) -> praw.models.reddit.subreddit.SubredditStream:
        if False:
            while True:
                i = 10
        'Provide an instance of :class:`.SubredditStream`.\n\n        Streams can be used to indefinitely retrieve new comments made to a subreddit,\n        like:\n\n        .. code-block:: python\n\n            for comment in reddit.subreddit("test").stream.comments():\n                print(comment)\n\n        Additionally, new submissions can be retrieved via the stream. In the following\n        example all submissions are fetched via the special r/all:\n\n        .. code-block:: python\n\n            for submission in reddit.subreddit("all").stream.submissions():\n                print(submission)\n\n        '
        return SubredditStream(self)

    @cachedproperty
    def stylesheet(self) -> praw.models.reddit.subreddit.SubredditStylesheet:
        if False:
            i = 10
            return i + 15
        'Provide an instance of :class:`.SubredditStylesheet`.\n\n        For example, to add the css data ``.test{color:blue}`` to the existing\n        stylesheet:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            stylesheet = subreddit.stylesheet()\n            stylesheet.stylesheet += ".test{color:blue}"\n            subreddit.stylesheet.update(stylesheet.stylesheet)\n\n        '
        return SubredditStylesheet(self)

    @cachedproperty
    def widgets(self) -> praw.models.SubredditWidgets:
        if False:
            for i in range(10):
                print('nop')
        'Provide an instance of :class:`.SubredditWidgets`.\n\n        **Example usage**\n\n        Get all sidebar widgets:\n\n        .. code-block:: python\n\n            for widget in reddit.subreddit("test").widgets.sidebar:\n                print(widget)\n\n        Get ID card widget:\n\n        .. code-block:: python\n\n            print(reddit.subreddit("test").widgets.id_card)\n\n        '
        return SubredditWidgets(self)

    @cachedproperty
    def wiki(self) -> praw.models.reddit.subreddit.SubredditWiki:
        if False:
            print('Hello World!')
        'Provide an instance of :class:`.SubredditWiki`.\n\n        This attribute can be used to discover all wikipages for a subreddit:\n\n        .. code-block:: python\n\n            for wikipage in reddit.subreddit("test").wiki:\n                print(wikipage)\n\n        To fetch the content for a given wikipage try:\n\n        .. code-block:: python\n\n            wikipage = reddit.subreddit("test").wiki["proof"]\n            print(wikipage.content_md)\n\n        '
        return SubredditWiki(self)

    @property
    def _kind(self) -> str:
        if False:
            return 10
        "Return the class's kind."
        return self._reddit.config.kinds['subreddit']

    def __init__(self, reddit: praw.Reddit, display_name: str | None=None, _data: dict[str, Any] | None=None):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.Subreddit` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n        :param display_name: The name of the subreddit.\n\n        .. note::\n\n            This class should not be initialized directly. Instead, obtain an instance\n            via:\n\n            .. code-block:: python\n\n                subreddit = reddit.subreddit("test")\n\n        '
        if (display_name, _data).count(None) != 1:
            msg = "Either 'display_name' or '_data' must be provided."
            raise TypeError(msg)
        if display_name:
            self.display_name = display_name
        super().__init__(reddit, _data=_data)
        self._path = API_PATH['subreddit'].format(subreddit=self)

    def _convert_to_fancypants(self, markdown_text: str) -> dict:
        if False:
            while True:
                i = 10
        'Convert a Markdown string to a dict for use with the ``richtext_json`` param.\n\n        :param markdown_text: A Markdown string to convert.\n\n        :returns: A dict in ``richtext_json`` format.\n\n        '
        text_data = {'output_mode': 'rtjson', 'markdown_text': markdown_text}
        return self._reddit.post(API_PATH['convert_rte_body'], data=text_data)['output']

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
        return ('subreddit_about', {'subreddit': self}, None)

    def _parse_xml_response(self, response: Response):
        if False:
            while True:
                i = 10
        'Parse the XML from a response and raise any errors found.'
        xml = response.text
        root = XML(xml)
        tags = [element.tag for element in root]
        if tags[:4] == ['Code', 'Message', 'ProposedSize', 'MaxSizeAllowed']:
            (code, message, actual, maximum_size) = (element.text for element in root[:4])
            raise TooLargeMediaException(actual=int(actual), maximum_size=int(maximum_size))

    def _read_and_post_media(self, media_path: str, upload_url: str, upload_data: dict[str, Any]) -> Response:
        if False:
            while True:
                i = 10
        with media_path.open('rb') as media:
            return self._reddit._core._requestor._http.post(upload_url, data=upload_data, files={'file': media})

    def _submit_media(self, *, data: dict[Any, Any], timeout: int, without_websockets: bool):
        if False:
            print('Hello World!')
        'Submit and return an ``image``, ``video``, or ``videogif``.\n\n        This is a helper method for submitting posts that are not link posts or self\n        posts.\n\n        '
        response = self._reddit.post(API_PATH['submit'], data=data)
        websocket_url = response['json']['data']['websocket_url']
        connection = None
        if websocket_url is not None and (not without_websockets):
            try:
                connection = websocket.create_connection(websocket_url, timeout=timeout)
            except (OSError, websocket.WebSocketException, BlockingIOError) as ws_exception:
                msg = 'Error establishing websocket connection.'
                raise WebSocketException(msg, ws_exception) from None
        if connection is None:
            return None
        try:
            ws_update = loads(connection.recv())
            connection.close()
        except (OSError, websocket.WebSocketException, BlockingIOError) as ws_exception:
            msg = 'Websocket error. Check your media file. Your post may still have been created.'
            raise WebSocketException(msg, ws_exception) from None
        if ws_update.get('type') == 'failed':
            raise MediaPostFailed
        url = ws_update['payload']['redirect']
        return self._reddit.submission(url=url)

    def _upload_inline_media(self, inline_media: praw.models.InlineMedia):
        if False:
            print('Hello World!')
        'Upload media for use in self posts and return ``inline_media``.\n\n        :param inline_media: An :class:`.InlineMedia` object to validate and upload.\n\n        '
        self._validate_inline_media(inline_media)
        inline_media.media_id = self._upload_media(media_path=inline_media.path, upload_type='selfpost')
        return inline_media

    def _upload_media(self, *, expected_mime_prefix: str | None=None, media_path: str, upload_type: str='link'):
        if False:
            for i in range(10):
                print('nop')
        'Upload media and return its URL and a websocket (Undocumented endpoint).\n\n        :param expected_mime_prefix: If provided, enforce that the media has a mime type\n            that starts with the provided prefix.\n        :param upload_type: One of ``"link"``, ``"gallery"\'\', or ``"selfpost"``\n            (default: ``"link"``).\n\n        :returns: A tuple containing ``(media_url, websocket_url)`` for the piece of\n            media. The websocket URL can be used to determine when media processing is\n            finished, or it can be ignored.\n\n        '
        if media_path is None:
            file = Path(__file__).absolute()
            media_path = file.parent.parent.parent / 'images' / 'PRAW logo.png'
        else:
            file = Path(media_path)
        file_name = file.name.lower()
        file_extension = file_name.rpartition('.')[2]
        mime_type = {'png': 'image/png', 'mov': 'video/quicktime', 'mp4': 'video/mp4', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif'}.get(file_extension, 'image/jpeg')
        if expected_mime_prefix is not None and mime_type.partition('/')[0] != expected_mime_prefix:
            msg = f'Expected a mimetype starting with {expected_mime_prefix!r} but got mimetype {mime_type!r} (from file extension {file_extension!r}).'
            raise ClientException(msg)
        img_data = {'filepath': file_name, 'mimetype': mime_type}
        url = API_PATH['media_asset']
        upload_response = self._reddit.post(url, data=img_data)
        upload_lease = upload_response['args']
        upload_url = f"https:{upload_lease['action']}"
        upload_data = {item['name']: item['value'] for item in upload_lease['fields']}
        response = self._read_and_post_media(file, upload_url, upload_data)
        if not response.ok:
            self._parse_xml_response(response)
        try:
            response.raise_for_status()
        except HTTPError as err:
            raise ServerError(response=err.response) from None
        upload_response['asset']['websocket_url']
        if upload_type == 'link':
            return f"{upload_url}/{upload_data['key']}"
        return upload_response['asset']['asset_id']

    def post_requirements(self) -> dict[str, str | int | bool]:
        if False:
            for i in range(10):
                print('nop')
        'Get the post requirements for a subreddit.\n\n        :returns: A dict with the various requirements.\n\n        The returned dict contains the following keys:\n\n        - ``domain_blacklist``\n        - ``body_restriction_policy``\n        - ``domain_whitelist``\n        - ``title_regexes``\n        - ``body_blacklisted_strings``\n        - ``body_required_strings``\n        - ``title_text_min_length``\n        - ``is_flair_required``\n        - ``title_text_max_length``\n        - ``body_regexes``\n        - ``link_repost_age``\n        - ``body_text_min_length``\n        - ``link_restriction_policy``\n        - ``body_text_max_length``\n        - ``title_required_strings``\n        - ``title_blacklisted_strings``\n        - ``guidelines_text``\n        - ``guidelines_display_policy``\n\n        For example, to fetch the post requirements for r/test:\n\n        .. code-block:: python\n\n            print(reddit.subreddit("test").post_requirements)\n\n        '
        return self._reddit.get(API_PATH['post_requirements'].format(subreddit=str(self)))

    def random(self) -> praw.models.Submission | None:
        if False:
            return 10
        'Return a random :class:`.Submission`.\n\n        Returns ``None`` on subreddits that do not support the random feature. One\n        example, at the time of writing, is r/wallpapers.\n\n        For example, to get a random submission off of r/AskReddit:\n\n        .. code-block:: python\n\n            submission = reddit.subreddit("AskReddit").random()\n            print(submission.title)\n\n        '
        url = API_PATH['subreddit_random'].format(subreddit=self)
        try:
            self._reddit.get(url, params={'unique': self._reddit._next_unique})
        except Redirect as redirect:
            path = redirect.path
        try:
            return self._submission_class(self._reddit, url=urljoin(self._reddit.config.reddit_url, path))
        except ClientException:
            return None

    @_deprecate_args('query', 'sort', 'syntax', 'time_filter')
    def search(self, query: str, *, sort: str='relevance', syntax: str='lucene', time_filter: str='all', **generator_kwargs: Any) -> Iterator[praw.models.Submission]:
        if False:
            for i in range(10):
                print('nop')
        'Return a :class:`.ListingGenerator` for items that match ``query``.\n\n        :param query: The query string to search for.\n        :param sort: Can be one of: ``"relevance"``, ``"hot"``, ``"top"``, ``"new"``, or\n            ``"comments"``. (default: ``"relevance"``).\n        :param syntax: Can be one of: ``"cloudsearch"``, ``"lucene"``, or ``"plain"``\n            (default: ``"lucene"``).\n        :param time_filter: Can be one of: ``"all"``, ``"day"``, ``"hour"``,\n            ``"month"``, ``"week"``, or ``"year"`` (default: ``"all"``).\n\n        For more information on building a search query see:\n        https://www.reddit.com/wiki/search\n\n        For example, to search all subreddits for ``"praw"`` try:\n\n        .. code-block:: python\n\n            for submission in reddit.subreddit("all").search("praw"):\n                print(submission.title)\n\n        '
        self._validate_time_filter(time_filter)
        not_all = self.display_name.lower() != 'all'
        self._safely_add_arguments(arguments=generator_kwargs, key='params', q=query, restrict_sr=not_all, sort=sort, syntax=syntax, t=time_filter)
        url = API_PATH['search'].format(subreddit=self)
        return ListingGenerator(self._reddit, url, **generator_kwargs)

    @_deprecate_args('number')
    def sticky(self, *, number: int=1) -> praw.models.Submission:
        if False:
            while True:
                i = 10
        'Return a :class:`.Submission` object for a sticky of the subreddit.\n\n        :param number: Specify which sticky to return. 1 appears at the top (default:\n            ``1``).\n\n        :raises: ``prawcore.NotFound`` if the sticky does not exist.\n\n        For example, to get the stickied post on r/test:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").sticky()\n\n        '
        url = API_PATH['about_sticky'].format(subreddit=self)
        try:
            self._reddit.get(url, params={'num': number})
        except Redirect as redirect:
            path = redirect.path
        return self._submission_class(self._reddit, url=urljoin(self._reddit.config.reddit_url, path))

    @_deprecate_args('title', 'selftext', 'url', 'flair_id', 'flair_text', 'resubmit', 'send_replies', 'nsfw', 'spoiler', 'collection_id', 'discussion_type', 'inline_media', 'draft_id')
    def submit(self, title: str, *, collection_id: str | None=None, discussion_type: str | None=None, draft_id: str | None=None, flair_id: str | None=None, flair_text: str | None=None, inline_media: dict[str, praw.models.InlineMedia] | None=None, nsfw: bool=False, resubmit: bool=True, selftext: str | None=None, send_replies: bool=True, spoiler: bool=False, url: str | None=None) -> praw.models.Submission:
        if False:
            for i in range(10):
                print('nop')
        'Add a submission to the :class:`.Subreddit`.\n\n        :param title: The title of the submission.\n        :param collection_id: The UUID of a :class:`.Collection` to add the\n            newly-submitted post to.\n        :param discussion_type: Set to ``"CHAT"`` to enable live discussion instead of\n            traditional comments (default: ``None``).\n        :param draft_id: The ID of a draft to submit.\n        :param flair_id: The flair template to select (default: ``None``).\n        :param flair_text: If the template\'s ``flair_text_editable`` value is ``True``,\n            this value will set a custom text (default: ``None``). ``flair_id`` is\n            required when ``flair_text`` is provided.\n        :param inline_media: A dict of :class:`.InlineMedia` objects where the key is\n            the placeholder name in ``selftext``.\n        :param nsfw: Whether the submission should be marked NSFW (default: ``False``).\n        :param resubmit: When ``False``, an error will occur if the URL has already been\n            submitted (default: ``True``).\n        :param selftext: The Markdown formatted content for a ``text`` submission. Use\n            an empty string, ``""``, to make a title-only submission.\n        :param send_replies: When ``True``, messages will be sent to the submission\n            author when comments are made to the submission (default: ``True``).\n        :param spoiler: Whether the submission should be marked as a spoiler (default:\n            ``False``).\n        :param url: The URL for a ``link`` submission.\n\n        :returns: A :class:`.Submission` object for the newly created submission.\n\n        Either ``selftext`` or ``url`` can be provided, but not both.\n\n        For example, to submit a URL to r/test do:\n\n        .. code-block:: python\n\n            title = "PRAW documentation"\n            url = "https://praw.readthedocs.io"\n            reddit.subreddit("test").submit(title, url=url)\n\n        For example, to submit a self post with inline media do:\n\n        .. code-block:: python\n\n            from praw.models import InlineGif, InlineImage, InlineVideo\n\n            gif = InlineGif(path="path/to/image.gif", caption="optional caption")\n            image = InlineImage(path="path/to/image.jpg", caption="optional caption")\n            video = InlineVideo(path="path/to/video.mp4", caption="optional caption")\n            selftext = "Text with a gif {gif1} an image {image1} and a video {video1} inline"\n            media = {"gif1": gif, "image1": image, "video1": video}\n            reddit.subreddit("test").submit("title", inline_media=media, selftext=selftext)\n\n        .. note::\n\n            Inserted media will have a padding of ``\\\\n\\\\n`` automatically added. This\n            is due to the weirdness with Reddit\'s API. Using the example above, the\n            result selftext body will look like so:\n\n            .. code-block::\n\n                Text with a gif\n\n                ![gif](u1rchuphryq51 "optional caption")\n\n                an image\n\n                ![img](srnr8tshryq51 "optional caption")\n\n                and video\n\n                ![video](gmc7rvthryq51 "optional caption")\n\n                inline\n\n        .. note::\n\n            To submit a post to a subreddit with the ``"news"`` flair, you can get the\n            flair id like this:\n\n            .. code-block::\n\n                choices = list(subreddit.flair.link_templates.user_selectable())\n                template_id = next(x for x in choices if x["flair_text"] == "news")["flair_template_id"]\n                subreddit.submit("title", flair_id=template_id, url="https://www.news.com/")\n\n        .. seealso::\n\n            - :meth:`~.Subreddit.submit_gallery` to submit more than one image in the\n              same post\n            - :meth:`~.Subreddit.submit_image` to submit images\n            - :meth:`~.Subreddit.submit_poll` to submit polls\n            - :meth:`~.Subreddit.submit_video` to submit videos and videogifs\n\n        '
        if (bool(selftext) or selftext == '') == bool(url):
            msg = "Either 'selftext' or 'url' must be provided."
            raise TypeError(msg)
        data = {'sr': str(self), 'resubmit': bool(resubmit), 'sendreplies': bool(send_replies), 'title': title, 'nsfw': bool(nsfw), 'spoiler': bool(spoiler), 'validate_on_submit': self._reddit.validate_on_submit}
        for (key, value) in (('flair_id', flair_id), ('flair_text', flair_text), ('collection_id', collection_id), ('discussion_type', discussion_type), ('draft_id', draft_id)):
            if value is not None:
                data[key] = value
        if selftext is not None:
            data.update(kind='self')
            if inline_media:
                body = selftext.format(**{placeholder: self._upload_inline_media(media) for (placeholder, media) in inline_media.items()})
                converted = self._convert_to_fancypants(body)
                data.update(richtext_json=dumps(converted))
            else:
                data.update(text=selftext)
        else:
            data.update(kind='link', url=url)
        return self._reddit.post(API_PATH['submit'], data=data)

    @_deprecate_args('title', 'images', 'collection_id', 'discussion_type', 'flair_id', 'flair_text', 'nsfw', 'send_replies', 'spoiler')
    def submit_gallery(self, title: str, images: list[dict[str, str]], *, collection_id: str | None=None, discussion_type: str | None=None, flair_id: str | None=None, flair_text: str | None=None, nsfw: bool=False, send_replies: bool=True, spoiler: bool=False) -> praw.models.Submission:
        if False:
            for i in range(10):
                print('nop')
        'Add an image gallery submission to the subreddit.\n\n        :param title: The title of the submission.\n        :param images: The images to post in dict with the following structure:\n            ``{"image_path": "path", "caption": "caption", "outbound_url": "url"}``,\n            only ``image_path`` is required.\n        :param collection_id: The UUID of a :class:`.Collection` to add the\n            newly-submitted post to.\n        :param discussion_type: Set to ``"CHAT"`` to enable live discussion instead of\n            traditional comments (default: ``None``).\n        :param flair_id: The flair template to select (default: ``None``).\n        :param flair_text: If the template\'s ``flair_text_editable`` value is ``True``,\n            this value will set a custom text (default: ``None``). ``flair_id`` is\n            required when ``flair_text`` is provided.\n        :param nsfw: Whether the submission should be marked NSFW (default: ``False``).\n        :param send_replies: When ``True``, messages will be sent to the submission\n            author when comments are made to the submission (default: ``True``).\n        :param spoiler: Whether the submission should be marked asa spoiler (default:\n            ``False``).\n\n        :returns: A :class:`.Submission` object for the newly created submission.\n\n        :raises: :class:`.ClientException` if ``image_path`` in ``images`` refers to a\n            file that is not an image.\n\n        For example, to submit an image gallery to r/test do:\n\n        .. code-block:: python\n\n            title = "My favorite pictures"\n            image = "/path/to/image.png"\n            image2 = "/path/to/image2.png"\n            image3 = "/path/to/image3.png"\n            images = [\n                {"image_path": image},\n                {\n                    "image_path": image2,\n                    "caption": "Image caption 2",\n                },\n                {\n                    "image_path": image3,\n                    "caption": "Image caption 3",\n                    "outbound_url": "https://example.com/link3",\n                },\n            ]\n            reddit.subreddit("test").submit_gallery(title, images)\n\n        .. seealso::\n\n            - :meth:`~.Subreddit.submit` to submit url posts and selftexts\n            - :meth:`~.Subreddit.submit_image` to submit single images\n            - :meth:`~.Subreddit.submit_poll` to submit polls\n            - :meth:`~.Subreddit.submit_video` to submit videos and videogifs\n\n        '
        self._validate_gallery(images)
        data = {'api_type': 'json', 'items': [], 'nsfw': bool(nsfw), 'sendreplies': bool(send_replies), 'show_error_list': True, 'spoiler': bool(spoiler), 'sr': str(self), 'title': title, 'validate_on_submit': self._reddit.validate_on_submit}
        for (key, value) in (('flair_id', flair_id), ('flair_text', flair_text), ('collection_id', collection_id), ('discussion_type', discussion_type)):
            if value is not None:
                data[key] = value
        for image in images:
            data['items'].append({'caption': image.get('caption', ''), 'outbound_url': image.get('outbound_url', ''), 'media_id': self._upload_media(expected_mime_prefix='image', media_path=image['image_path'], upload_type='gallery')})
        response = self._reddit.request(json=data, method='POST', path=API_PATH['submit_gallery_post'])['json']
        if response['errors']:
            raise RedditAPIException(response['errors'])
        return self._reddit.submission(url=response['data']['url'])

    @_deprecate_args('title', 'image_path', 'flair_id', 'flair_text', 'resubmit', 'send_replies', 'nsfw', 'spoiler', 'timeout', 'collection_id', 'without_websockets', 'discussion_type')
    def submit_image(self, title: str, image_path: str, *, collection_id: str | None=None, discussion_type: str | None=None, flair_id: str | None=None, flair_text: str | None=None, nsfw: bool=False, resubmit: bool=True, send_replies: bool=True, spoiler: bool=False, timeout: int=10, without_websockets: bool=False) -> praw.models.Submission | None:
        if False:
            while True:
                i = 10
        'Add an image submission to the subreddit.\n\n        :param collection_id: The UUID of a :class:`.Collection` to add the\n            newly-submitted post to.\n        :param discussion_type: Set to ``"CHAT"`` to enable live discussion instead of\n            traditional comments (default: ``None``).\n        :param flair_id: The flair template to select (default: ``None``).\n        :param flair_text: If the template\'s ``flair_text_editable`` value is ``True``,\n            this value will set a custom text (default: ``None``). ``flair_id`` is\n            required when ``flair_text`` is provided.\n        :param image_path: The path to an image, to upload and post.\n        :param nsfw: Whether the submission should be marked NSFW (default: ``False``).\n        :param resubmit: When ``False``, an error will occur if the URL has already been\n            submitted (default: ``True``).\n        :param send_replies: When ``True``, messages will be sent to the submission\n            author when comments are made to the submission (default: ``True``).\n        :param spoiler: Whether the submission should be marked as a spoiler (default:\n            ``False``).\n        :param timeout: Specifies a particular timeout, in seconds. Use to avoid\n            "Websocket error" exceptions (default: ``10``).\n        :param title: The title of the submission.\n        :param without_websockets: Set to ``True`` to disable use of WebSockets (see\n            note below for an explanation). If ``True``, this method doesn\'t return\n            anything (default: ``False``).\n\n        :returns: A :class:`.Submission` object for the newly created submission, unless\n            ``without_websockets`` is ``True``.\n\n        :raises: :class:`.ClientException` if ``image_path`` refers to a file that is\n            not an image.\n\n        .. note::\n\n            Reddit\'s API uses WebSockets to respond with the link of the newly created\n            post. If this fails, the method will raise :class:`.WebSocketException`.\n            Occasionally, the Reddit post will still be created. More often, there is an\n            error with the image file. If you frequently get exceptions but successfully\n            created posts, try setting the ``timeout`` parameter to a value above 10.\n\n            To disable the use of WebSockets, set ``without_websockets=True``. This will\n            make the method return ``None``, though the post will still be created. You\n            may wish to do this if you are running your program in a restricted network\n            environment, or using a proxy that doesn\'t support WebSockets connections.\n\n        For example, to submit an image to r/test do:\n\n        .. code-block:: python\n\n            title = "My favorite picture"\n            image = "/path/to/image.png"\n            reddit.subreddit("test").submit_image(title, image)\n\n        .. seealso::\n\n            - :meth:`~.Subreddit.submit` to submit url posts and selftexts\n            - :meth:`~.Subreddit.submit_gallery` to submit more than one image in the\n              same post\n            - :meth:`~.Subreddit.submit_poll` to submit polls\n            - :meth:`~.Subreddit.submit_video` to submit videos and videogifs\n\n        '
        data = {'sr': str(self), 'resubmit': bool(resubmit), 'sendreplies': bool(send_replies), 'title': title, 'nsfw': bool(nsfw), 'spoiler': bool(spoiler), 'validate_on_submit': self._reddit.validate_on_submit}
        for (key, value) in (('flair_id', flair_id), ('flair_text', flair_text), ('collection_id', collection_id), ('discussion_type', discussion_type)):
            if value is not None:
                data[key] = value
        image_url = self._upload_media(expected_mime_prefix='image', media_path=image_path)
        data.update(kind='image', url=image_url)
        return self._submit_media(data=data, timeout=timeout, without_websockets=without_websockets)

    @_deprecate_args('title', 'selftext', 'options', 'duration', 'flair_id', 'flair_text', 'resubmit', 'send_replies', 'nsfw', 'spoiler', 'collection_id', 'discussion_type')
    def submit_poll(self, title: str, *, collection_id: str | None=None, discussion_type: str | None=None, duration: int, flair_id: str | None=None, flair_text: str | None=None, nsfw: bool=False, options: list[str], resubmit: bool=True, selftext: str, send_replies: bool=True, spoiler: bool=False) -> praw.models.Submission:
        if False:
            print('Hello World!')
        'Add a poll submission to the subreddit.\n\n        :param title: The title of the submission.\n        :param collection_id: The UUID of a :class:`.Collection` to add the\n            newly-submitted post to.\n        :param discussion_type: Set to ``"CHAT"`` to enable live discussion instead of\n            traditional comments (default: ``None``).\n        :param duration: The number of days the poll should accept votes, as an ``int``.\n            Valid values are between ``1`` and ``7``, inclusive.\n        :param flair_id: The flair template to select (default: ``None``).\n        :param flair_text: If the template\'s ``flair_text_editable`` value is ``True``,\n            this value will set a custom text (default: ``None``). ``flair_id`` is\n            required when ``flair_text`` is provided.\n        :param nsfw: Whether the submission should be marked NSFW (default: ``False``).\n        :param options: A list of two to six poll options as ``str``.\n        :param resubmit: When ``False``, an error will occur if the URL has already been\n            submitted (default: ``True``).\n        :param selftext: The Markdown formatted content for the submission. Use an empty\n            string, ``""``, to make a submission with no text contents.\n        :param send_replies: When ``True``, messages will be sent to the submission\n            author when comments are made to the submission (default: ``True``).\n        :param spoiler: Whether the submission should be marked as a spoiler (default:\n            ``False``).\n\n        :returns: A :class:`.Submission` object for the newly created submission.\n\n        For example, to submit a poll to r/test do:\n\n        .. code-block:: python\n\n            title = "Do you like PRAW?"\n            reddit.subreddit("test").submit_poll(\n                title, selftext="", options=["Yes", "No"], duration=3\n            )\n\n        .. seealso::\n\n            - :meth:`~.Subreddit.submit` to submit url posts and selftexts\n            - :meth:`~.Subreddit.submit_gallery` to submit more than one image in the\n              same post\n            - :meth:`~.Subreddit.submit_image` to submit single images\n            - :meth:`~.Subreddit.submit_video` to submit videos and videogifs\n\n        '
        data = {'sr': str(self), 'text': selftext, 'options': options, 'duration': duration, 'resubmit': bool(resubmit), 'sendreplies': bool(send_replies), 'title': title, 'nsfw': bool(nsfw), 'spoiler': bool(spoiler), 'validate_on_submit': self._reddit.validate_on_submit}
        for (key, value) in (('flair_id', flair_id), ('flair_text', flair_text), ('collection_id', collection_id), ('discussion_type', discussion_type)):
            if value is not None:
                data[key] = value
        return self._reddit.post(API_PATH['submit_poll_post'], json=data)

    @_deprecate_args('title', 'video_path', 'videogif', 'thumbnail_path', 'flair_id', 'flair_text', 'resubmit', 'send_replies', 'nsfw', 'spoiler', 'timeout', 'collection_id', 'without_websockets', 'discussion_type')
    def submit_video(self, title: str, video_path: str, *, collection_id: str | None=None, discussion_type: str | None=None, flair_id: str | None=None, flair_text: str | None=None, nsfw: bool=False, resubmit: bool=True, send_replies: bool=True, spoiler: bool=False, thumbnail_path: str | None=None, timeout: int=10, videogif: bool=False, without_websockets: bool=False) -> praw.models.Submission | None:
        if False:
            for i in range(10):
                print('nop')
        'Add a video or videogif submission to the subreddit.\n\n        :param title: The title of the submission.\n        :param video_path: The path to a video, to upload and post.\n        :param collection_id: The UUID of a :class:`.Collection` to add the\n            newly-submitted post to.\n        :param discussion_type: Set to ``"CHAT"`` to enable live discussion instead of\n            traditional comments (default: ``None``).\n        :param flair_id: The flair template to select (default: ``None``).\n        :param flair_text: If the template\'s ``flair_text_editable`` value is ``True``,\n            this value will set a custom text (default: ``None``). ``flair_id`` is\n            required when ``flair_text`` is provided.\n        :param nsfw: Whether the submission should be marked NSFW (default: ``False``).\n        :param resubmit: When ``False``, an error will occur if the URL has already been\n            submitted (default: ``True``).\n        :param send_replies: When ``True``, messages will be sent to the submission\n            author when comments are made to the submission (default: ``True``).\n        :param spoiler: Whether the submission should be marked as a spoiler (default:\n            ``False``).\n        :param thumbnail_path: The path to an image, to be uploaded and used as the\n            thumbnail for this video. If not provided, the PRAW logo will be used as the\n            thumbnail.\n        :param timeout: Specifies a particular timeout, in seconds. Use to avoid\n            "Websocket error" exceptions (default: ``10``).\n        :param videogif: If ``True``, the video is uploaded as a videogif, which is\n            essentially a silent video (default: ``False``).\n        :param without_websockets: Set to ``True`` to disable use of WebSockets (see\n            note below for an explanation). If ``True``, this method doesn\'t return\n            anything (default: ``False``).\n\n        :returns: A :class:`.Submission` object for the newly created submission, unless\n            ``without_websockets`` is ``True``.\n\n        :raises: :class:`.ClientException` if ``video_path`` refers to a file that is\n            not a video.\n\n        .. note::\n\n            Reddit\'s API uses WebSockets to respond with the link of the newly created\n            post. If this fails, the method will raise :class:`.WebSocketException`.\n            Occasionally, the Reddit post will still be created. More often, there is an\n            error with the image file. If you frequently get exceptions but successfully\n            created posts, try setting the ``timeout`` parameter to a value above 10.\n\n            To disable the use of WebSockets, set ``without_websockets=True``. This will\n            make the method return ``None``, though the post will still be created. You\n            may wish to do this if you are running your program in a restricted network\n            environment, or using a proxy that doesn\'t support WebSockets connections.\n\n        For example, to submit a video to r/test do:\n\n        .. code-block:: python\n\n            title = "My favorite movie"\n            video = "/path/to/video.mp4"\n            reddit.subreddit("test").submit_video(title, video)\n\n        .. seealso::\n\n            - :meth:`~.Subreddit.submit` to submit url posts and selftexts\n            - :meth:`~.Subreddit.submit_image` to submit images\n            - :meth:`~.Subreddit.submit_gallery` to submit more than one image in the\n              same post\n            - :meth:`~.Subreddit.submit_poll` to submit polls\n\n        '
        data = {'sr': str(self), 'resubmit': bool(resubmit), 'sendreplies': bool(send_replies), 'title': title, 'nsfw': bool(nsfw), 'spoiler': bool(spoiler), 'validate_on_submit': self._reddit.validate_on_submit}
        for (key, value) in (('flair_id', flair_id), ('flair_text', flair_text), ('collection_id', collection_id), ('discussion_type', discussion_type)):
            if value is not None:
                data[key] = value
        video_url = self._upload_media(expected_mime_prefix='video', media_path=video_path)
        data.update(kind='videogif' if videogif else 'video', url=video_url, video_poster_url=self._upload_media(media_path=thumbnail_path))
        return self._submit_media(data=data, timeout=timeout, without_websockets=without_websockets)

    @_deprecate_args('other_subreddits')
    def subscribe(self, *, other_subreddits: list[praw.models.Subreddit] | None=None):
        if False:
            print('Hello World!')
        'Subscribe to the subreddit.\n\n        :param other_subreddits: When provided, also subscribe to the provided list of\n            subreddits.\n\n        For example, to subscribe to r/test:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").subscribe()\n\n        '
        data = {'action': 'sub', 'skip_inital_defaults': True, 'sr_name': self._subreddit_list(other_subreddits=other_subreddits, subreddit=self)}
        self._reddit.post(API_PATH['subscribe'], data=data)

    def traffic(self) -> dict[str, list[list[int]]]:
        if False:
            print('Hello World!')
        'Return a dictionary of the :class:`.Subreddit`\'s traffic statistics.\n\n        :raises: ``prawcore.NotFound`` when the traffic stats aren\'t available to the\n            authenticated user, that is, they are not public and the authenticated user\n            is not a moderator of the subreddit.\n\n        The traffic method returns a dict with three keys. The keys are ``day``,\n        ``hour`` and ``month``. Each key contains a list of lists with 3 or 4 values.\n        The first value is a timestamp indicating the start of the category (start of\n        the day for the ``day`` key, start of the hour for the ``hour`` key, etc.). The\n        second, third, and fourth values indicate the unique pageviews, total pageviews,\n        and subscribers, respectively.\n\n        .. note::\n\n            The ``hour`` key does not contain subscribers, and therefore each sub-list\n            contains three values.\n\n        For example, to get the traffic stats for r/test:\n\n        .. code-block:: python\n\n            stats = reddit.subreddit("test").traffic()\n\n        '
        return self._reddit.get(API_PATH['about_traffic'].format(subreddit=self))

    @_deprecate_args('other_subreddits')
    def unsubscribe(self, *, other_subreddits: list[praw.models.Subreddit] | None=None):
        if False:
            print('Hello World!')
        'Unsubscribe from the subreddit.\n\n        :param other_subreddits: When provided, also unsubscribe from the provided list\n            of subreddits.\n\n        To unsubscribe from r/test:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").unsubscribe()\n\n        '
        data = {'action': 'unsub', 'sr_name': self._subreddit_list(other_subreddits=other_subreddits, subreddit=self)}
        self._reddit.post(API_PATH['subscribe'], data=data)
WidgetEncoder._subreddit_class = Subreddit

class SubredditLinkFlairTemplates(SubredditFlairTemplates):
    """Provide functions to interact with link flair templates."""

    def __iter__(self) -> Generator[dict[str, str | int | bool | list[dict[str, str]]], None, None]:
        if False:
            print('Hello World!')
        'Iterate through the link flair templates as a moderator.\n\n        For example:\n\n        .. code-block:: python\n\n            for template in reddit.subreddit("test").flair.link_templates:\n                print(template)\n\n        '
        url = API_PATH['link_flair'].format(subreddit=self.subreddit)
        yield from self.subreddit._reddit.get(url)

    @_deprecate_args('text', 'css_class', 'text_editable', 'background_color', 'text_color', 'mod_only', 'allowable_content', 'max_emojis')
    def add(self, text: str, *, allowable_content: str | None=None, background_color: str | None=None, css_class: str='', max_emojis: int | None=None, mod_only: bool | None=None, text_color: str | None=None, text_editable: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Add a link flair template to the associated subreddit.\n\n        :param text: The flair template\'s text.\n        :param allowable_content: If specified, most be one of ``"all"``, ``"emoji"``,\n            or ``"text"`` to restrict content to that type. If set to ``"emoji"`` then\n            the ``"text"`` param must be a valid emoji string, for example,\n            ``":snoo:"``.\n        :param background_color: The flair template\'s new background color, as a hex\n            color.\n        :param css_class: The flair template\'s css_class (default: ``""``).\n        :param max_emojis: Maximum emojis in the flair (Reddit defaults this value to\n            ``10``).\n        :param mod_only: Indicate if the flair can only be used by moderators.\n        :param text_color: The flair template\'s new text color, either ``"light"`` or\n            ``"dark"``.\n        :param text_editable: Indicate if the flair text can be modified for each\n            redditor that sets it (default: ``False``).\n\n        For example, to add an editable link flair try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").flair.link_templates.add(\n                "PRAW",\n                css_class="praw",\n                text_editable=True,\n            )\n\n        '
        self._add(allowable_content=allowable_content, background_color=background_color, css_class=css_class, is_link=True, max_emojis=max_emojis, mod_only=mod_only, text=text, text_color=text_color, text_editable=text_editable)

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Remove all link flair templates from the subreddit.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").flair.link_templates.clear()\n\n        '
        self._clear(is_link=True)

    def reorder(self, flair_list: list[str]):
        if False:
            for i in range(10):
                print('nop')
        'Reorder a list of flairs.\n\n        :param flair_list: A list of flair IDs.\n\n        For example, to reverse the order of the link flair list try:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            flairs = [flair["id"] for flair in subreddit.flair.link_templates]\n            subreddit.flair.link_templates.reorder(list(reversed(flairs)))\n\n        '
        self._reorder(flair_list, is_link=True)

    def user_selectable(self) -> Generator[dict[str, str | bool], None, None]:
        if False:
            while True:
                i = 10
        'Iterate through the link flair templates as a regular user.\n\n        For example:\n\n        .. code-block:: python\n\n            for template in reddit.subreddit("test").flair.link_templates.user_selectable():\n                print(template)\n\n        '
        url = API_PATH['flairselector'].format(subreddit=self.subreddit)
        yield from self.subreddit._reddit.post(url, data={'is_newlink': True})['choices']

class SubredditRedditorFlairTemplates(SubredditFlairTemplates):
    """Provide functions to interact with :class:`.Redditor` flair templates."""

    def __iter__(self) -> Generator[dict[str, str | int | bool | list[dict[str, str]]], None, None]:
        if False:
            while True:
                i = 10
        'Iterate through the user flair templates.\n\n        For example:\n\n        .. code-block:: python\n\n            for template in reddit.subreddit("test").flair.templates:\n                print(template)\n\n        '
        url = API_PATH['user_flair'].format(subreddit=self.subreddit)
        params = {'unique': self.subreddit._reddit._next_unique}
        yield from self.subreddit._reddit.get(url, params=params)

    @_deprecate_args('text', 'css_class', 'text_editable', 'background_color', 'text_color', 'mod_only', 'allowable_content', 'max_emojis')
    def add(self, text: str, *, allowable_content: str | None=None, background_color: str | None=None, css_class: str='', max_emojis: int | None=None, mod_only: bool | None=None, text_color: str | None=None, text_editable: bool=False):
        if False:
            while True:
                i = 10
        'Add a redditor flair template to the associated subreddit.\n\n        :param text: The flair template\'s text.\n        :param allowable_content: If specified, most be one of ``"all"``, ``"emoji"``,\n            or ``"text"`` to restrict content to that type. If set to ``"emoji"`` then\n            the ``"text"`` param must be a valid emoji string, for example,\n            ``":snoo:"``.\n        :param background_color: The flair template\'s new background color, as a hex\n            color.\n        :param css_class: The flair template\'s css_class (default: ``""``).\n        :param max_emojis: Maximum emojis in the flair (Reddit defaults this value to\n            ``10``).\n        :param mod_only: Indicate if the flair can only be used by moderators.\n        :param text_color: The flair template\'s new text color, either ``"light"`` or\n            ``"dark"``.\n        :param text_editable: Indicate if the flair text can be modified for each\n            redditor that sets it (default: ``False``).\n\n        For example, to add an editable redditor flair try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").flair.templates.add(\n                "PRAW",\n                css_class="praw",\n                text_editable=True,\n            )\n\n        '
        self._add(allowable_content=allowable_content, background_color=background_color, css_class=css_class, is_link=False, max_emojis=max_emojis, mod_only=mod_only, text=text, text_color=text_color, text_editable=text_editable)

    def clear(self):
        if False:
            while True:
                i = 10
        'Remove all :class:`.Redditor` flair templates from the subreddit.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").flair.templates.clear()\n\n        '
        self._clear(is_link=False)

    def reorder(self, flair_list: list[str]):
        if False:
            while True:
                i = 10
        'Reorder a list of flairs.\n\n        :param flair_list: A list of flair IDs.\n\n        For example, to reverse the order of the :class:`.Redditor` flair templates list\n        try:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            flairs = [flair["id"] for flair in subreddit.flair.templates]\n            subreddit.flair.templates.reorder(list(reversed(flairs)))\n\n        '
        self._reorder(flair_list, is_link=False)