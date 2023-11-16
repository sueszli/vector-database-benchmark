"""Provide models for new modmail."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ...const import API_PATH
from ...util import _deprecate_args, snake_case_keys
from .base import RedditBase
if TYPE_CHECKING:
    import praw

class ModmailObject(RedditBase):
    """A base class for objects within a modmail conversation."""
    AUTHOR_ATTRIBUTE = 'author'
    STR_FIELD = 'id'

    def __setattr__(self, attribute: str, value: Any):
        if False:
            while True:
                i = 10
        'Objectify the AUTHOR_ATTRIBUTE attribute.'
        if attribute == self.AUTHOR_ATTRIBUTE:
            value = self._reddit._objector.objectify(value)
        super().__setattr__(attribute, value)

class ModmailConversation(RedditBase):
    """A class for modmail conversations.

    .. include:: ../../typical_attributes.rst

    ==================== ===============================================================
    Attribute            Description
    ==================== ===============================================================
    ``authors``          Provides an ordered list of :class:`.Redditor` instances. The
                         authors of each message in the modmail conversation.
    ``id``               The ID of the :class:`.ModmailConversation`.
    ``is_highlighted``   Whether or not the :class:`.ModmailConversation` is
                         highlighted.
    ``is_internal``      Whether or not the :class:`.ModmailConversation` is a private
                         mod conversation.
    ``last_mod_update``  Time of the last mod message reply, represented in the `ISO
                         8601`_ standard with timezone.
    ``last_updated``     Time of the last message reply, represented in the `ISO 8601`_
                         standard with timezone.
    ``last_user_update`` Time of the last user message reply, represented in the `ISO
                         8601`_ standard with timezone.
    ``num_messages``     The number of messages in the :class:`.ModmailConversation`.
    ``obj_ids``          Provides a list of dictionaries representing mod actions on the
                         :class:`.ModmailConversation`. Each dict contains attributes of
                         ``"key"`` and ``"id"``. The key can be either ``""messages"``
                         or ``"ModAction"``. ``"ModAction"`` represents
                         archiving/highlighting etc.
    ``owner``            Provides an instance of :class:`.Subreddit`. The subreddit that
                         the :class:`.ModmailConversation` belongs to.
    ``participant``      Provides an instance of :class:`.Redditor`. The participating
                         user in the :class:`.ModmailConversation`.
    ``subject``          The subject of the :class:`.ModmailConversation`.
    ==================== ===============================================================

    .. _iso 8601: https://en.wikipedia.org/wiki/ISO_8601

    """
    STR_FIELD = 'id'

    @staticmethod
    def _convert_conversation_objects(data: dict[str, Any], reddit: praw.Reddit):
        if False:
            for i in range(10):
                print('nop')
        'Convert messages and mod actions to PRAW objects.'
        result = {'messages': [], 'modActions': []}
        for thing in data['objIds']:
            key = thing['key']
            thing_data = data[key][thing['id']]
            result[key].append(reddit._objector.objectify(thing_data))
        data.update(result)

    @staticmethod
    def _convert_user_summary(data: dict[str, Any], reddit: praw.Reddit):
        if False:
            while True:
                i = 10
        'Convert dictionaries of recent user history to PRAW objects.'
        parsers = {'recentComments': reddit._objector.parsers[reddit.config.kinds['comment']], 'recentConvos': ModmailConversation, 'recentPosts': reddit._objector.parsers[reddit.config.kinds['submission']]}
        for (kind, parser) in parsers.items():
            objects = []
            for (thing_id, summary) in data[kind].items():
                thing = parser(reddit, id=thing_id.rsplit('_', 1)[-1])
                if parser is not ModmailConversation:
                    del summary['permalink']
                for (key, value) in summary.items():
                    setattr(thing, key, value)
                objects.append(thing)
            data[kind] = sorted(objects, key=lambda x: int(x.id, base=36), reverse=True)

    @classmethod
    def parse(cls, data: dict[str, Any], reddit: praw.Reddit) -> ModmailConversation:
        if False:
            for i in range(10):
                print('nop')
        'Return an instance of :class:`.ModmailConversation` from ``data``.\n\n        :param data: The structured data.\n        :param reddit: An instance of :class:`.Reddit`.\n\n        '
        data['authors'] = [reddit._objector.objectify(author) for author in data['authors']]
        for entity in ('owner', 'participant'):
            data[entity] = reddit._objector.objectify(data[entity])
        if data.get('user'):
            cls._convert_user_summary(data['user'], reddit)
            data['user'] = reddit._objector.objectify(data['user'])
        data = snake_case_keys(data)
        return cls(reddit, _data=data)

    def __init__(self, reddit: praw.Reddit, id: str | None=None, mark_read: bool=False, _data: dict[str, Any] | None=None):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.ModmailConversation` instance.\n\n        :param mark_read: If ``True``, conversation is marked as read (default:\n            ``False``).\n\n        '
        if bool(id) == bool(_data):
            msg = "Either 'id' or '_data' must be provided."
            raise TypeError(msg)
        if id:
            self.id = id
        super().__init__(reddit, _data=_data)
        self._info_params = {'markRead': True} if mark_read else None

    def _build_conversation_list(self, other_conversations: list[ModmailConversation]) -> str:
        if False:
            return 10
        'Return a comma-separated list of conversation IDs.'
        conversations = [self] + (other_conversations or [])
        return ','.join((conversation.id for conversation in conversations))

    def _fetch(self):
        if False:
            return 10
        data = self._fetch_data()
        other = self._reddit._objector.objectify(data)
        self.__dict__.update(other.__dict__)
        super()._fetch()

    def _fetch_info(self):
        if False:
            return 10
        return ('modmail_conversation', {'id': self.id}, self._info_params)

    def archive(self):
        if False:
            print('Hello World!')
        'Archive the conversation.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").archive()\n\n        '
        self._reddit.post(API_PATH['modmail_archive'].format(id=self.id))

    def highlight(self):
        if False:
            return 10
        'Highlight the conversation.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").highlight()\n\n        '
        self._reddit.post(API_PATH['modmail_highlight'].format(id=self.id))

    @_deprecate_args('num_days')
    def mute(self, *, num_days: int=3):
        if False:
            return 10
        'Mute the non-mod user associated with the conversation.\n\n        :param num_days: Duration of mute in days. Valid options are ``3``, ``7``, or\n            ``28`` (default: ``3``).\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").mute()\n\n        To mute for 7 days:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").mute(num_days=7)\n\n        '
        params = {'num_hours': num_days * 24} if num_days != 3 else {}
        self._reddit.request(method='POST', params=params, path=API_PATH['modmail_mute'].format(id=self.id))

    @_deprecate_args('other_conversations')
    def read(self, *, other_conversations: list[ModmailConversation] | None=None):
        if False:
            return 10
        'Mark the conversation(s) as read.\n\n        :param other_conversations: A list of other conversations to mark (default:\n            ``None``).\n\n        For example, to mark the conversation as read along with other recent\n        conversations from the same user:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            conversation = subreddit.modmail.conversation("2gmz")\n            conversation.read(other_conversations=conversation.user.recent_convos)\n\n        '
        data = {'conversationIds': self._build_conversation_list(other_conversations)}
        self._reddit.post(API_PATH['modmail_read'], data=data)

    @_deprecate_args('body', 'author_hidden', 'internal')
    def reply(self, *, author_hidden: bool=False, body: str, internal: bool=False) -> ModmailMessage:
        if False:
            print('Hello World!')
        'Reply to the conversation.\n\n        :param author_hidden: When ``True``, author is hidden from non-moderators\n            (default: ``False``).\n        :param body: The Markdown formatted content for a message.\n        :param internal: When ``True``, message is a private moderator note, hidden from\n            non-moderators (default: ``False``).\n\n        :returns: A :class:`.ModmailMessage` object for the newly created message.\n\n        For example, to reply to the non-mod user while hiding your username:\n\n        .. code-block:: python\n\n            conversation = reddit.subreddit("test").modmail("2gmz")\n            conversation.reply(body="Message body", author_hidden=True)\n\n        To create a private moderator note on the conversation:\n\n        .. code-block:: python\n\n            conversation.reply(body="Message body", internal=True)\n\n        '
        data = {'body': body, 'isAuthorHidden': author_hidden, 'isInternal': internal}
        response = self._reddit.post(API_PATH['modmail_conversation'].format(id=self.id), data=data)
        if isinstance(response, dict):
            message_id = response['conversation']['objIds'][-1]['id']
            message_data = response['messages'][message_id]
            return self._reddit._objector.objectify(message_data)
        for message in response.messages:
            if message.id == response.obj_ids[-1]['id']:
                return message

    def unarchive(self):
        if False:
            i = 10
            return i + 15
        'Unarchive the conversation.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").unarchive()\n\n        '
        self._reddit.post(API_PATH['modmail_unarchive'].format(id=self.id))

    def unhighlight(self):
        if False:
            return 10
        'Un-highlight the conversation.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").unhighlight()\n\n        '
        self._reddit.delete(API_PATH['modmail_highlight'].format(id=self.id))

    def unmute(self):
        if False:
            print('Hello World!')
        'Unmute the non-mod user associated with the conversation.\n\n        For example:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").modmail("2gmz").unmute()\n\n        '
        self._reddit.request(method='POST', path=API_PATH['modmail_unmute'].format(id=self.id))

    @_deprecate_args('other_conversations')
    def unread(self, *, other_conversations: list[ModmailConversation] | None=None):
        if False:
            return 10
        'Mark the conversation(s) as unread.\n\n        :param other_conversations: A list of other conversations to mark (default:\n            ``None``).\n\n        For example, to mark the conversation as unread along with other recent\n        conversations from the same user:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            conversation = subreddit.modmail.conversation("2gmz")\n            conversation.unread(other_conversations=conversation.user.recent_convos)\n\n        '
        data = {'conversationIds': self._build_conversation_list(other_conversations)}
        self._reddit.post(API_PATH['modmail_unread'], data=data)

class ModmailAction(ModmailObject):
    """A class for moderator actions on modmail conversations."""

class ModmailMessage(ModmailObject):
    """A class for modmail messages."""