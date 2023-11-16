"""Provides the Objector class."""
from __future__ import annotations
from datetime import datetime
from json import loads
from typing import TYPE_CHECKING, Any
from .exceptions import ClientException, RedditAPIException
from .util import snake_case_keys
if TYPE_CHECKING:
    import praw
    from .models.reddit.base import RedditBase

class Objector:
    """The objector builds :class:`.RedditBase` objects."""

    @classmethod
    def check_error(cls, data: list[Any] | dict[str, dict[str, str]]):
        if False:
            i = 10
            return i + 15
        'Raise an error if the argument resolves to an error object.'
        error = cls.parse_error(data)
        if error:
            raise error

    @classmethod
    def parse_error(cls, data: list[Any] | dict[str, dict[str, str]]) -> RedditAPIException | None:
        if False:
            for i in range(10):
                print('nop')
        "Convert JSON response into an error object.\n\n        :param data: The dict to be converted.\n\n        :returns: An instance of :class:`.RedditAPIException`, or ``None`` if ``data``\n            doesn't fit this model.\n\n        "
        if isinstance(data, list):
            return None
        errors = data.get('json', {}).get('errors')
        if errors is None:
            return None
        if len(errors) < 1:
            msg = 'successful error response'
            raise ClientException(msg, data)
        return RedditAPIException(errors)

    def __init__(self, reddit: praw.Reddit, parsers: dict[str, Any] | None=None):
        if False:
            return 10
        'Initialize an :class:`.Objector` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n\n        '
        self.parsers = {} if parsers is None else parsers
        self._reddit = reddit

    def _objectify_dict(self, data: dict[str:Any]) -> RedditBase:
        if False:
            i = 10
            return i + 15
        'Create :class:`.RedditBase` objects from dicts.\n\n        :param data: The structured data, assumed to be a dict.\n\n        :returns: An instance of :class:`.RedditBase`.\n\n        '
        if {'messages', 'modActions'}.issubset(data) and {'conversations', 'conversation'}.intersection(data):
            data.update(data.pop('conversation') if 'conversation' in data else data.pop('conversations'))
            parser = self.parsers['ModmailConversation']
            parser._convert_conversation_objects(data, self._reddit)
        elif {'messages', 'modActions'}.issubset(data) or {'legacyFirstMessageId', 'state'}.issubset(data):
            del data['objIds']
            parser = self.parsers['ModmailConversation']
        elif {'conversationIds', 'conversations', 'messages'}.issubset(data):
            conversations = []
            for conversation_id in data['conversationIds']:
                conversation = data['conversations'][conversation_id]
                if conversation['numMessages'] == len([obj for obj in conversation['objIds'] if obj['key'] == 'messages']):
                    conversation['messages'] = [self.objectify(data['messages'][obj_id['id']]) for obj_id in conversation['objIds']]
                conversations.append(conversation)
            data['conversations'] = conversations
            data = snake_case_keys(data)
            parser = self.parsers['ModmailConversations-list']
        elif {'actionTypeId', 'author', 'date'}.issubset(data):
            data = snake_case_keys(data)
            parser = self.parsers['ModmailAction']
        elif {'bodyMarkdown', 'isInternal'}.issubset(data):
            data = snake_case_keys(data)
            parser = self.parsers['ModmailMessage']
        elif {'kind', 'short_name', 'violation_reason'}.issubset(data):
            parser = self.parsers['rule']
        elif {'isAdmin', 'isDeleted'}.issubset(data):
            data = snake_case_keys(data)
            del data['id']
            data['is_subreddit_mod'] = data.pop('is_mod')
            parser = self.parsers[self._reddit.config.kinds['redditor']]
        elif {'banStatus', 'muteStatus', 'recentComments'}.issubset(data):
            data = snake_case_keys(data)
            data['created_string'] = data.pop('created')
            parser = self.parsers[self._reddit.config.kinds['redditor']]
        elif {'displayName', 'id', 'type'}.issubset(data):
            data = snake_case_keys(data)
            parser = self.parsers[self._reddit.config.kinds[data['type']]]
        elif {'date', 'id', 'name'}.issubset(data) or {'id', 'name', 'permissions'}.issubset(data):
            parser = self.parsers[self._reddit.config.kinds['redditor']]
        elif {'text', 'url'}.issubset(data):
            if 'color' in data or 'linkUrl' in data:
                parser = self.parsers['Button']
            else:
                parser = self.parsers['MenuLink']
        elif {'children', 'text'}.issubset(data):
            parser = self.parsers['Submenu']
        elif {'height', 'url', 'width'}.issubset(data):
            parser = self.parsers['Image']
        elif {'isSubscribed', 'name', 'subscribers'}.issubset(data):
            return self._reddit.subreddit(data['name'])
        elif {'authorFlairType', 'name'}.issubset(data):
            return self._reddit.redditor(data['name'])
        elif {'parent_id'}.issubset(data):
            parser = self.parsers[self._reddit.config.kinds['comment']]
        elif 'collection_id' in data:
            parser = self.parsers['Collection']
        elif {'moderators', 'moderatorIds', 'allUsersLoaded', 'subredditId'}.issubset(data):
            data = snake_case_keys(data)
            moderators = []
            for mod_id in data['moderator_ids']:
                mod = snake_case_keys(data['moderators'][mod_id])
                mod['mod_permissions'] = list(mod['mod_permissions'].keys())
                moderators.append(mod)
            data['moderators'] = moderators
            parser = self.parsers['moderator-list']
        elif 'username' in data:
            data['name'] = data.pop('username')
            parser = self.parsers[self._reddit.config.kinds['redditor']]
        elif {'mod_permissions', 'name', 'sr', 'subscribers'}.issubset(data):
            data['display_name'] = data['sr']
            parser = self.parsers[self._reddit.config.kinds['subreddit']]
        elif {'drafts', 'subreddits'}.issubset(data):
            subreddit_parser = self.parsers[self._reddit.config.kinds['subreddit']]
            user_subreddit_parser = self.parsers['UserSubreddit']
            subreddits = {subreddit['name']: user_subreddit_parser.parse(subreddit, self._reddit) if subreddit['display_name_prefixed'].startswith('u/') else subreddit_parser.parse(subreddit, self._reddit) for subreddit in data.pop('subreddits')}
            for draft in data['drafts']:
                if draft['subreddit']:
                    draft['subreddit'] = subreddits[draft['subreddit']]
                draft['modified'] = datetime.fromtimestamp(draft['modified'] / 1000).astimezone()
            parser = self.parsers['DraftList']
        elif {'mod_action_data', 'user_note_data'}.issubset(data):
            data['moderator'] = self._reddit.redditor(data['operator'])
            data['subreddit'] = self._reddit.subreddit(data['subreddit'])
            data['user'] = self._reddit.redditor(data['user'])
            data.update(data['mod_action_data'])
            del data['mod_action_data']
            data.update(data['user_note_data'])
            del data['user_note_data']
            parser = self.parsers['mod_note']
        elif 'created' in data and isinstance(data['created'], dict) and {'mod_action_data', 'user_note_data'}.issubset(data['created']):
            data = data['created']
            return self._objectify_dict(data)
        else:
            if 'user' in data:
                parser = self.parsers[self._reddit.config.kinds['redditor']]
                data['user'] = parser.parse({'name': data['user']}, self._reddit)
            return data
        return parser.parse(data, self._reddit)

    def objectify(self, data: dict[str, Any] | list[Any] | bool | None) -> RedditBase | dict[str, Any] | list[Any] | bool | None:
        if False:
            print('Hello World!')
        'Create :class:`.RedditBase` objects from data.\n\n        :param data: The structured data.\n\n        :returns: An instance of :class:`.RedditBase`, or ``None`` if given ``data`` is\n            ``None``.\n\n        '
        if data is None:
            return None
        if isinstance(data, list):
            return [self.objectify(item) for item in data]
        if isinstance(data, bool):
            return data
        if 'json' in data and 'errors' in data['json']:
            errors = data['json']['errors']
            if len(errors) > 0:
                raise RedditAPIException(errors)
        if 'kind' in data and ('shortName' in data or data['kind'] in ('menu', 'moderators')):
            parser = self.parsers.get(data['kind'], self.parsers['widget'])
            return parser.parse(data, self._reddit)
        if {'kind', 'data'}.issubset(data) and data['kind'] in self.parsers:
            parser = self.parsers[data['kind']]
            if data['kind'] == 'ModeratedList':
                return parser.parse(data, self._reddit)
            return parser.parse(data['data'], self._reddit)
        if 'json' in data and 'data' in data['json']:
            if 'websocket_url' in data['json']['data']:
                return data
            if 'things' in data['json']['data']:
                return self.objectify(data['json']['data']['things'])
            if 'rules' in data['json']['data']:
                return self.objectify(loads(data['json']['data']['rules']))
            if 'drafts_count' in data['json']['data'] and all((key not in data['json']['data'] for key in ['name', 'url'])):
                data['json']['data'].pop('drafts_count')
                return self.parsers['Draft'].parse(data['json']['data'], self._reddit)
            if 'url' in data['json']['data']:
                del data['json']['data']['url']
                parser = self.parsers[self._reddit.config.kinds['submission']]
                if data['json']['data']['id'].startswith(f"{self._reddit.config.kinds['submission']}_"):
                    data['json']['data']['id'] = data['json']['data']['id'].split('_', 1)[1]
            else:
                parser = self.parsers['LiveUpdateEvent']
            return parser.parse(data['json']['data'], self._reddit)
        if {'is_public_link', 'title', 'body'}.issubset(data):
            parser = self.parsers['Draft']
            return parser.parse(data, self._reddit)
        if 'rules' in data:
            return self.objectify(data['rules'])
        if isinstance(data, dict):
            return self._objectify_dict(data)
        return data