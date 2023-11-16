"""
haxor
Unofficial Python wrapper for official Hacker News API

@author avinash sajjanshetty
@email hi@avi.im
"""
from __future__ import absolute_import
from __future__ import unicode_literals
import datetime
import json
import sys
import requests
from .settings import supported_api_versions
__all__ = ['User', 'Item', 'HackerNewsApi', 'InvalidAPIVersion', 'InvalidItemID', 'InvalidUserID']

class InvalidItemID(Exception):
    pass

class InvalidUserID(Exception):
    pass

class InvalidAPIVersion(Exception):
    pass

class HTTPError(Exception):
    pass

class HackerNewsApi(object):

    def __init__(self, version='v0'):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            version (string): specifies Hacker News API version. Default is `v0`.\n\n        Raises:\n          InvalidAPIVersion: If Hacker News version is not supported.\n\n        '
        self.session = requests.Session()
        try:
            self.base_url = supported_api_versions[version]
        except KeyError:
            raise InvalidAPIVersion

    def _get(self, url):
        if False:
            return 10
        "Internal method used for GET requests\n\n        Args:\n            url (string): URL to send GET.\n\n        Returns:\n            requests' response object\n\n        Raises:\n          HTTPError: If HTTP request failed.\n\n        "
        response = self.session.get(url)
        if response.status_code == requests.codes.ok:
            return response
        else:
            raise HTTPError

    def _get_page(self, page):
        if False:
            while True:
                i = 10
        return self._get('{0}{1}.json'.format(self.base_url, page))

    def _get_page_param(self, page, param):
        if False:
            return 10
        return self._get('{0}{1}/{2}.json'.format(self.base_url, page, param))

    def get_item(self, item_id):
        if False:
            print('Hello World!')
        'Returns Hacker News `Item` object.\n\n        Args:\n            item_id (int or string): Unique item id of Hacker News story, comment etc.\n\n        Returns:\n            `Item` object representing Hacker News item.\n\n        Raises:\n          InvalidItemID: If corresponding Hacker News story does not exist.\n\n        '
        response = self._get_page_param('item', item_id).json()
        if not response:
            raise InvalidItemID
        return Item(response)

    def get_user(self, user_id):
        if False:
            i = 10
            return i + 15
        'Returns Hacker News `User` object.\n\n        Args:\n            user_id (string): unique user id of a Hacker News user.\n\n        Returns:\n            `User` object representing a user on Hacker News.\n\n        Raises:\n          InvalidUserID: If no such user exists on Hacker News.\n\n        '
        response = self._get_page_param('user', user_id).json()
        if not response:
            raise InvalidUserID
        return User(response)

    def top_stories(self, limit=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns list of item ids of current top stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `list` object containing ids of top stories.\n        '
        return self._get_page('topstories').json()[:limit]

    def new_stories(self, limit=None):
        if False:
            return 10
        'Returns list of item ids of current new stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `list` object containing ids of new stories.\n        '
        return self._get_page('newstories').json()[:limit]

    def ask_stories(self, limit=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns list of item ids of latest Ask HN stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `list` object containing ids of Ask HN stories.\n        '
        return self._get_page('askstories').json()[:limit]

    def best_stories(self, limit=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns list of item ids of best HN stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `list` object containing ids of best stories.\n        '
        return self._get_page('beststories').json()[:limit]

    def show_stories(self, limit=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns list of item ids of latest Show HN stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `list` object containing ids of Show HN stories.\n        '
        return self._get_page('showstories').json()[:limit]

    def job_stories(self, limit=None):
        if False:
            while True:
                i = 10
        'Returns list of item ids of latest Job stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `list` object containing ids of Job stories.\n        '
        return self._get_page('jobstories').json()[:limit]

    def updates(self):
        if False:
            i = 10
            return i + 15
        'Returns list of item ids and user ids that have been\n        changed/updated recently.\n\n        Returns:\n            `dict` with two keys whose values are `list` objects\n        '
        return self._get_page('updates').json()

    def get_max_item(self):
        if False:
            return 10
        'Returns list of item ids of current top stories\n\n        Args:\n            limit (int): specifies the number of stories to be returned.\n\n        Returns:\n            `int` if successful.\n        '
        return self._get_page('maxitem').json()

class Item(object):
    """
    Represents stories, comments, jobs, Ask HNs and polls
    """

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.item_id = data.get('id')
        self.deleted = data.get('deleted')
        self.item_type = data.get('type')
        self.by = data.get('by')
        self.submission_time = datetime.datetime.fromtimestamp(data.get('time', 0))
        self.text = data.get('text')
        self.dead = data.get('dead')
        self.parent = data.get('parent')
        self.kids = data.get('kids')
        self.url = data.get('url')
        self.score = data.get('score')
        self.title = data.get('title')
        self.parts = data.get('parts')
        self.descendants = data.get('descendants')
        self.raw = json.dumps(data)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        retval = '<hackernews.Item: {0} - {1}>'.format(self.item_id, self.title)
        if sys.version_info.major < 3:
            return retval.encode('utf-8', errors='backslashreplace')
        return retval

class User(object):
    """
    Represents a hacker i.e. a user on Hacker News
    """

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.user_id = data.get('id')
        self.delay = data.get('delay')
        self.created = datetime.datetime.fromtimestamp(data.get('created', 0))
        self.karma = data.get('karma')
        self.about = data.get('about')
        self.submitted = data.get('submitted')
        self.raw = json.dumps(data)

    def __repr__(self):
        if False:
            print('Hello World!')
        retval = '<hackernews.User: {0}>'.format(self.user_id)
        if sys.version_info.major < 3:
            return retval.encode('utf-8', errors='backslashreplace')
        return retval