"""
github3.gists.gist
==================

This module contains the Gist class alone for simplicity.

"""
from __future__ import unicode_literals
from json import dumps
from ..models import GitHubCore
from ..decorators import requires_auth
from .comment import GistComment
from .file import GistFile
from .history import GistHistory
from ..users import User

class Gist(GitHubCore):
    """This object holds all the information returned by Github about a gist.

    With it you can comment on or fork the gist (assuming you are
    authenticated), edit or delete the gist (assuming you own it).  You can
    also "star" or "unstar" the gist (again assuming you have authenticated).

    Two gist instances can be checked like so::

        g1 == g2
        g1 != g2

    And is equivalent to::

        g1.id == g2.id
        g1.id != g2.id

    See also: http://developer.github.com/v3/gists/

    """

    def _update_attributes(self, data):
        if False:
            i = 10
            return i + 15
        self.comments_count = data.get('comments', 0)
        self.id = '{0}'.format(data.get('id', ''))
        self.description = data.get('description', '')
        self._api = data.get('url', '')
        self.html_url = data.get('html_url')
        self.public = data.get('public')
        self._forks = data.get('forks', [])
        self.git_pull_url = data.get('git_pull_url', '')
        self.git_push_url = data.get('git_push_url', '')
        self.created_at = self._strptime(data.get('created_at'))
        self.updated_at = self._strptime(data.get('updated_at'))
        owner = data.get('owner')
        self.owner = User(owner, self) if owner else None
        self._files = [GistFile(data['files'][f]) for f in data['files']]
        self.history = [GistHistory(h, self) for h in data.get('history', [])]
        self.comments_url = data.get('comments_url', '')
        self.commits_url = data.get('commits_url', '')
        self.forks_url = data.get('forks_url', '')
        self.truncated = data.get('truncated')

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.id

    def _repr(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Gist [{0}]>'.format(self.id)

    @requires_auth
    def create_comment(self, body):
        if False:
            print('Hello World!')
        'Create a comment on this gist.\n\n        :param str body: (required), body of the comment\n        :returns: :class:`GistComment <github3.gists.comment.GistComment>`\n\n        '
        json = None
        if body:
            url = self._build_url('comments', base_url=self._api)
            json = self._json(self._post(url, data={'body': body}), 201)
        return self._instance_or_null(GistComment, json)

    @requires_auth
    def delete(self):
        if False:
            i = 10
            return i + 15
        'Delete this gist.\n\n        :returns: bool -- whether the deletion was successful\n\n        '
        return self._boolean(self._delete(self._api), 204, 404)

    @requires_auth
    def edit(self, description='', files={}):
        if False:
            print('Hello World!')
        "Edit this gist.\n\n        :param str description: (optional), description of the gist\n        :param dict files: (optional), files that make up this gist; the\n            key(s) should be the file name(s) and the values should be another\n            (optional) dictionary with (optional) keys: 'content' and\n            'filename' where the former is the content of the file and the\n            latter is the new name of the file.\n        :returns: bool -- whether the edit was successful\n\n        "
        data = {}
        json = None
        if description:
            data['description'] = description
        if files:
            data['files'] = files
        if data:
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
        if json:
            self._update_attributes(json)
            return True
        return False

    @requires_auth
    def fork(self):
        if False:
            for i in range(10):
                print('nop')
        'Fork this gist.\n\n        :returns: :class:`Gist <Gist>` if successful, ``None`` otherwise\n\n        '
        url = self._build_url('forks', base_url=self._api)
        json = self._json(self._post(url), 201)
        return self._instance_or_null(Gist, json)

    @requires_auth
    def is_starred(self):
        if False:
            while True:
                i = 10
        'Check to see if this gist is starred by the authenticated user.\n\n        :returns: bool -- True if it is starred, False otherwise\n\n        '
        url = self._build_url('star', base_url=self._api)
        return self._boolean(self._get(url), 204, 404)

    def comments(self, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over comments on this gist.\n\n        :param int number: (optional), number of comments to iterate over.\n            Default: -1 will iterate over all comments on the gist\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of\n            :class:`GistComment <github3.gists.comment.GistComment>`\n\n        '
        url = self._build_url('comments', base_url=self._api)
        return self._iter(int(number), url, GistComment, etag=etag)

    def commits(self, number=-1, etag=None):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over the commits on this gist.\n\n        These commits will be requested from the API and should be the same as\n        what is in ``Gist.history``.\n\n        .. versionadded:: 0.6\n\n        .. versionchanged:: 0.9\n\n            Added param ``etag``.\n\n        :param int number: (optional), number of commits to iterate over.\n            Default: -1 will iterate over all commits associated with this\n            gist.\n        :param str etag: (optional), ETag from a previous request to this\n            endpoint.\n        :returns: generator of\n            :class:`GistHistory <github3.gists.history.GistHistory>`\n\n        '
        url = self._build_url('commits', base_url=self._api)
        return self._iter(int(number), url, GistHistory)

    def files(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterator over the files stored in this gist.\n\n        :returns: generator of :class`GistFile <github3.gists.file.GistFile>`\n\n        '
        return iter(self._files)

    def forks(self, number=-1, etag=None):
        if False:
            for i in range(10):
                print('nop')
        'Iterator of forks of this gist.\n\n        .. versionchanged:: 0.9\n\n            Added params ``number`` and ``etag``.\n\n        :param int number: (optional), number of forks to iterate over.\n            Default: -1 will iterate over all forks of this gist.\n        :param str etag: (optional), ETag from a previous request to this\n            endpoint.\n        :returns: generator of :class:`Gist <Gist>`\n\n        '
        url = self._build_url('forks', base_url=self._api)
        return self._iter(int(number), url, Gist, etag=etag)

    @requires_auth
    def star(self):
        if False:
            for i in range(10):
                print('nop')
        'Star this gist.\n\n        :returns: bool -- True if successful, False otherwise\n\n        '
        url = self._build_url('star', base_url=self._api)
        return self._boolean(self._put(url), 204, 404)

    @requires_auth
    def unstar(self):
        if False:
            while True:
                i = 10
        'Un-star this gist.\n\n        :returns: bool -- True if successful, False otherwise\n\n        '
        url = self._build_url('star', base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)