"""
github3.pulls
=============

This module contains all the classes relating to pull requests.

"""
from __future__ import unicode_literals
from re import match
from json import dumps
from . import models
from .repos.contents import Contents
from .repos.commit import RepoCommit
from .users import User
from .decorators import requires_auth
from .issues import Issue
from .issues.comment import IssueComment
from uritemplate import URITemplate

class PullDestination(models.GitHubCore):
    """The :class:`PullDestination <PullDestination>` object.

    See also: http://developer.github.com/v3/pulls/#get-a-single-pull-request
    """

    def __init__(self, dest, direction):
        if False:
            print('Hello World!')
        super(PullDestination, self).__init__(dest)
        self.direction = direction
        self.ref = dest.get('ref')
        self.label = dest.get('label')
        self.user = None
        if dest.get('user'):
            self.user = User(dest.get('user'), None)
        self.sha = dest.get('sha')
        self._repo_name = ''
        self._repo_owner = ''
        if dest.get('repo'):
            self._repo_name = dest['repo'].get('name')
            self._repo_owner = dest['repo']['owner'].get('login')
        self.repo = (self._repo_owner, self._repo_name)

    def _repr(self):
        if False:
            print('Hello World!')
        return '<{0} [{1}]>'.format(self.direction, self.label)

class PullFile(models.GitHubCore):
    """The :class:`PullFile <PullFile>` object.

    See also: http://developer.github.com/v3/pulls/#list-pull-requests-files
    """

    def _update_attributes(self, pfile):
        if False:
            i = 10
            return i + 15
        self.sha = pfile.get('sha')
        self.filename = pfile.get('filename')
        self.status = pfile.get('status')
        self.additions_count = pfile.get('additions')
        self.deletions_count = pfile.get('deletions')
        self.changes_count = pfile.get('changes')
        self.blob_url = pfile.get('blob_url')
        self.raw_url = pfile.get('raw_url')
        self.patch = pfile.get('patch')
        self.contents_url = pfile.get('contents_url')

    def _repr(self):
        if False:
            while True:
                i = 10
        return '<Pull Request File [{0}]>'.format(self.filename)

    def contents(self):
        if False:
            while True:
                i = 10
        'Return the contents of the file.\n\n        :returns: :class:`Contents <github3.repos.contents.Contents>`\n        '
        json = self._json(self._get(self.contents_url), 200)
        return self._instance_or_null(Contents, json)

class PullRequest(models.GitHubCore):
    """The :class:`PullRequest <PullRequest>` object.

    Two pull request instances can be checked like so::

        p1 == p2
        p1 != p2

    And is equivalent to::

        p1.id == p2.id
        p1.id != p2.id

    See also: http://developer.github.com/v3/pulls/
    """

    def _update_attributes(self, pull):
        if False:
            print('Hello World!')
        self._api = pull.get('url', '')
        self.base = PullDestination(pull.get('base'), 'Base')
        self.body = pull.get('body', '')
        self.body_html = pull.get('body_html', '')
        self.body_text = pull.get('body_text', '')
        self.additions_count = pull.get('additions')
        self.deletions_count = pull.get('deletions')
        self.closed_at = self._strptime(pull.get('closed_at'))
        self.comments_count = pull.get('comments')
        self.comments_url = pull.get('comments_url')
        self.commits_count = pull.get('commits')
        self.commits_url = pull.get('commits_url')
        self.created_at = self._strptime(pull.get('created_at'))
        self.diff_url = pull.get('diff_url')
        self.head = PullDestination(pull.get('head'), 'Head')
        self.html_url = pull.get('html_url')
        self.id = pull.get('id')
        self.issue_url = pull.get('issue_url')
        self.statuses_url = pull.get('statuses_url')
        self.links = pull.get('_links')
        self.merged = pull.get('merged')
        self.merged_at = self._strptime(pull.get('merged_at'))
        self.mergeable = pull.get('mergeable', False)
        self.mergeable_state = pull.get('mergeable_state', '')
        user = pull.get('merged_by')
        self.merged_by = User(user, self) if user else None
        self.number = pull.get('number')
        self.patch_url = pull.get('patch_url')
        comments = pull.get('review_comment_url')
        self.review_comment_url = URITemplate(comments) if comments else None
        self.review_comments_count = pull.get('review_comments')
        self.review_comments_url = pull.get('review_comments_url')
        m = match('https?://[\\w\\d\\-\\.\\:]+/(\\S+)/(\\S+)/(?:issues|pull)?/\\d+', self.issue_url)
        self.repository = m.groups()
        self.state = pull.get('state')
        self.title = pull.get('title')
        self.updated_at = self._strptime(pull.get('updated_at'))
        self.user = pull.get('user')
        if self.user:
            self.user = User(self.user, self)
        self.assignee = pull.get('assignee')
        if self.assignee:
            self.assignee = User(self.assignee, self)

    def _repr(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Pull Request [#{0}]>'.format(self.number)

    @requires_auth
    def close(self):
        if False:
            return 10
        'Close this Pull Request without merging.\n\n        :returns: bool\n        '
        return self.update(self.title, self.body, 'closed')

    @requires_auth
    def create_comment(self, body):
        if False:
            return 10
        "Create a comment on this pull request's issue.\n\n        :param str body: (required), comment body\n        :returns: :class:`IssueComment <github3.issues.comment.IssueComment>`\n        "
        url = self.comments_url
        json = None
        if body:
            json = self._json(self._post(url, data={'body': body}), 201)
        return self._instance_or_null(IssueComment, json)

    @requires_auth
    def create_review_comment(self, body, commit_id, path, position):
        if False:
            while True:
                i = 10
        'Create a review comment on this pull request.\n\n        All parameters are required by the GitHub API.\n\n        :param str body: The comment text itself\n        :param str commit_id: The SHA of the commit to comment on\n        :param str path: The relative path of the file to comment on\n        :param int position: The line index in the diff to comment on.\n        :returns: The created review comment.\n        :rtype: :class:`~github3.pulls.ReviewComment`\n        '
        url = self._build_url('comments', base_url=self._api)
        data = {'body': body, 'commit_id': commit_id, 'path': path, 'position': int(position)}
        json = self._json(self._post(url, data=data), 201)
        return self._instance_or_null(ReviewComment, json)

    def diff(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the diff.\n\n        :returns: bytestring representation of the diff.\n        '
        resp = self._get(self._api, headers={'Accept': 'application/vnd.github.diff'})
        return resp.content if self._boolean(resp, 200, 404) else b''

    def is_merged(self):
        if False:
            return 10
        'Check to see if the pull request was merged.\n\n        :returns: bool\n        '
        if self.merged:
            return self.merged
        url = self._build_url('merge', base_url=self._api)
        return self._boolean(self._get(url), 204, 404)

    def issue(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve the issue associated with this pull request.\n\n        :returns: :class:`~github3.issues.Issue`\n        '
        json = self._json(self._get(self.issue_url), 200)
        return self._instance_or_null(Issue, json)

    def commits(self, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over the commits on this pull request.\n\n        :param int number: (optional), number of commits to return. Default:\n            -1 returns all available commits.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of\n            :class:`RepoCommit <github3.repos.commit.RepoCommit>`\\ s\n        '
        url = self._build_url('commits', base_url=self._api)
        return self._iter(int(number), url, RepoCommit, etag=etag)

    def files(self, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over the files associated with this pull request.\n\n        :param int number: (optional), number of files to return. Default:\n            -1 returns all available files.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`PullFile <PullFile>`\\ s\n        '
        url = self._build_url('files', base_url=self._api)
        return self._iter(int(number), url, PullFile, etag=etag)

    def issue_comments(self, number=-1, etag=None):
        if False:
            return 10
        'Iterate over the issue comments on this pull request.\n\n        :param int number: (optional), number of comments to return. Default:\n            -1 returns all available comments.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`IssueComment <IssueComment>`\\ s\n        '
        comments = self.links.get('comments', {})
        url = comments.get('href')
        if not url:
            url = self._build_url('comments', base_url=self._api.replace('pulls', 'issues'))
        return self._iter(int(number), url, IssueComment, etag=etag)

    @requires_auth
    def merge(self, commit_message='', sha=None):
        if False:
            for i in range(10):
                print('nop')
        'Merge this pull request.\n\n        :param str commit_message: (optional), message to be used for the\n            merge commit\n        :returns: bool\n        '
        parameters = {'commit_message': commit_message}
        if sha:
            parameters['sha'] = sha
        url = self._build_url('merge', base_url=self._api)
        json = self._json(self._put(url, data=dumps(parameters)), 200)
        if not json:
            return False
        return json['merged']

    def patch(self):
        if False:
            print('Hello World!')
        'Return the patch.\n\n        :returns: bytestring representation of the patch\n        '
        resp = self._get(self._api, headers={'Accept': 'application/vnd.github.patch'})
        return resp.content if self._boolean(resp, 200, 404) else b''

    @requires_auth
    def reopen(self):
        if False:
            i = 10
            return i + 15
        'Re-open a closed Pull Request.\n\n        :returns: bool\n        '
        return self.update(self.title, self.body, 'open')

    def review_comments(self, number=-1, etag=None):
        if False:
            while True:
                i = 10
        'Iterate over the review comments on this pull request.\n\n        :param int number: (optional), number of comments to return. Default:\n            -1 returns all available comments.\n        :param str etag: (optional), ETag from a previous request to the same\n            endpoint\n        :returns: generator of :class:`ReviewComment <ReviewComment>`\\ s\n        '
        url = self._build_url('comments', base_url=self._api)
        return self._iter(int(number), url, ReviewComment, etag=etag)

    @requires_auth
    def update(self, title=None, body=None, state=None):
        if False:
            while True:
                i = 10
        "Update this pull request.\n\n        :param str title: (optional), title of the pull\n        :param str body: (optional), body of the pull request\n        :param str state: (optional), ('open', 'closed')\n        :returns: bool\n        "
        data = {'title': title, 'body': body, 'state': state}
        json = None
        self._remove_none(data)
        if data:
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
        if json:
            self._update_attributes(json)
            return True
        return False

class ReviewComment(models.BaseComment):
    """The :class:`ReviewComment <ReviewComment>` object.

    This is used to represent comments on pull requests.

    Two comment instances can be checked like so::

        c1 == c2
        c1 != c2

    And is equivalent to::

        c1.id == c2.id
        c1.id != c2.id

    See also: http://developer.github.com/v3/pulls/comments/
    """

    def _update_attributes(self, comment):
        if False:
            return 10
        super(ReviewComment, self)._update_attributes(comment)
        self.user = None
        if comment.get('user'):
            self.user = User(comment.get('user'), self)
        self.original_position = comment.get('original_position')
        self.path = comment.get('path')
        self.position = comment.get('position') or 0
        self.commit_id = comment.get('commit_id')
        self.diff_hunk = comment.get('diff_hunk')
        self.original_commit_id = comment.get('original_commit_id')
        self.pull_request_url = comment.get('pull_request_url')

    def _repr(self):
        if False:
            i = 10
            return i + 15
        return '<Review Comment [{0}]>'.format(self.user.login)

    @requires_auth
    def reply(self, body):
        if False:
            while True:
                i = 10
        'Reply to this review comment with a new review comment.\n\n        :param str body: The text of the comment.\n        :returns: The created review comment.\n        :rtype: :class:`~github3.pulls.ReviewComment`\n        '
        url = self._build_url('comments', base_url=self.pull_request_url)
        index = self._api.rfind('/') + 1
        in_reply_to = self._api[index:]
        json = self._json(self._post(url, data={'body': body, 'in_reply_to': in_reply_to}), 201)
        return self._instance_or_null(ReviewComment, json)