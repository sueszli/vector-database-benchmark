"""
github3.auths
=============

This module contains the Authorization object.

"""
from __future__ import unicode_literals
from .decorators import requires_basic_auth
from .models import GitHubCore

class Authorization(GitHubCore):
    """The :class:`Authorization <Authorization>` object.

    Two authorization instances can be checked like so::

        a1 == a2
        a1 != a2

    And is equivalent to::

        a1.id == a2.id
        a1.id != a2.id

    See also: http://developer.github.com/v3/oauth/#oauth-authorizations-api

    """

    def _update_attributes(self, auth):
        if False:
            while True:
                i = 10
        self._api = auth.get('url')
        self.app = auth.get('app', {})
        self.token = auth.get('token', '')
        self.name = self.app.get('name', '')
        self.note_url = auth.get('note_url') or ''
        self.note = auth.get('note') or ''
        self.scopes = auth.get('scopes', [])
        self.id = auth.get('id', 0)
        self.created_at = self._strptime(auth.get('created_at'))
        self.updated_at = self._strptime(auth.get('updated_at'))

    def _repr(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Authorization [{0}]>'.format(self.name)

    def _update(self, scopes_data, note, note_url):
        if False:
            i = 10
            return i + 15
        'Helper for add_scopes, replace_scopes, remove_scopes.'
        if note is not None:
            scopes_data['note'] = note
        if note_url is not None:
            scopes_data['note_url'] = note_url
        json = self._json(self._post(self._api, data=scopes_data), 200)
        if json:
            self._update_attributes(json)
            return True
        return False

    @requires_basic_auth
    def add_scopes(self, scopes, note=None, note_url=None):
        if False:
            i = 10
            return i + 15
        'Adds the scopes to this authorization.\n\n        .. versionadded:: 1.0\n\n        :param list scopes: Adds these scopes to the ones present on this\n            authorization\n        :param str note: (optional), Note about the authorization\n        :param str note_url: (optional), URL to link to when the user views\n            the authorization\n        :returns: True if successful, False otherwise\n        :rtype: bool\n        '
        return self._update({'add_scopes': scopes}, note, note_url)

    @requires_basic_auth
    def delete(self):
        if False:
            return 10
        'Delete this authorization.'
        return self._boolean(self._delete(self._api), 204, 404)

    @requires_basic_auth
    def remove_scopes(self, scopes, note=None, note_url=None):
        if False:
            print('Hello World!')
        'Remove the scopes from this authorization.\n\n        .. versionadded:: 1.0\n\n        :param list scopes: Remove these scopes from the ones present on this\n            authorization\n        :param str note: (optional), Note about the authorization\n        :param str note_url: (optional), URL to link to when the user views\n            the authorization\n        :returns: True if successful, False otherwise\n        :rtype: bool\n        '
        return self._update({'rm_scopes': scopes}, note, note_url)

    @requires_basic_auth
    def replace_scopes(self, scopes, note=None, note_url=None):
        if False:
            print('Hello World!')
        'Replace the scopes on this authorization.\n\n        .. versionadded:: 1.0\n\n        :param list scopes: Use these scopes instead of the previous list\n        :param str note: (optional), Note about the authorization\n        :param str note_url: (optional), URL to link to when the user views\n            the authorization\n        :returns: True if successful, False otherwise\n        :rtype: bool\n        '
        return self._update({'scopes': scopes}, note, note_url)