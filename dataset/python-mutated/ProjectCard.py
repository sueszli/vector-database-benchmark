from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Issue
import github.NamedUser
import github.ProjectColumn
import github.PullRequest
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt

class ProjectCard(CompletableGithubObject):
    """
    This class represents Project Cards. The reference can be found here https://docs.github.com/en/rest/reference/projects#cards
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._archived: Attribute[bool] = NotSet
        self._column_url: Attribute[str] = NotSet
        self._content_url: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._creator: Attribute[github.NamedUser.NamedUser] = NotSet
        self._id: Attribute[int] = NotSet
        self._node_id: Attribute[str] = NotSet
        self._note: Attribute[str] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'id': self._id.value})

    @property
    def archived(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._archived.value

    @property
    def column_url(self) -> str:
        if False:
            while True:
                i = 10
        return self._column_url.value

    @property
    def content_url(self) -> str:
        if False:
            return 10
        return self._content_url.value

    @property
    def created_at(self) -> datetime:
        if False:
            print('Hello World!')
        return self._created_at.value

    @property
    def creator(self) -> github.NamedUser.NamedUser:
        if False:
            return 10
        return self._creator.value

    @property
    def id(self) -> int:
        if False:
            while True:
                i = 10
        return self._id.value

    @property
    def node_id(self) -> str:
        if False:
            while True:
                i = 10
        return self._node_id.value

    @property
    def note(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._note.value

    @property
    def updated_at(self) -> datetime:
        if False:
            i = 10
            return i + 15
        return self._updated_at.value

    @property
    def url(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._url.value

    def get_content(self, content_type: Opt[str]=NotSet) -> github.PullRequest.PullRequest | github.Issue.Issue | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :calls: `GET /repos/{owner}/{repo}/pulls/{number} <https://docs.github.com/en/rest/reference/pulls#get-a-pull-request>`_\n        '
        assert content_type is NotSet or isinstance(content_type, str), content_type
        if self.content_url is None:
            return None
        retclass: type[github.PullRequest.PullRequest] | type[github.Issue.Issue]
        if content_type == 'PullRequest':
            url = self.content_url.replace('issues', 'pulls')
            retclass = github.PullRequest.PullRequest
        elif content_type is NotSet or content_type == 'Issue':
            url = self.content_url
            retclass = github.Issue.Issue
        else:
            raise ValueError(f'Unknown content type: {content_type}')
        (headers, data) = self._requester.requestJsonAndCheck('GET', url)
        return retclass(self._requester, headers, data, completed=True)

    def move(self, position: str, column: github.ProjectColumn.ProjectColumn | int) -> bool:
        if False:
            print('Hello World!')
        '\n        :calls: `POST /projects/columns/cards/{card_id}/moves <https://docs.github.com/en/rest/reference/projects#cards>`_\n        '
        assert isinstance(position, str), position
        assert isinstance(column, github.ProjectColumn.ProjectColumn) or isinstance(column, int), column
        post_parameters = {'position': position, 'column_id': column.id if isinstance(column, github.ProjectColumn.ProjectColumn) else column}
        (status, _, _) = self._requester.requestJson('POST', f'{self.url}/moves', input=post_parameters, headers={'Accept': Consts.mediaTypeProjectsPreview})
        return status == 201

    def delete(self) -> bool:
        if False:
            return 10
        '\n        :calls: `DELETE /projects/columns/cards/{card_id} <https://docs.github.com/en/rest/reference/projects#cards>`_\n        '
        (status, _, _) = self._requester.requestJson('DELETE', self.url, headers={'Accept': Consts.mediaTypeProjectsPreview})
        return status == 204

    def edit(self, note: Opt[str]=NotSet, archived: Opt[bool]=NotSet) -> None:
        if False:
            i = 10
            return i + 15
        '\n        :calls: `PATCH /projects/columns/cards/{card_id} <https://docs.github.com/en/rest/reference/projects#cards>`_\n        '
        assert note is NotSet or isinstance(note, str), note
        assert archived is NotSet or isinstance(archived, bool), archived
        patch_parameters: dict[str, Any] = NotSet.remove_unset_items({'note': note, 'archived': archived})
        (headers, data) = self._requester.requestJsonAndCheck('PATCH', self.url, input=patch_parameters, headers={'Accept': Consts.mediaTypeProjectsPreview})
        self._useAttributes(data)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'archived' in attributes:
            self._archived = self._makeBoolAttribute(attributes['archived'])
        if 'column_url' in attributes:
            self._column_url = self._makeStringAttribute(attributes['column_url'])
        if 'content_url' in attributes:
            self._content_url = self._makeStringAttribute(attributes['content_url'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'creator' in attributes:
            self._creator = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['creator'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'node_id' in attributes:
            self._node_id = self._makeStringAttribute(attributes['node_id'])
        if 'note' in attributes:
            self._note = self._makeStringAttribute(attributes['note'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])