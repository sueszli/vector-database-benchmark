"""Provides a Python interface to various parts of the GitHub API.

NOTE: not using PyGithub https://github.com/PyGithub/PyGithub as it doesn't
have full API coverage and it's easy enough to use the endpoints we need like
this (with zero dependencies as a bonus!)
"""
from typing import Any, Optional
import urllib
import requests

class GitHubAPI:
    """Wraps the GitHub REST API."""

    def __init__(self, token: Optional[str]=None):
        if False:
            print('Hello World!')
        self._session = requests.Session()
        self._session.headers['Accept'] = 'application/vnd.github+json'
        if token:
            self._session.headers['Authorization'] = f'token {token}'

    def _make_request(self, verb: str, endpoint: str, **kwargs: dict[str, Any]) -> requests.Response:
        if False:
            while True:
                i = 10
        'Helper method to make a request and raise an HTTPError if one occurred.\n\n    Arguments:\n      verb: The HTTP verb to use\n      endpoint: The endpoint to make the request to\n      **kwargs: The json that will be sent as the body of the request.\n\n    Returns:\n      a requests.Response object containing the response from the API.\n\n    Raises:\n      requests.exceptions.HTTPError\n    '
        res = self._session.request(verb, urllib.parse.urljoin('https://api.github.com', endpoint), json=kwargs)
        res.raise_for_status()
        return res.json()

    def get_commit(self, repo: str, commit_id: str) -> requests.Response:
        if False:
            while True:
                i = 10
        "Gets a commit by it's SHA-1 hash.\n\n    https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#get-a-\n    commit\n\n    Arguments:\n      repo: a string of the form `owner/repo_name`, e.g. openxla/xla.\n      commit_id: a string describing the commit to get, e.g. `deadbeef` or\n        `HEAD`.\n\n    Returns:\n      a requests.Response object containing the response from the API.\n\n    Raises:\n      requests.exceptions.HTTPError\n    "
        endpoint = f'repos/{repo}/commits/{commit_id}'
        return self._make_request('GET', endpoint)

    def write_issue_comment(self, repo: str, issue_number: int, body: str) -> requests.Response:
        if False:
            while True:
                i = 10
        'Writes a comment on an issue (or PR).\n\n    https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-\n    28#create-an-issue-comment\n\n    Arguments:\n      repo: a string of the form `owner/repo_name`, e.g. openxla/xla\n      issue_number: the issue (or PR) to comment on\n      body: the body of the comment\n\n    Returns:\n      a requests.Response object containing the response from the API.\n\n    Raises:\n      requests.exceptions.HTTPError\n    '
        endpoint = f'repos/{repo}/issues/{issue_number}/comments'
        return self._make_request('POST', endpoint, body=body)

    def set_issue_status(self, repo: str, issue_number: int, status: str) -> requests.Response:
        if False:
            while True:
                i = 10
        'Sets the status of an issue (or PR).\n\n    https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#update-\n    an-issue\n\n    Arguments:\n      repo: a string of the form `owner/repo_name`, e.g. openxla/xla\n      issue_number: the issue (or PR) to set the status of\n      status: the status to set\n\n    Returns:\n      a requests.Response object containing the response from the API.\n\n    Raises:\n      requests.exceptions.HTTPError\n    '
        endpoint = f'repos/{repo}/issues/{issue_number}'
        return self._make_request('POST', endpoint, status=status)