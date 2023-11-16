"""A Python module for interacting and consuming responses from Slack."""
import asyncio
import logging
from typing import Union
import slack_sdk.errors as e

class LegacySlackResponse(object):
    """An iterable container of response data.

    Attributes:
        data (dict): The json-encoded content of the response. Along
            with the headers and status code information.

    Methods:
        validate: Check if the response from Slack was successful.
        get: Retrieves any key from the response data.
        next: Retrieves the next portion of results,
            if 'next_cursor' is present.

    Example:
    ```python
    import os
    import slack

    client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'])

    response1 = client.auth_revoke(test='true')
    assert not response1['revoked']

    response2 = client.auth_test()
    assert response2.get('ok', False)

    users = []
    for page in client.users_list(limit=2):
        TODO: This example should specify when to break.
        users = users + page['members']
    ```

    Note:
        Some responses return collections of information
        like channel and user lists. If they do it's likely
        that you'll only receive a portion of results. This
        object allows you to iterate over the response which
        makes subsequent API requests until your code hits
        'break' or there are no more results to be found.

        Any attributes or methods prefixed with _underscores are
        intended to be "private" internal use only. They may be changed or
        removed at anytime.
    """

    def __init__(self, *, client, http_verb: str, api_url: str, req_args: dict, data: Union[dict, bytes], headers: dict, status_code: int, use_sync_aiohttp: bool=True):
        if False:
            print('Hello World!')
        self.http_verb = http_verb
        self.api_url = api_url
        self.req_args = req_args
        self.data = data
        self.headers = headers
        self.status_code = status_code
        self._initial_data = data
        self._client = client
        self._use_sync_aiohttp = use_sync_aiohttp
        self._logger = logging.getLogger(__name__)

    def __str__(self):
        if False:
            print('Hello World!')
        'Return the Response data if object is converted to a string.'
        if isinstance(self.data, bytes):
            raise ValueError('As the response.data is binary data, this operation is unsupported')
        return f'{self.data}'

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Retrieves any key from the data store.\n\n        Note:\n            This is implemented so users can reference the\n            SlackResponse object like a dictionary.\n            e.g. response["ok"]\n\n        Returns:\n            The value from data or None.\n        '
        if isinstance(self.data, bytes):
            raise ValueError('As the response.data is binary data, this operation is unsupported')
        return self.data.get(key, None)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        "Enables the ability to iterate over the response.\n        It's required for the iterator protocol.\n\n        Note:\n            This enables Slack cursor-based pagination.\n\n        Returns:\n            (SlackResponse) self\n        "
        if isinstance(self.data, bytes):
            raise ValueError('As the response.data is binary data, this operation is unsupported')
        self._iteration = 0
        self.data = self._initial_data
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        "Retrieves the next portion of results, if 'next_cursor' is present.\n\n        Note:\n            Some responses return collections of information\n            like channel and user lists. If they do it's likely\n            that you'll only receive a portion of results. This\n            method allows you to iterate over the response until\n            your code hits 'break' or there are no more results\n            to be found.\n\n        Returns:\n            (SlackResponse) self\n                With the new response data now attached to this object.\n\n        Raises:\n            SlackApiError: If the request to the Slack API failed.\n            StopIteration: If 'next_cursor' is not present or empty.\n        "
        if isinstance(self.data, bytes):
            raise ValueError('As the response.data is binary data, this operation is unsupported')
        self._iteration += 1
        if self._iteration == 1:
            return self
        if self._next_cursor_is_present(self.data):
            params = self.req_args.get('params', {})
            if params is None:
                params = {}
            params.update({'cursor': self.data['response_metadata']['next_cursor']})
            self.req_args.update({'params': params})
            if self._use_sync_aiohttp:
                response = asyncio.get_event_loop().run_until_complete(self._client._request(http_verb=self.http_verb, api_url=self.api_url, req_args=self.req_args))
            else:
                response = self._client._request_for_pagination(api_url=self.api_url, req_args=self.req_args)
            self.data = response['data']
            self.headers = response['headers']
            self.status_code = response['status_code']
            return self.validate()
        else:
            raise StopIteration

    def get(self, key, default=None):
        if False:
            i = 10
            return i + 15
        'Retrieves any key from the response data.\n\n        Note:\n            This is implemented so users can reference the\n            SlackResponse object like a dictionary.\n            e.g. response.get("ok", False)\n\n        Returns:\n            The value from data or the specified default.\n        '
        if isinstance(self.data, bytes):
            raise ValueError('As the response.data is binary data, this operation is unsupported')
        return self.data.get(key, default)

    def validate(self):
        if False:
            for i in range(10):
                print('nop')
        "Check if the response from Slack was successful.\n\n        Returns:\n            (SlackResponse)\n                This method returns it's own object. e.g. 'self'\n\n        Raises:\n            SlackApiError: The request to the Slack API failed.\n        "
        if self._logger.level <= logging.DEBUG:
            body = self.data if isinstance(self.data, dict) else '(binary)'
            self._logger.debug(f'Received the following response - status: {self.status_code}, headers: {dict(self.headers)}, body: {body}')
        if self.status_code == 200 and self.data and (isinstance(self.data, bytes) or self.data.get('ok', False)):
            return self
        msg = 'The request to the Slack API failed.'
        raise e.SlackApiError(message=msg, response=self)

    @staticmethod
    def _next_cursor_is_present(data):
        if False:
            for i in range(10):
                print('nop')
        "Determine if the response contains 'next_cursor'\n        and 'next_cursor' is not empty.\n\n        Returns:\n            A boolean value.\n        "
        present = 'response_metadata' in data and 'next_cursor' in data['response_metadata'] and (data['response_metadata']['next_cursor'] != '')
        return present