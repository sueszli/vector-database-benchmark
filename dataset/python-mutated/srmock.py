"""WSGI start_response mock.

This module implements a callable StartResponseMock class that can be
used, along with a mock environ dict, to simulate a WSGI request.
"""
from falcon import util

class StartResponseMock:
    """Mock object representing a WSGI `start_response` callable.

    Attributes:
        call_count (int): Number of times `start_response` was called.
        status (str): HTTP status line, e.g. '785 TPS Cover Sheet
            not attached'.
        headers (list): Raw headers list passed to `start_response`,
            per PEP-333.
        headers_dict (dict): Headers as a case-insensitive
            ``dict``-like object, instead of a ``list``.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._called = 0
        self.status = None
        self.headers = None
        self.exc_info = None

    def __call__(self, status, headers, exc_info=None):
        if False:
            for i in range(10):
                print('nop')
        'Implement the PEP-3333 `start_response` protocol.'
        self._called += 1
        self.status = status
        self.headers = [(name.lower(), value) for (name, value) in headers]
        self.headers_dict = util.CaseInsensitiveDict(headers)
        self.exc_info = exc_info

    @property
    def call_count(self):
        if False:
            for i in range(10):
                print('nop')
        return self._called