"""Python file with invalid syntax, used by scripts/linters/
python_linter_test. This file is using request() which is not allowed.
"""
from __future__ import annotations
import urllib2

class FakeClass:
    """This is a fake docstring for invalid syntax purposes."""

    def __init__(self, fake_arg):
        if False:
            i = 10
            return i + 15
        self.fake_arg = fake_arg

    def fake_method(self, source_url, data, headers):
        if False:
            return 10
        "This doesn't do anything.\n\n        Args:\n            source_url: str. The URL.\n            data: str. Additional data to send to the server.\n            headers: dict. The request headers.\n\n        Returns:\n            Request(object): Returns Request object.\n        "
        return urllib2.Request(source_url, data, headers)