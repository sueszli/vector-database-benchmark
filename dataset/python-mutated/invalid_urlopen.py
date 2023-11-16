"""Python file with invalid syntax, used by scripts/linters/
python_linter_test. This file is using urlopen which is not allowed.
"""
from __future__ import annotations
import urllib2

class FakeClass:
    """This is a fake docstring for invalid syntax purposes."""

    def __init__(self, fake_arg):
        if False:
            for i in range(10):
                print('nop')
        self.fake_arg = fake_arg

    def fake_method(self, source_url):
        if False:
            i = 10
            return i + 15
        "This doesn't do anything.\n\n        Args:\n            source_url: str. The URL.\n\n        Returns:\n            urlopen(object): Returns urlopen object.\n        "
        return urllib2.urlopen(source_url)