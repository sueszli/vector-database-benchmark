"""Python file with invalid syntax, used by scripts/linters/
python_linter_test. This file has a disallowed function usage but
uses a pragma to ignore this invalid call.
"""
from __future__ import annotations
__author__ = 'Something'

class FakeClass:
    """This is a fake docstring for invalid syntax purposes."""

    def __init__(self, fake_arg):
        if False:
            print('Hello World!')
        self.fake_arg = fake_arg

    def fake_method(self, name):
        if False:
            while True:
                i = 10
        "This doesn't do anything.\n        Args:\n            name: str. Means nothing.\n        Yields:\n            tuple(str, str). The argument passed in but twice\n            in a tuple.\n        "
        yield (name, name)