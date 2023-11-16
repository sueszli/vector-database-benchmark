"""Python file with invalid syntax, used by scripts/linters/
python_linter_test. This file use __author__ tag which is not allowed.
"""
from __future__ import annotations
__author__ = 'Author Name'

class FakeClass:
    """This is a fake docstring for invalid syntax purposes."""

    def __init__(self, fake_arg):
        if False:
            i = 10
            return i + 15
        self.fake_arg = fake_arg

    def fake_method(self, name):
        if False:
            for i in range(10):
                print('nop')
        "This doesn't do anything.\n\n        Args:\n            name: str. Means nothing.\n\n        Yields:\n            tuple(str, str). The argument passed in but twice in a tuple.\n        "
        yield (name, name)