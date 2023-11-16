"""Module that is off sys.path by default, for testing local-plugin-paths."""
from __future__ import annotations

class ExtensionTestPlugin2:
    """Extension test plugin in its own directory."""

    def __init__(self, tree):
        if False:
            for i in range(10):
                print('nop')
        'Construct an instance of test plugin.'

    def run(self):
        if False:
            while True:
                i = 10
        'Do nothing.'