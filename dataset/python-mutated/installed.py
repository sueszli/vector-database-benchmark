"""Source provider for content which has been installed."""
from __future__ import annotations
import os
from . import SourceProvider

class InstalledSource(SourceProvider):
    """Source provider for content which has been installed."""
    sequence = 0

    @staticmethod
    def is_content_root(path: str) -> bool:
        if False:
            while True:
                i = 10
        'Return True if the given path is a content root for this provider.'
        return False

    def get_paths(self, path: str) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Return the list of available content paths under the given path.'
        paths = []
        kill_extensions = ('.pyc', '.pyo')
        for (root, _dummy, file_names) in os.walk(path):
            rel_root = os.path.relpath(root, path)
            if rel_root == '.':
                rel_root = ''
            paths.extend([os.path.join(rel_root, file_name) for file_name in file_names if not os.path.splitext(file_name)[1] in kill_extensions])
        return paths