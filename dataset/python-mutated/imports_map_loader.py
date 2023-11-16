"""Import and set up the imports_map."""
import collections
import logging
import os
from typing import Dict, List, Optional, Tuple
from pytype.platform_utils import path_utils
log = logging.getLogger(__name__)
MultimapType = Dict[str, List[str]]
ItemType = Tuple[str, str]
ImportsMapType = Dict[str, str]

class ImportsMapBuilder:
    """Build an imports map from (short_path, path) pairs."""

    def __init__(self, options):
        if False:
            i = 10
            return i + 15
        self.options = options

    def _read_from_file(self, path) -> List[ItemType]:
        if False:
            return 10
        'Read the imports_map file.'
        items = []
        with self.options.open_function(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    (short_path, path) = line.split(' ', 1)
                    items.append((short_path, path))
        return items

    def _build_multimap(self, items: List[ItemType]) -> MultimapType:
        if False:
            while True:
                i = 10
        'Build a multimap from a list of (short_path, path) pairs.'
        imports_multimap = collections.defaultdict(set)
        for (short_path, path) in items:
            (short_path, _) = path_utils.splitext(short_path)
            imports_multimap[short_path].add(path)
        return {short_path: sorted(paths, key=path_utils.basename) for (short_path, paths) in imports_multimap.items()}

    def _validate(self, imports_map: MultimapType) -> List[ItemType]:
        if False:
            while True:
                i = 10
        'Validate the imports map against the command line arguments.\n\n    Args:\n      imports_map: The map returned by _read_imports_map.\n    Returns:\n      A list of invalid entries, in the form (short_path, long_path)\n    '
        errors = []
        for (short_path, paths) in imports_map.items():
            for path in paths:
                if not path_utils.exists(path):
                    errors.append((short_path, path))
        if errors:
            log.error('Invalid imports_map entries (checking from root dir: %s)', path_utils.abspath('.'))
            for (short_path, path) in errors:
                log.error('  file does not exist: %r (mapped from %r)', path, short_path)
        return errors

    def _finalize(self, imports_multimap: MultimapType, path: str='') -> ImportsMapType:
        if False:
            print('Hello World!')
        'Generate the final imports map.'
        for (short_path, paths) in imports_multimap.items():
            if len(paths) > 1:
                log.warning('Multiple files for %r => %r ignoring %r', short_path, paths[0], paths[1:])
        imports_map = {short_path: path_utils.abspath(paths[0]) for (short_path, paths) in imports_multimap.items()}
        errors = self._validate(imports_multimap)
        if errors:
            msg = f'Invalid imports_map: {path}\nBad entries:\n'
            msg += '\n'.join((f'  {k} -> {v}' for (k, v) in errors))
            raise ValueError(msg)
        dir_paths = {}
        for (short_path, full_path) in sorted(imports_map.items()):
            dir_paths[short_path] = full_path
            short_path_pieces = short_path.split(path_utils.sep)
            for i in range(1, len(short_path_pieces)):
                intermediate_dir_init = path_utils.join(*short_path_pieces[:i] + ['__init__'])
                if intermediate_dir_init not in imports_map and intermediate_dir_init not in dir_paths:
                    log.warning('Created empty __init__ %r', intermediate_dir_init)
                    dir_paths[intermediate_dir_init] = os.devnull
        return dir_paths

    def build_from_file(self, path: Optional[str]) -> Optional[ImportsMapType]:
        if False:
            for i in range(10):
                print('nop')
        'Create an ImportsMap from a .imports_info file.\n\n    Builds a dict of short_path to full name\n       (e.g. "path/to/file.py" =>\n             "$GENDIR/rulename~~pytype-gen/path_to_file.py~~pytype"\n    Args:\n      path: The file with the info (may be None, for do-nothing)\n    Returns:\n      Dict of .py short_path to list of .pytd path or None if no path\n    Raises:\n      ValueError if the imports map is invalid\n    '
        if not path:
            return None
        items = self._read_from_file(path)
        return self.build_from_items(items, path)

    def build_from_items(self, items: Optional[List[ItemType]], path=None) -> Optional[ImportsMapType]:
        if False:
            for i in range(10):
                print('nop')
        'Create a file mapping from a list of (short path, path) tuples.\n\n    Builds a dict of short_path to full name\n       (e.g. "path/to/file.py" =>\n             "$GENDIR/rulename~~pytype-gen/path_to_file.py~~pytype"\n    Args:\n      items: A list of (short_path, full_path) tuples.\n      path: The file from which the items were read (for error messages)\n    Returns:\n      Dict of .py short_path to list of .pytd path or None if no items\n    Raises:\n      ValueError if the imports map is invalid\n    '
        if not items:
            return None
        imports_multimap = self._build_multimap(items)
        assert imports_multimap is not None
        return self._finalize(imports_multimap, path)