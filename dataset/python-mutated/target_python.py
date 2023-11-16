import sys
from typing import List, Optional, Set, Tuple
from pip._vendor.packaging.tags import Tag
from pip._internal.utils.compatibility_tags import get_supported, version_info_to_nodot
from pip._internal.utils.misc import normalize_version_info

class TargetPython:
    """
    Encapsulates the properties of a Python interpreter one is targeting
    for a package install, download, etc.
    """
    __slots__ = ['_given_py_version_info', 'abis', 'implementation', 'platforms', 'py_version', 'py_version_info', '_valid_tags', '_valid_tags_set']

    def __init__(self, platforms: Optional[List[str]]=None, py_version_info: Optional[Tuple[int, ...]]=None, abis: Optional[List[str]]=None, implementation: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        :param platforms: A list of strings or None. If None, searches for\n            packages that are supported by the current system. Otherwise, will\n            find packages that can be built on the platforms passed in. These\n            packages will only be downloaded for distribution: they will\n            not be built locally.\n        :param py_version_info: An optional tuple of ints representing the\n            Python version information to use (e.g. `sys.version_info[:3]`).\n            This can have length 1, 2, or 3 when provided.\n        :param abis: A list of strings or None. This is passed to\n            compatibility_tags.py's get_supported() function as is.\n        :param implementation: A string or None. This is passed to\n            compatibility_tags.py's get_supported() function as is.\n        "
        self._given_py_version_info = py_version_info
        if py_version_info is None:
            py_version_info = sys.version_info[:3]
        else:
            py_version_info = normalize_version_info(py_version_info)
        py_version = '.'.join(map(str, py_version_info[:2]))
        self.abis = abis
        self.implementation = implementation
        self.platforms = platforms
        self.py_version = py_version
        self.py_version_info = py_version_info
        self._valid_tags: Optional[List[Tag]] = None
        self._valid_tags_set: Optional[Set[Tag]] = None

    def format_given(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Format the given, non-None attributes for display.\n        '
        display_version = None
        if self._given_py_version_info is not None:
            display_version = '.'.join((str(part) for part in self._given_py_version_info))
        key_values = [('platforms', self.platforms), ('version_info', display_version), ('abis', self.abis), ('implementation', self.implementation)]
        return ' '.join((f'{key}={value!r}' for (key, value) in key_values if value is not None))

    def get_sorted_tags(self) -> List[Tag]:
        if False:
            print('Hello World!')
        '\n        Return the supported PEP 425 tags to check wheel candidates against.\n\n        The tags are returned in order of preference (most preferred first).\n        '
        if self._valid_tags is None:
            py_version_info = self._given_py_version_info
            if py_version_info is None:
                version = None
            else:
                version = version_info_to_nodot(py_version_info)
            tags = get_supported(version=version, platforms=self.platforms, abis=self.abis, impl=self.implementation)
            self._valid_tags = tags
        return self._valid_tags

    def get_unsorted_tags(self) -> Set[Tag]:
        if False:
            return 10
        'Exactly the same as get_sorted_tags, but returns a set.\n\n        This is important for performance.\n        '
        if self._valid_tags_set is None:
            self._valid_tags_set = set(self.get_sorted_tags())
        return self._valid_tags_set