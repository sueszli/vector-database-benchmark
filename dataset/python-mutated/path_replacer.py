import re
from abc import ABC, abstractmethod
from typing import Set

class PathReplacer(ABC):

    @abstractmethod
    def replace_path(self, path: str) -> str:
        if False:
            i = 10
            return i + 15
        pass

class FixedPathReplacer(PathReplacer):

    def __init__(self, path: str):
        if False:
            return 10
        self.path = path

    def replace_path(self, path: str) -> str:
        if False:
            print('Hello World!')
        return self.path

class KeepAfterPatternMatchPathReplacer(PathReplacer):
    """
    Replaces the path with the part of the path after the first pattern match.

    For example, if the pattern is `/sentry/.*` and the path is `/Users/sentry/myfile.js` then the path will be replaced with `/sentry/myfile.js`.

    :param patterns: A set of regular expressions.
    :param fallback_path: The path to use if no pattern matches.
    """

    def __init__(self, patterns: Set[str], fallback_path: str):
        if False:
            for i in range(10):
                print('nop')
        self.patterns = {re.compile(element, re.IGNORECASE) for element in patterns}
        self.fallback_path = fallback_path

    def replace_path(self, path: str) -> str:
        if False:
            i = 10
            return i + 15
        for pattern in self.patterns:
            match = pattern.search(path)
            if match:
                return path[match.start():]
        return self.fallback_path