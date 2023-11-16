from typing import Iterable, Iterator, Sequence
from robot.utils import MultiMatcher

class NamePatterns(Iterable[str]):

    def __init__(self, patterns: Sequence[str]=(), ignore: Sequence[str]='_'):
        if False:
            i = 10
            return i + 15
        self.matcher = MultiMatcher(patterns, ignore)

    def match(self, name: str, full_name: 'str|None'=None) -> bool:
        if False:
            i = 10
            return i + 15
        match = self.matcher.match
        return bool(match(name) or (full_name and match(full_name)))

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self.matcher)

    def __iter__(self) -> Iterator[str]:
        if False:
            print('Hello World!')
        for matcher in self.matcher:
            yield matcher.pattern