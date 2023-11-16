from __future__ import annotations
import functools
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable

@dataclass
class Result:
    name: str = ''
    version: str = ''
    description: str = ''

class SearchResultParser(HTMLParser):
    """A simple HTML parser for pypi.org search results."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.results: list[Result] = []
        self._current: Result | None = None
        self._nest_anchors = 0
        self._data_callback: Callable[[str], None] | None = None

    @staticmethod
    def _match_class(attrs: list[tuple[str, str | None]], name: str) -> bool:
        if False:
            print('Hello World!')
        attrs_map = dict(attrs)
        return name in (attrs_map.get('class') or '').split()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if False:
            print('Hello World!')
        if not self._current:
            if tag == 'a' and self._match_class(attrs, 'package-snippet'):
                self._current = Result()
                self._nest_anchors = 1
        elif tag == 'span' and self._match_class(attrs, 'package-snippet__name'):
            self._data_callback = functools.partial(setattr, self._current, 'name')
        elif tag == 'span' and self._match_class(attrs, 'package-snippet__version'):
            self._data_callback = functools.partial(setattr, self._current, 'version')
        elif tag == 'p' and self._match_class(attrs, 'package-snippet__description'):
            self._data_callback = functools.partial(setattr, self._current, 'description')
        elif tag == 'a':
            self._nest_anchors += 1

    def handle_data(self, data: str) -> None:
        if False:
            i = 10
            return i + 15
        if self._data_callback is not None:
            self._data_callback(data)
            self._data_callback = None

    def handle_endtag(self, tag: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if tag != 'a' or self._current is None:
            return
        self._nest_anchors -= 1
        if self._nest_anchors == 0:
            if self._current.name and self._current.version:
                self.results.append(self._current)
            self._current = None