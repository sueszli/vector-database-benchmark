import re
from typing import Optional

class LooseVersion:
    component_re = re.compile('(\\d+ | [a-z]+ | \\.)', re.VERBOSE)

    def __init__(self, vstring: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        if vstring:
            self.parse(vstring)

    def parse(self, vstring: str) -> None:
        if False:
            i = 10
            return i + 15
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring) if x and x != '.']
        for (i, obj) in enumerate(components):
            try:
                components[i] = int(obj)
            except ValueError:
                pass
        self.version = components

    def __str__(self) -> str:
        if False:
            return 10
        return self.vstring

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return "LooseVersion ('%s')" % str(self)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self, other):
        if False:
            return 10
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0

    def _cmp(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, str):
            other = LooseVersion(other)
        elif not isinstance(other, LooseVersion):
            return NotImplemented
        if self.version == other.version:
            return 0
        if self.version < other.version:
            return -1
        if self.version > other.version:
            return 1