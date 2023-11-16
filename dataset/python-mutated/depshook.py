from __future__ import annotations
from mypy.nodes import MypyFile
from mypy.plugin import Plugin

class DepsPlugin(Plugin):

    def get_additional_deps(self, file: MypyFile) -> list[tuple[int, str, int]]:
        if False:
            while True:
                i = 10
        if file.fullname == '__main__':
            return [(10, 'err', -1)]
        return []

def plugin(version: str) -> type[DepsPlugin]:
    if False:
        i = 10
        return i + 15
    return DepsPlugin