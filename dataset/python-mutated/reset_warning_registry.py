from __future__ import annotations
import re
import sys
from typing import Match

class reset_warning_registry:
    """
    context manager which archives & clears warning registry for duration of
    context.

    :param pattern:
          optional regex pattern, causes manager to only reset modules whose
          names match this pattern. defaults to ``".*"``.
    """
    _pattern: Match[str] | None = None
    _backup: dict | None = None

    def __init__(self, pattern=None):
        if False:
            i = 10
            return i + 15
        self._pattern = re.compile(pattern or '.*')

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = self._pattern
        backup = self._backup = {}
        for (name, mod) in list(sys.modules.items()):
            if pattern.match(name):
                reg = getattr(mod, '__warningregistry__', None)
                if reg and isinstance(reg, dict):
                    backup[name] = reg.copy()
                    reg.clear()
        return self

    def __exit__(self, *exc_info):
        if False:
            while True:
                i = 10
        modules = sys.modules
        backup = self._backup
        for (name, content) in backup.items():
            mod = modules.get(name)
            if mod is None:
                continue
            reg = getattr(mod, '__warningregistry__', None)
            if reg is None:
                setattr(mod, '__warningregistry__', content)
            else:
                reg.clear()
                reg.update(content)
        pattern = self._pattern
        for (name, mod) in list(modules.items()):
            if pattern.match(name) and name not in backup:
                reg = getattr(mod, '__warningregistry__', None)
                if reg and isinstance(reg, dict):
                    reg.clear()