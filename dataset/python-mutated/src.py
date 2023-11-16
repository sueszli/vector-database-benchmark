from __future__ import annotations
from pathlib import Path
from poetry.layouts.layout import Layout

class SrcLayout(Layout):

    @property
    def basedir(self) -> Path:
        if False:
            for i in range(10):
                print('nop')
        return Path('src')