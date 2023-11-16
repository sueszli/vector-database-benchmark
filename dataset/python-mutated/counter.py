from typing import Callable, Optional
from nicegui.element import Element

class Counter(Element, component='counter.js'):

    def __init__(self, title: str, *, on_change: Optional[Callable]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._props['title'] = title
        self.on('change', on_change)

    def reset(self) -> None:
        if False:
            while True:
                i = 10
        self.run_method('reset')