from typing import Any, Iterator, List
from rich.console import Console
from rich.segment import Segment

class SyntaxLine:

    def __init__(self, segments: List[Segment]):
        if False:
            return 10
        self.segments = segments

    def __rich_console__(self, console: Console, _options: Any) -> Iterator[Segment]:
        if False:
            while True:
                i = 10
        yield from self.segments