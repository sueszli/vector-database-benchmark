from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .viztracer import VizTracer

class VizEvent:

    def __init__(self, tracer: 'VizTracer', event_name: str, file_name: str, lineno: int) -> None:
        if False:
            print('Hello World!')
        self._tracer = tracer
        self.event_name = event_name
        self.file_name = file_name
        self.lineno = lineno
        self.start = 0.0

    def __enter__(self) -> None:
        if False:
            while True:
                i = 10
        self.start = self._tracer.getts()

    def __exit__(self, type, value, trace) -> None:
        if False:
            while True:
                i = 10
        dur = self._tracer.getts() - self.start
        raw_data = {'ph': 'X', 'name': f'{self.event_name} ({self.file_name}:{self.lineno})', 'ts': self.start, 'dur': dur, 'cat': 'FEE'}
        self._tracer.add_raw(raw_data)