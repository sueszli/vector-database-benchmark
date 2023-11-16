from logging import Handler, LogRecord
from typing import Optional
from .viztracer import VizTracer

class VizLoggingHandler(Handler):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._tracer: Optional[VizTracer] = None

    def emit(self, record: LogRecord) -> None:
        if False:
            print('Hello World!')
        if not self._tracer:
            return
        self._tracer.add_instant(f'logging - {self.format(record)}', scope='p')

    def setTracer(self, tracer: VizTracer) -> None:
        if False:
            i = 10
            return i + 15
        self._tracer = tracer