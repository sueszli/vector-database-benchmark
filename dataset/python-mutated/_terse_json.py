"""
Log formatters that output terse JSON.
"""
import json
import logging
_encoder = json.JSONEncoder(ensure_ascii=False, separators=(',', ':'))
_IGNORED_LOG_RECORD_ATTRIBUTES = {'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename', 'funcName', 'levelname', 'levelno', 'lineno', 'message', 'module', 'msecs', 'msg', 'name', 'pathname', 'process', 'processName', 'relativeCreated', 'stack_info', 'taskName', 'thread', 'threadName'}

class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        if False:
            for i in range(10):
                print('nop')
        event = {'log': record.getMessage(), 'namespace': record.name, 'level': record.levelname}
        return self._format(record, event)

    def _format(self, record: logging.LogRecord, event: dict) -> str:
        if False:
            for i in range(10):
                print('nop')
        for (key, value) in record.__dict__.items():
            if key not in _IGNORED_LOG_RECORD_ATTRIBUTES:
                event[key] = value
        if record.exc_info:
            (exc_type, exc_value, _) = record.exc_info
            if exc_type:
                event['exc_type'] = f'{exc_type.__name__}'
                event['exc_value'] = f'{exc_value}'
        return _encoder.encode(event)

class TerseJsonFormatter(JsonFormatter):

    def format(self, record: logging.LogRecord) -> str:
        if False:
            print('Hello World!')
        event = {'log': record.getMessage(), 'namespace': record.name, 'level': record.levelname, 'time': round(record.created, 2)}
        return self._format(record, event)