import logging
import pathlib
import json
import random
import string
import socket
import os
import threading
from typing import Dict, Optional
from datetime import datetime
from google.protobuf.json_format import MessageToDict, Parse
from ray.core.generated.event_pb2 import Event
global_logger = logging.getLogger(__name__)

def get_event_id():
    if False:
        for i in range(10):
            print('nop')
    return ''.join([random.choice(string.hexdigits) for _ in range(36)])

class EventLoggerAdapter:

    def __init__(self, source: Event.SourceType, logger: logging.Logger):
        if False:
            i = 10
            return i + 15
        "Adapter for the Python logger that's used to emit events.\n\n        When events are emitted, they are aggregated and available via\n        state API and dashboard.\n\n        This class is thread-safe.\n        "
        self.logger = logger
        self.source = source
        self.source_hostname = socket.gethostname()
        self.source_pid = os.getpid()
        self.lock = threading.Lock()
        self.global_context = {}

    def set_global_context(self, global_context: Dict[str, str]=None):
        if False:
            return 10
        'Set the global metadata.\n\n        This method overwrites the global metadata if it is called more than once.\n        '
        with self.lock:
            self.global_context = {} if not global_context else global_context

    def trace(self, message: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._emit(Event.Severity.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._emit(Event.Severity.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._emit(Event.Severity.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        if False:
            return 10
        self._emit(Event.Severity.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        if False:
            i = 10
            return i + 15
        self._emit(Event.Severity.ERROR, message, **kwargs)

    def fatal(self, message: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._emit(Event.Severity.FATAL, message, **kwargs)

    def _emit(self, severity: Event.Severity, message: str, **kwargs):
        if False:
            i = 10
            return i + 15
        event = Event()
        event.event_id = get_event_id()
        event.timestamp = int(datetime.now().timestamp())
        event.message = message
        event.severity = severity
        event.label = ''
        event.source_type = self.source
        event.source_hostname = self.source_hostname
        event.source_pid = self.source_pid
        custom_fields = event.custom_fields
        with self.lock:
            for (k, v) in self.global_context.items():
                if v is not None and k is not None:
                    custom_fields[k] = v
        for (k, v) in kwargs.items():
            if v is not None and k is not None:
                custom_fields[k] = v
        self.logger.info(json.dumps(MessageToDict(event, including_default_value_fields=True, preserving_proto_field_name=True, use_integers_for_enums=False)))
        self.logger.handlers[0].flush()

def _build_event_file_logger(source: Event.SourceType, sink_dir: str):
    if False:
        while True:
            i = 10
    logger = logging.getLogger('_ray_event_logger')
    logger.setLevel(logging.INFO)
    dir_path = pathlib.Path(sink_dir) / 'events'
    filepath = dir_path / f'event_{source}.log'
    dir_path.mkdir(exist_ok=True)
    filepath.touch(exist_ok=True)
    handler = logging.FileHandler(filepath)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
_event_logger_lock = threading.Lock()
_event_logger = {}

def get_event_logger(source: Event.SourceType, sink_dir: str):
    if False:
        for i in range(10):
            print('nop')
    "Get the event logger of the current process.\n\n    There's only 1 event logger per (process, source).\n\n    TODO(sang): Support more impl than file-based logging.\n                Currently, the interface also ties to the\n                file-based logging impl.\n\n    Args:\n        source: The source of the event.\n        sink_dir: The directory to sink event logs.\n    "
    with _event_logger_lock:
        global _event_logger
        source_name = Event.SourceType.Name(source)
        if source_name not in _event_logger:
            logger = _build_event_file_logger(source_name, sink_dir)
            _event_logger[source_name] = EventLoggerAdapter(source, logger)
        return _event_logger[source_name]

def parse_event(event_str: str) -> Optional[Event]:
    if False:
        return 10
    'Parse an event from a string.\n\n    Args:\n        event_str: The string to parse. Expect to be a JSON serialized\n            Event protobuf.\n\n    Returns:\n        The parsed event if parsable, else None\n    '
    try:
        return Parse(event_str, Event())
    except Exception:
        global_logger.exception(f'Failed to parse event: {event_str}')
        return None

def filter_event_by_level(event: Event, filter_event_level: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Filter an event based on event level.\n\n    Args:\n        event: The event to filter.\n        filter_event_level: The event level string to filter by. Any events\n            that are lower than this level will be filtered.\n\n    Returns:\n        True if the event should be filtered, else False.\n    '
    event_levels = {Event.Severity.TRACE: 0, Event.Severity.DEBUG: 1, Event.Severity.INFO: 2, Event.Severity.WARNING: 3, Event.Severity.ERROR: 4, Event.Severity.FATAL: 5}
    filter_event_level = filter_event_level.upper()
    filter_event_level = Event.Severity.Value(filter_event_level)
    if event_levels[event.severity] < event_levels[filter_event_level]:
        return True
    return False