from __future__ import annotations
import logging
import logging.handlers
import multiprocessing
import queue
from typing import Any
old_factory = logging.getLogRecordFactory()

def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    if False:
        while True:
            i = 10
    record = old_factory(*args, **kwargs)
    record.custom_attribute = 3737844653
    return record
logging.setLogRecordFactory(record_factory)
logging.handlers.QueueHandler(queue.Queue())
logging.handlers.QueueHandler(queue.SimpleQueue())
logging.handlers.QueueHandler(multiprocessing.Queue())
logging.handlers.QueueListener(queue.Queue())
logging.handlers.QueueListener(queue.SimpleQueue())
logging.handlers.QueueListener(multiprocessing.Queue())