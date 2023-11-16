from __future__ import annotations
import asyncio
import logging
from contextvars import ContextVar
from copy import copy
from logging.handlers import QueueHandler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from airflow.utils.log.file_task_handler import FileTaskHandler
ctx_task_instance: ContextVar = ContextVar('task_instance')
ctx_trigger_id: ContextVar = ContextVar('trigger_id')
ctx_trigger_end: ContextVar = ContextVar('trigger_end')
ctx_indiv_trigger: ContextVar = ContextVar('__individual_trigger')

class TriggerMetadataFilter(logging.Filter):
    """
    Injects TI key, triggerer job_id, and trigger_id into the log record.

    :meta private:
    """

    def filter(self, record):
        if False:
            while True:
                i = 10
        for var in (ctx_task_instance, ctx_trigger_id, ctx_trigger_end, ctx_indiv_trigger):
            val = var.get(None)
            if val is not None:
                setattr(record, var.name, val)
        return True

class DropTriggerLogsFilter(logging.Filter):
    """
    If record has attr with name ctx_indiv_trigger, filter the record.

    The purpose here is to prevent trigger logs from going to stdout
    in the trigger service.

    :meta private:
    """

    def filter(self, record):
        if False:
            for i in range(10):
                print('nop')
        return getattr(record, ctx_indiv_trigger.name, None) is None

class TriggererHandlerWrapper(logging.Handler):
    """
    Wrap inheritors of FileTaskHandler and direct log messages to them based on trigger_id.

    :meta private:
    """
    trigger_should_queue = True

    def __init__(self, base_handler: FileTaskHandler, level=logging.NOTSET):
        if False:
            i = 10
            return i + 15
        super().__init__(level=level)
        self.base_handler: FileTaskHandler = base_handler
        self.handlers: dict[int, FileTaskHandler] = {}

    def _make_handler(self, ti):
        if False:
            for i in range(10):
                print('nop')
        h = copy(self.base_handler)
        h.set_context(ti=ti)
        return h

    def _get_or_create_handler(self, trigger_id, ti):
        if False:
            for i in range(10):
                print('nop')
        if trigger_id not in self.handlers:
            self.handlers[trigger_id] = self._make_handler(ti)
        return self.handlers[trigger_id]

    def emit(self, record):
        if False:
            while True:
                i = 10
        h = self._get_or_create_handler(record.trigger_id, record.task_instance)
        h.emit(record)

    def handle(self, record):
        if False:
            i = 10
            return i + 15
        if not getattr(record, ctx_indiv_trigger.name, None):
            return False
        if record.trigger_end:
            self.close_one(record.trigger_id)
            return False
        emit = self.filter(record)
        if emit:
            self.emit(record)
        return emit

    def close_one(self, trigger_id):
        if False:
            print('Hello World!')
        h = self.handlers.get(trigger_id)
        if h:
            h.close()
            del self.handlers[trigger_id]

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        for h in self.handlers.values():
            h.flush()

    def close(self):
        if False:
            i = 10
            return i + 15
        for trigger_id in list(self.handlers.keys()):
            h = self.handlers[trigger_id]
            h.close()
            del self.handlers[trigger_id]

class LocalQueueHandler(QueueHandler):
    """
    Send messages to queue.

    :meta private:
    """

    def emit(self, record: logging.LogRecord) -> None:
        if False:
            print('Hello World!')
        try:
            self.enqueue(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.handleError(record)