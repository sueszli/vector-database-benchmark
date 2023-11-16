from __future__ import annotations
import contextlib
import logging
import traceback
from typing import TYPE_CHECKING
from airflow.utils.session import create_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
__current_task_instance_session: Session | None = None
log = logging.getLogger(__name__)

def get_current_task_instance_session() -> Session:
    if False:
        i = 10
        return i + 15
    global __current_task_instance_session
    if not __current_task_instance_session:
        log.warning('No task session set for this task. Continuing but this likely causes a resource leak.')
        log.warning('Please report this and stacktrace below to https://github.com/apache/airflow/issues')
        for (filename, line_number, name, line) in traceback.extract_stack():
            log.warning('File: "%s", %s , in %s', filename, line_number, name)
            if line:
                log.warning('  %s', line.strip())
        __current_task_instance_session = create_session()
    return __current_task_instance_session

@contextlib.contextmanager
def set_current_task_instance_session(session: Session):
    if False:
        for i in range(10):
            print('nop')
    global __current_task_instance_session
    if __current_task_instance_session:
        raise RuntimeError("Session already set for this task. You can only have one 'set_current_task_session' context manager active at a time.")
    __current_task_instance_session = session
    try:
        yield
    finally:
        __current_task_instance_session = None