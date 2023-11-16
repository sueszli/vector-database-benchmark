import json
import logging
import os
import typing
from threading import Thread
UNHEALTHY_THREADS_FILE_PATH = '/tmp/task-processor-unhealthy-threads.json'
logger = logging.getLogger(__name__)

def clear_unhealthy_threads():
    if False:
        while True:
            i = 10
    if _unhealthy_threads_file_exists():
        os.remove(UNHEALTHY_THREADS_FILE_PATH)

def write_unhealthy_threads(unhealthy_threads: typing.List[Thread]):
    if False:
        while True:
            i = 10
    unhealthy_thread_names = [t.name for t in unhealthy_threads]
    logger.warning('Writing unhealthy threads: %s', unhealthy_thread_names)
    with open(UNHEALTHY_THREADS_FILE_PATH, 'w+') as f:
        f.write(json.dumps(unhealthy_thread_names))

def get_unhealthy_thread_names() -> typing.List[str]:
    if False:
        while True:
            i = 10
    if not _unhealthy_threads_file_exists():
        return []
    with open(UNHEALTHY_THREADS_FILE_PATH, 'r') as f:
        return json.loads(f.read())

def _unhealthy_threads_file_exists():
    if False:
        i = 10
        return i + 15
    return os.path.exists(UNHEALTHY_THREADS_FILE_PATH)