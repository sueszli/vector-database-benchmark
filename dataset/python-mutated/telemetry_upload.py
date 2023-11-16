import datetime
import os
import threading
import zlib
from contextlib import contextmanager
from typing import Iterator
from .telemetry import MAX_BYTES, get_or_create_dir_from_dagster_home

def get_dagster_telemetry_url() -> str:
    if False:
        for i in range(10):
            print('nop')
    return os.getenv('DAGSTER_TELEMETRY_URL', default='http://telemetry.dagster.io/actions')

def is_running_in_test() -> bool:
    if False:
        while True:
            i = 10
    return os.getenv('BUILDKITE') is not None or os.getenv('TF_BUILD') is not None or os.getenv('DAGSTER_DISABLE_TELEMETRY') is not None

@contextmanager
def uploading_logging_thread() -> Iterator[None]:
    if False:
        print('Hello World!')
    stop_event = threading.Event()
    logging_thread = threading.Thread(target=upload_logs, args=[stop_event], name='telemetry-upload')
    try:
        logging_thread.start()
        yield
    finally:
        stop_event.set()
        logging_thread.join()

def upload_logs(stop_event: threading.Event, raise_errors: bool=False) -> None:
    if False:
        while True:
            i = 10
    'Upload logs to telemetry server every hour, or when log directory size is > 10MB.'
    if is_running_in_test():
        return
    try:
        last_run = datetime.datetime.now() - datetime.timedelta(minutes=120)
        dagster_log_dir = get_or_create_dir_from_dagster_home('logs')
        dagster_log_queue_dir = get_or_create_dir_from_dagster_home('.logs_queue')
        in_progress = False
        while not stop_event.is_set():
            log_size = 0
            if os.path.isdir(dagster_log_dir):
                log_size = sum((os.path.getsize(os.path.join(dagster_log_dir, f)) for f in os.listdir(dagster_log_dir) if os.path.isfile(os.path.join(dagster_log_dir, f))))
            log_queue_size = 0
            if os.path.isdir(dagster_log_queue_dir):
                log_queue_size = sum((os.path.getsize(os.path.join(dagster_log_queue_dir, f)) for f in os.listdir(dagster_log_queue_dir) if os.path.isfile(os.path.join(dagster_log_queue_dir, f))))
            if log_size == 0 and log_queue_size == 0:
                return
            if not in_progress and (datetime.datetime.now() - last_run > datetime.timedelta(minutes=60) or log_size >= MAX_BYTES or log_queue_size >= MAX_BYTES):
                in_progress = True
                last_run = datetime.datetime.now()
                dagster_log_dir = get_or_create_dir_from_dagster_home('logs')
                dagster_log_queue_dir = get_or_create_dir_from_dagster_home('.logs_queue')
                _upload_logs(dagster_log_dir, log_size, dagster_log_queue_dir, raise_errors=raise_errors)
                in_progress = False
            stop_event.wait(600)
    except Exception:
        if raise_errors:
            raise

def _upload_logs(dagster_log_dir: str, log_size: int, dagster_log_queue_dir: str, raise_errors: bool) -> None:
    if False:
        while True:
            i = 10
    'Send POST request to telemetry server with the contents of $DAGSTER_HOME/logs/ directory.'
    try:
        import requests
        if log_size > 0:
            for f in os.listdir(dagster_log_queue_dir):
                os.remove(os.path.join(dagster_log_queue_dir, f))
            os.rmdir(dagster_log_queue_dir)
            os.rename(dagster_log_dir, dagster_log_queue_dir)
        for curr_path in os.listdir(dagster_log_queue_dir):
            curr_full_path = os.path.join(dagster_log_queue_dir, curr_path)
            retry_num = 0
            max_retries = 3
            success = False
            while not success and retry_num <= max_retries:
                with open(curr_full_path, 'rb') as curr_file:
                    byte = curr_file.read()
                    data = zlib.compress(byte, zlib.Z_BEST_COMPRESSION)
                    headers = {'content-encoding': 'gzip'}
                    r = requests.post(get_dagster_telemetry_url(), data=data, headers=headers)
                    if r.status_code == 200:
                        success = True
                    retry_num += 1
            if success:
                os.remove(curr_full_path)
    except Exception:
        if raise_errors:
            raise