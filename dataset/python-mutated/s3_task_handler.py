from __future__ import annotations
import os
import pathlib
import shutil
from functools import cached_property
from packaging.version import Version
from airflow.configuration import conf
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.log.file_task_handler import FileTaskHandler
from airflow.utils.log.logging_mixin import LoggingMixin

def get_default_delete_local_copy():
    if False:
        return 10
    'Load delete_local_logs conf if Airflow version > 2.6 and return False if not.\n\n    TODO: delete this function when min airflow version >= 2.6\n    '
    from airflow.version import version
    if Version(version) < Version('2.6'):
        return False
    return conf.getboolean('logging', 'delete_local_logs')

class S3TaskHandler(FileTaskHandler, LoggingMixin):
    """
    S3TaskHandler is a python log handler that handles and reads task instance logs.

    It extends airflow FileTaskHandler and uploads to and reads from S3 remote storage.
    """
    trigger_should_wrap = True

    def __init__(self, base_log_folder: str, s3_log_folder: str, filename_template: str | None=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(base_log_folder, filename_template)
        self.remote_base = s3_log_folder
        self.log_relative_path = ''
        self._hook = None
        self.closed = False
        self.upload_on_close = True
        self.delete_local_copy = kwargs['delete_local_copy'] if 'delete_local_copy' in kwargs else get_default_delete_local_copy()

    @cached_property
    def hook(self):
        if False:
            return 10
        'Returns S3Hook.'
        return S3Hook(aws_conn_id=conf.get('logging', 'REMOTE_LOG_CONN_ID'), transfer_config_args={'use_threads': False})

    def set_context(self, ti):
        if False:
            while True:
                i = 10
        super().set_context(ti)
        full_path = self.handler.baseFilename
        self.log_relative_path = pathlib.Path(full_path).relative_to(self.local_base).as_posix()
        is_trigger_log_context = getattr(ti, 'is_trigger_log_context', False)
        self.upload_on_close = is_trigger_log_context or not ti.raw
        if self.upload_on_close:
            with open(self.handler.baseFilename, 'w'):
                pass

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close and upload local log file to remote storage S3.'
        if self.closed:
            return
        super().close()
        if not self.upload_on_close:
            return
        local_loc = os.path.join(self.local_base, self.log_relative_path)
        remote_loc = os.path.join(self.remote_base, self.log_relative_path)
        if os.path.exists(local_loc):
            log = pathlib.Path(local_loc).read_text()
            write_to_s3 = self.s3_write(log, remote_loc)
            if write_to_s3 and self.delete_local_copy:
                shutil.rmtree(os.path.dirname(local_loc))
        self.closed = True

    def _read_remote_logs(self, ti, try_number, metadata=None) -> tuple[list[str], list[str]]:
        if False:
            return 10
        worker_log_rel_path = self._render_filename(ti, try_number)
        logs = []
        messages = []
        (bucket, prefix) = self.hook.parse_s3_url(s3url=os.path.join(self.remote_base, worker_log_rel_path))
        keys = self.hook.list_keys(bucket_name=bucket, prefix=prefix)
        if keys:
            keys = sorted((f's3://{bucket}/{key}' for key in keys))
            messages.append('Found logs in s3:')
            messages.extend((f'  * {key}' for key in keys))
            for key in keys:
                logs.append(self.s3_read(key, return_error=True))
        else:
            messages.append(f'No logs found on s3 for ti={ti}')
        return (messages, logs)

    def _read(self, ti, try_number, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read logs of given task instance and try_number from S3 remote storage.\n\n        If failed, read the log from task instance host machine.\n\n        todo: when min airflow version >= 2.6 then remove this method (``_read``)\n\n        :param ti: task instance object\n        :param try_number: task instance try_number to read logs from\n        :param metadata: log metadata,\n                         can be used for steaming log reading and auto-tailing.\n        '
        if hasattr(super(), '_read_remote_logs'):
            return super()._read(ti, try_number, metadata)
        (messages, logs) = self._read_remote_logs(ti, try_number, metadata)
        if logs:
            return (''.join((f'*** {x}\n' for x in messages)) + '\n'.join(logs), {'end_of_log': True})
        else:
            if metadata and metadata.get('log_pos', 0) > 0:
                log_prefix = ''
            else:
                log_prefix = '*** Falling back to local log\n'
            (local_log, metadata) = super()._read(ti, try_number, metadata)
            return (f'{log_prefix}{local_log}', metadata)

    def s3_log_exists(self, remote_log_location: str) -> bool:
        if False:
            print('Hello World!')
        "\n        Check if remote_log_location exists in remote storage.\n\n        :param remote_log_location: log's location in remote storage\n        :return: True if location exists else False\n        "
        return self.hook.check_for_key(remote_log_location)

    def s3_read(self, remote_log_location: str, return_error: bool=False) -> str:
        if False:
            while True:
                i = 10
        "\n        Return the log found at the remote_log_location or '' if no logs are found or there is an error.\n\n        :param remote_log_location: the log's location in remote storage\n        :param return_error: if True, returns a string error message if an\n            error occurs. Otherwise returns '' when an error occurs.\n        :return: the log found at the remote_log_location\n        "
        try:
            return self.hook.read_key(remote_log_location)
        except Exception as error:
            msg = f'Could not read logs from {remote_log_location} with error: {error}'
            self.log.exception(msg)
            if return_error:
                return msg
        return ''

    def s3_write(self, log: str, remote_log_location: str, append: bool=True, max_retry: int=1) -> bool:
        if False:
            print('Hello World!')
        "\n        Write the log to the remote_log_location; return `True` or fails silently and return `False`.\n\n        :param log: the log to write to the remote_log_location\n        :param remote_log_location: the log's location in remote storage\n        :param append: if False, any existing log file is overwritten. If True,\n            the new log is appended to any existing logs.\n        :param max_retry: Maximum number of times to retry on upload failure\n        :return: whether the log is successfully written to remote location or not.\n        "
        try:
            if append and self.s3_log_exists(remote_log_location):
                old_log = self.s3_read(remote_log_location)
                log = f'{old_log}\n{log}' if old_log else log
        except Exception:
            self.log.exception('Could not verify previous log to append')
            return False
        for try_num in range(1 + max_retry):
            try:
                self.hook.load_string(log, key=remote_log_location, replace=True, encrypt=conf.getboolean('logging', 'ENCRYPT_S3_LOGS'))
                break
            except Exception:
                if try_num < max_retry:
                    self.log.warning('Failed attempt to write logs to %s, will retry', remote_log_location)
                else:
                    self.log.exception('Could not write logs to %s', remote_log_location)
                    return False
        return True