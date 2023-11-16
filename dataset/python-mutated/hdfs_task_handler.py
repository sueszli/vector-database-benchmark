from __future__ import annotations
import os
import pathlib
import shutil
from functools import cached_property
from urllib.parse import urlsplit
from airflow.configuration import conf
from airflow.providers.apache.hdfs.hooks.webhdfs import WebHDFSHook
from airflow.utils.log.file_task_handler import FileTaskHandler
from airflow.utils.log.logging_mixin import LoggingMixin

class HdfsTaskHandler(FileTaskHandler, LoggingMixin):
    """Logging handler to upload and read from HDFS."""

    def __init__(self, base_log_folder: str, hdfs_log_folder: str, filename_template: str | None=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(base_log_folder, filename_template)
        self.remote_base = urlsplit(hdfs_log_folder).path
        self.log_relative_path = ''
        self._hook = None
        self.closed = False
        self.upload_on_close = True
        self.delete_local_copy = kwargs['delete_local_copy'] if 'delete_local_copy' in kwargs else conf.getboolean('logging', 'delete_local_logs', fallback=False)

    @cached_property
    def hook(self):
        if False:
            i = 10
            return i + 15
        'Returns WebHDFSHook.'
        return WebHDFSHook(webhdfs_conn_id=conf.get('logging', 'REMOTE_LOG_CONN_ID'))

    def set_context(self, ti):
        if False:
            return 10
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
            print('Hello World!')
        'Close and upload local log file to HDFS.'
        if self.closed:
            return
        super().close()
        if not self.upload_on_close:
            return
        local_loc = os.path.join(self.local_base, self.log_relative_path)
        remote_loc = os.path.join(self.remote_base, self.log_relative_path)
        if os.path.exists(local_loc) and os.path.isfile(local_loc):
            self.hook.load_file(local_loc, remote_loc)
            if self.delete_local_copy:
                shutil.rmtree(os.path.dirname(local_loc))
        self.closed = True

    def _read_remote_logs(self, ti, try_number, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        worker_log_rel_path = self._render_filename(ti, try_number)
        logs = []
        messages = []
        file_path = os.path.join(self.remote_base, worker_log_rel_path)
        if self.hook.check_for_path(file_path):
            logs.append(self.hook.read_file(file_path).decode('utf-8'))
        else:
            messages.append(f'No logs found on hdfs for ti={ti}')
        return (messages, logs)

    def _read(self, ti, try_number, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read logs of given task instance and try_number from HDFS.\n\n        If failed, read the log from task instance host machine.\n\n        todo: when min airflow version >= 2.6 then remove this method (``_read``)\n\n        :param ti: task instance object\n        :param try_number: task instance try_number to read logs from\n        :param metadata: log metadata,\n                         can be used for steaming log reading and auto-tailing.\n        '
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