from __future__ import annotations
import contextlib
import os
import pathlib
import shutil
from functools import cached_property
from packaging.version import Version
from airflow.configuration import conf
from airflow.providers.alibaba.cloud.hooks.oss import OSSHook
from airflow.utils.log.file_task_handler import FileTaskHandler
from airflow.utils.log.logging_mixin import LoggingMixin

def get_default_delete_local_copy():
    if False:
        while True:
            i = 10
    'Load delete_local_logs conf if Airflow version > 2.6 and return False if not.\n\n    TODO: delete this function when min airflow version >= 2.6\n    '
    from airflow.version import version
    if Version(version) < Version('2.6'):
        return False
    return conf.getboolean('logging', 'delete_local_logs')

class OSSTaskHandler(FileTaskHandler, LoggingMixin):
    """
    OSSTaskHandler is a python log handler that handles and reads task instance logs.

    Extends airflow FileTaskHandler and uploads to and reads from OSS remote storage.
    """

    def __init__(self, base_log_folder, oss_log_folder, filename_template=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.log.info('Using oss_task_handler for remote logging...')
        super().__init__(base_log_folder, filename_template)
        (self.bucket_name, self.base_folder) = OSSHook.parse_oss_url(oss_log_folder)
        self.log_relative_path = ''
        self._hook = None
        self.closed = False
        self.upload_on_close = True
        self.delete_local_copy = kwargs['delete_local_copy'] if 'delete_local_copy' in kwargs else get_default_delete_local_copy()

    @cached_property
    def hook(self):
        if False:
            for i in range(10):
                print('nop')
        remote_conn_id = conf.get('logging', 'REMOTE_LOG_CONN_ID')
        self.log.info('remote_conn_id: %s', remote_conn_id)
        try:
            return OSSHook(oss_conn_id=remote_conn_id)
        except Exception as e:
            self.log.exception(e)
            self.log.error('Could not create an OSSHook with connection id "%s". Please make sure that airflow[oss] is installed and the OSS connection exists.', remote_conn_id)

    def set_context(self, ti):
        if False:
            while True:
                i = 10
        'Set the context of the handler.'
        super().set_context(ti)
        self.log_relative_path = self._render_filename(ti, ti.try_number)
        self.upload_on_close = not ti.raw
        if self.upload_on_close:
            with open(self.handler.baseFilename, 'w'):
                pass

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close and upload local log file to remote storage OSS.'
        if self.closed:
            return
        super().close()
        if not self.upload_on_close:
            return
        local_loc = os.path.join(self.local_base, self.log_relative_path)
        remote_loc = self.log_relative_path
        if os.path.exists(local_loc):
            log = pathlib.Path(local_loc).read_text()
            oss_write = self.oss_write(log, remote_loc)
            if oss_write and self.delete_local_copy:
                shutil.rmtree(os.path.dirname(local_loc))
        self.closed = True

    def _read(self, ti, try_number, metadata=None):
        if False:
            while True:
                i = 10
        '\n        Read logs of given task instance and try_number from OSS remote storage.\n\n        If failed, read the log from task instance host machine.\n\n        :param ti: task instance object\n        :param try_number: task instance try_number to read logs from\n        :param metadata: log metadata,\n                         can be used for steaming log reading and auto-tailing.\n        '
        log_relative_path = self._render_filename(ti, try_number)
        remote_loc = log_relative_path
        if not self.oss_log_exists(remote_loc):
            return super()._read(ti, try_number, metadata)
        remote_log = self.oss_read(remote_loc, return_error=True)
        log = f'*** Reading remote log from {remote_loc}.\n{remote_log}\n'
        return (log, {'end_of_log': True})

    def oss_log_exists(self, remote_log_location):
        if False:
            i = 10
            return i + 15
        "\n        Check if remote_log_location exists in remote storage.\n\n        :param remote_log_location: log's location in remote storage\n        :return: True if location exists else False\n        "
        oss_remote_log_location = f'{self.base_folder}/{remote_log_location}'
        with contextlib.suppress(Exception):
            return self.hook.key_exist(self.bucket_name, oss_remote_log_location)
        return False

    def oss_read(self, remote_log_location, return_error=False):
        if False:
            i = 10
            return i + 15
        "\n        Return the log at the remote_log_location or '' if no logs are found or there is an error.\n\n        :param remote_log_location: the log's location in remote storage\n        :param return_error: if True, returns a string error message if an\n            error occurs. Otherwise, returns '' when an error occurs.\n        "
        try:
            oss_remote_log_location = f'{self.base_folder}/{remote_log_location}'
            self.log.info('read remote log: %s', oss_remote_log_location)
            return self.hook.read_key(self.bucket_name, oss_remote_log_location)
        except Exception:
            msg = f'Could not read logs from {oss_remote_log_location}'
            self.log.exception(msg)
            if return_error:
                return msg

    def oss_write(self, log, remote_log_location, append=True) -> bool:
        if False:
            while True:
                i = 10
        "\n        Write the log to remote_log_location and return `True`; fails silently and returns `False` on error.\n\n        :param log: the log to write to the remote_log_location\n        :param remote_log_location: the log's location in remote storage\n        :param append: if False, any existing log file is overwritten. If True,\n            the new log is appended to any existing logs.\n        :return: whether the log is successfully written to remote location or not.\n        "
        oss_remote_log_location = f'{self.base_folder}/{remote_log_location}'
        pos = 0
        if append and self.oss_log_exists(oss_remote_log_location):
            head = self.hook.head_key(self.bucket_name, oss_remote_log_location)
            pos = head.content_length
        self.log.info('log write pos is: %s', pos)
        try:
            self.log.info('writing remote log: %s', oss_remote_log_location)
            self.hook.append_string(self.bucket_name, log, oss_remote_log_location, pos)
        except Exception:
            self.log.exception('Could not write logs to %s, log write pos is: %s, Append is %s', oss_remote_log_location, pos, append)
            return False
        return True