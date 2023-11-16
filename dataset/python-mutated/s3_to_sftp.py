from __future__ import annotations
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Sequence
from urllib.parse import urlsplit
from airflow.models import BaseOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.ssh.hooks.ssh import SSHHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class S3ToSFTPOperator(BaseOperator):
    """
    This operator enables the transferring of files from S3 to a SFTP server.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:S3ToSFTPOperator`

    :param sftp_conn_id: The sftp connection id. The name or identifier for
        establishing a connection to the SFTP server.
    :param sftp_path: The sftp remote path. This is the specified file path for
        uploading file to the SFTP server.
    :param aws_conn_id: aws connection to use
    :param s3_bucket: The targeted s3 bucket. This is the S3 bucket from
        where the file is downloaded.
    :param s3_key: The targeted s3 key. This is the specified file path for
        downloading the file from S3.
    """
    template_fields: Sequence[str] = ('s3_key', 'sftp_path', 's3_bucket')

    def __init__(self, *, s3_bucket: str, s3_key: str, sftp_path: str, sftp_conn_id: str='ssh_default', aws_conn_id: str='aws_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.sftp_conn_id = sftp_conn_id
        self.sftp_path = sftp_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.aws_conn_id = aws_conn_id

    @staticmethod
    def get_s3_key(s3_key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Parse the correct format for S3 keys regardless of how the S3 url is passed.'
        parsed_s3_key = urlsplit(s3_key)
        return parsed_s3_key.path.lstrip('/')

    def execute(self, context: Context) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.s3_key = self.get_s3_key(self.s3_key)
        ssh_hook = SSHHook(ssh_conn_id=self.sftp_conn_id)
        s3_hook = S3Hook(self.aws_conn_id)
        s3_client = s3_hook.get_conn()
        sftp_client = ssh_hook.get_conn().open_sftp()
        with NamedTemporaryFile('w') as f:
            s3_client.download_file(self.s3_bucket, self.s3_key, f.name)
            sftp_client.put(f.name, self.sftp_path)