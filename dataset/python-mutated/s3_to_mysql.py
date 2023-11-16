from __future__ import annotations
import os
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.mysql.hooks.mysql import MySqlHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class S3ToMySqlOperator(BaseOperator):
    """
    Loads a file from S3 into a MySQL table.

    :param s3_source_key: The path to the file (S3 key) that will be loaded into MySQL.
    :param mysql_table: The MySQL table into where the data will be sent.
    :param mysql_duplicate_key_handling: Specify what should happen to duplicate data.
        You can choose either `IGNORE` or `REPLACE`.

        .. seealso::
            https://dev.mysql.com/doc/refman/8.0/en/load-data.html#load-data-duplicate-key-handling
    :param mysql_extra_options: MySQL options to specify exactly how to load the data.
    :param aws_conn_id: The S3 connection that contains the credentials to the S3 Bucket.
    :param mysql_conn_id: Reference to :ref:`mysql connection id <howto/connection:mysql>`.
    :param mysql_local_infile: flag to enable local_infile option on the MySQLHook. This
        loads MySQL directly using the LOAD DATA LOCAL INFILE command. Defaults to False.
    """
    template_fields: Sequence[str] = ('s3_source_key', 'mysql_table')
    template_ext: Sequence[str] = ()
    ui_color = '#f4a460'

    def __init__(self, *, s3_source_key: str, mysql_table: str, mysql_duplicate_key_handling: str='IGNORE', mysql_extra_options: str | None=None, aws_conn_id: str='aws_default', mysql_conn_id: str='mysql_default', mysql_local_infile: bool=False, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.s3_source_key = s3_source_key
        self.mysql_table = mysql_table
        self.mysql_duplicate_key_handling = mysql_duplicate_key_handling
        self.mysql_extra_options = mysql_extra_options or ''
        self.aws_conn_id = aws_conn_id
        self.mysql_conn_id = mysql_conn_id
        self.mysql_local_infile = mysql_local_infile

    def execute(self, context: Context) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Executes the transfer operation from S3 to MySQL.\n\n        :param context: The context that is being provided when executing.\n        '
        self.log.info('Loading %s to MySql table %s...', self.s3_source_key, self.mysql_table)
        s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
        file = s3_hook.download_file(key=self.s3_source_key)
        try:
            mysql = MySqlHook(mysql_conn_id=self.mysql_conn_id, local_infile=self.mysql_local_infile)
            mysql.bulk_load_custom(table=self.mysql_table, tmp_file=file, duplicate_key_handling=self.mysql_duplicate_key_handling, extra_options=self.mysql_extra_options)
        finally:
            os.remove(file)