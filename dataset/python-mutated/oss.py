"""This module contains Alibaba Cloud OSS operators."""
from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models import BaseOperator
from airflow.providers.alibaba.cloud.hooks.oss import OSSHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class OSSCreateBucketOperator(BaseOperator):
    """
    This operator creates an OSS bucket.

    :param region: OSS region you want to create bucket
    :param bucket_name: This is bucket name you want to create
    :param oss_conn_id: The Airflow connection used for OSS credentials.
    """

    def __init__(self, region: str, bucket_name: str | None=None, oss_conn_id: str='oss_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.oss_conn_id = oss_conn_id
        self.region = region
        self.bucket_name = bucket_name

    def execute(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        oss_hook = OSSHook(oss_conn_id=self.oss_conn_id, region=self.region)
        oss_hook.create_bucket(bucket_name=self.bucket_name)

class OSSDeleteBucketOperator(BaseOperator):
    """
    This operator to delete an OSS bucket.

    :param region: OSS region you want to create bucket
    :param bucket_name: This is bucket name you want to delete
    :param oss_conn_id: The Airflow connection used for OSS credentials.
    """

    def __init__(self, region: str, bucket_name: str | None=None, oss_conn_id: str='oss_default', **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.oss_conn_id = oss_conn_id
        self.region = region
        self.bucket_name = bucket_name

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        oss_hook = OSSHook(oss_conn_id=self.oss_conn_id, region=self.region)
        oss_hook.delete_bucket(bucket_name=self.bucket_name)

class OSSUploadObjectOperator(BaseOperator):
    """
    This operator to upload an file-like object.

    :param key: the OSS path of the object
    :param file: local file to upload.
    :param region: OSS region you want to create bucket
    :param bucket_name: This is bucket name you want to create
    :param oss_conn_id: The Airflow connection used for OSS credentials.
    """

    def __init__(self, key: str, file: str, region: str, bucket_name: str | None=None, oss_conn_id: str='oss_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.key = key
        self.file = file
        self.oss_conn_id = oss_conn_id
        self.region = region
        self.bucket_name = bucket_name

    def execute(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        oss_hook = OSSHook(oss_conn_id=self.oss_conn_id, region=self.region)
        oss_hook.upload_local_file(bucket_name=self.bucket_name, key=self.key, file=self.file)

class OSSDownloadObjectOperator(BaseOperator):
    """
    This operator to Download an OSS object.

    :param key: key of the object to download.
    :param local_file: local path + file name to save.
    :param region: OSS region
    :param bucket_name: OSS bucket name
    :param oss_conn_id: The Airflow connection used for OSS credentials.
    """

    def __init__(self, key: str, file: str, region: str, bucket_name: str | None=None, oss_conn_id: str='oss_default', **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.key = key
        self.file = file
        self.oss_conn_id = oss_conn_id
        self.region = region
        self.bucket_name = bucket_name

    def execute(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        oss_hook = OSSHook(oss_conn_id=self.oss_conn_id, region=self.region)
        oss_hook.download_file(bucket_name=self.bucket_name, key=self.key, local_file=self.file)

class OSSDeleteBatchObjectOperator(BaseOperator):
    """
    This operator to delete OSS objects.

    :param key: key list of the objects to delete.
    :param region: OSS region
    :param bucket_name: OSS bucket name
    :param oss_conn_id: The Airflow connection used for OSS credentials.
    """

    def __init__(self, keys: list, region: str, bucket_name: str | None=None, oss_conn_id: str='oss_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.keys = keys
        self.oss_conn_id = oss_conn_id
        self.region = region
        self.bucket_name = bucket_name

    def execute(self, context: Context):
        if False:
            return 10
        oss_hook = OSSHook(oss_conn_id=self.oss_conn_id, region=self.region)
        oss_hook.delete_objects(bucket_name=self.bucket_name, key=self.keys)

class OSSDeleteObjectOperator(BaseOperator):
    """
    This operator to delete an OSS object.

    :param key: key of the object to delete.
    :param region: OSS region
    :param bucket_name: OSS bucket name
    :param oss_conn_id: The Airflow connection used for OSS credentials.
    """

    def __init__(self, key: str, region: str, bucket_name: str | None=None, oss_conn_id: str='oss_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.key = key
        self.oss_conn_id = oss_conn_id
        self.region = region
        self.bucket_name = bucket_name

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        oss_hook = OSSHook(oss_conn_id=self.oss_conn_id, region=self.region)
        oss_hook.delete_object(bucket_name=self.bucket_name, key=self.key)