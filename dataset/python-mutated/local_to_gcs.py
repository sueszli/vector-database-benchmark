"""This module contains operator for uploading local file(s) to GCS."""
from __future__ import annotations
import os
from glob import glob
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class LocalFilesystemToGCSOperator(BaseOperator):
    """
    Uploads a file or list of files to Google Cloud Storage; optionally can compress the file for upload.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:LocalFilesystemToGCSOperator`

    :param src: Path to the local file, or list of local files. Path can be either absolute
        (e.g. /path/to/file.ext) or relative (e.g. ../../foo/*/*.csv). (templated)
    :param dst: Destination path within the specified bucket on GCS (e.g. /path/to/file.ext).
        If multiple files are being uploaded, specify object prefix with trailing backslash
        (e.g. /path/to/directory/) (templated)
    :param bucket: The bucket to upload to. (templated)
    :param gcp_conn_id: (Optional) The connection ID used to connect to Google Cloud.
    :param mime_type: The mime-type string
    :param gzip: Allows for file to be compressed and uploaded as gzip
    :param impersonation_chain: Optional service account to impersonate using short-term
        credentials, or chained list of accounts required to get the access_token
        of the last account in the list, which will be impersonated in the request.
        If set as a string, the account must grant the originating account
        the Service Account Token Creator IAM role.
        If set as a sequence, the identities from the list must grant
        Service Account Token Creator IAM role to the directly preceding identity, with first
        account from the list granting this role to the originating account (templated).
    """
    template_fields: Sequence[str] = ('src', 'dst', 'bucket', 'impersonation_chain')

    def __init__(self, *, src, dst, bucket, gcp_conn_id='google_cloud_default', mime_type='application/octet-stream', gzip=False, impersonation_chain: str | Sequence[str] | None=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.src = src
        self.dst = dst
        self.bucket = bucket
        self.gcp_conn_id = gcp_conn_id
        self.mime_type = mime_type
        self.gzip = gzip
        self.impersonation_chain = impersonation_chain

    def execute(self, context: Context):
        if False:
            for i in range(10):
                print('nop')
        'Uploads a file or list of files to Google Cloud Storage.'
        hook = GCSHook(gcp_conn_id=self.gcp_conn_id, impersonation_chain=self.impersonation_chain)
        filepaths = self.src if isinstance(self.src, list) else glob(self.src)
        if not filepaths:
            raise FileNotFoundError(self.src)
        if os.path.basename(self.dst):
            if len(filepaths) > 1:
                raise ValueError("'dst' parameter references filepath. Please specify directory (with trailing backslash) to upload multiple files. e.g. /path/to/directory/")
            object_paths = [self.dst]
        else:
            object_paths = [os.path.join(self.dst, os.path.basename(filepath)) for filepath in filepaths]
        for (filepath, object_path) in zip(filepaths, object_paths):
            hook.upload(bucket_name=self.bucket, object_name=object_path, mime_type=self.mime_type, filename=filepath, gzip=self.gzip)