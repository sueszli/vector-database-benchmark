"""
Utilities for local files handling.
"""
import os
import tempfile
import uuid
from contextlib import contextmanager
from typing import Optional
from samcli.lib.utils.hash import file_checksum, str_checksum

@contextmanager
def mktempfile():
    if False:
        for i in range(10):
            print('nop')
    directory = tempfile.gettempdir()
    filename = os.path.join(directory, uuid.uuid4().hex)
    try:
        with open(filename, 'w+') as handle:
            yield handle
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def get_uploaded_s3_object_name(precomputed_md5: Optional[str]=None, file_content: Optional[str]=None, file_path: Optional[str]=None, extension: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    '\n    Generate the file name that will be used while creating the S3 Object based on the file hash value.\n    This method expect either the precomuted hash value of the file, or the file content, or the file path\n\n    Parameters\n    ----------\n    precomputed_md5: str\n        the precomputed hash value of the file.\n    file_content : str\n        The file content to be uploaded to S3.\n    file_path : str\n        The file path to be uploaded to S3\n    extension : str\n        The file extension in S3\n    Returns\n    -------\n    str\n        The generated S3 Object name\n    '
    if precomputed_md5:
        filemd5 = precomputed_md5
    elif file_content:
        filemd5 = str_checksum(file_content)
    elif file_path:
        filemd5 = file_checksum(file_path)
    else:
        raise Exception('Either File Content, File Path, or Precomputed Hash should has a value')
    if extension:
        filemd5 = filemd5 + '.' + extension
    return filemd5