"""
This package provides commonly used methods for dealing with file operation,
including working with network file system like S3, http, etc.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import os
import logging
import shutil
__logger__ = logging.getLogger(__name__)
__RETRY_TIMES = 5
__SLEEP_SECONDS_BETWEEN_RETRIES = 2

def get_protocol(path):
    if False:
        return 10
    "Given a path, returns the protocol the path uses\n\n    For example,\n      's3://a/b/c/'  returns 's3'\n      'http://a/b/c' returns 'http'\n      'tmp/a/bc/'    returns ''\n\n    "
    pos = path.find('://')
    if pos < 0:
        return ''
    return path[0:pos].lower()

def expand_full_path(path):
    if False:
        for i in range(10):
            print('nop')
    "Expand a relative path to a full path\n\n    For example,\n      '~/tmp' may be expanded to '/Users/username/tmp'\n      'abc/def' may be expanded to '/pwd/abc/def'\n    "
    return os.path.abspath(os.path.expanduser(path))

def exists(path, aws_credentials={}):
    if False:
        for i in range(10):
            print('nop')
    if is_local_path(path):
        return os.path.exists(path)
    else:
        raise ValueError('Unsupported protocol %s' % path)

def is_local_path(path):
    if False:
        print('Hello World!')
    'Returns True if the path indicates a local path, otherwise False'
    protocol = get_protocol(path)
    return protocol != 'hdfs' and protocol != 's3' and (protocol != 'http') and (protocol != 'https')

def copy_from_local(localpath, remotepath, is_dir=False, silent=True):
    if False:
        while True:
            i = 10
    if is_local_path(remotepath):
        if is_dir:
            shutil.copytree(localpath, remotepath)
        else:
            shutil.copy(localpath, remotepath)
    else:
        raise ValueError('Unsupported protocol %s' % remotepath)