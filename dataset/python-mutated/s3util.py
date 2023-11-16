from __future__ import print_function
from datetime import datetime
import random
import time
import sys
import os
from metaflow.exception import MetaflowException
from metaflow.metaflow_config import DATATOOLS_CLIENT_PARAMS, DATATOOLS_SESSION_VARS, S3_RETRY_COUNT, RETRY_WARNING_THRESHOLD
TEST_S3_RETRY = 'TEST_S3_RETRY' in os.environ
TRANSIENT_RETRY_LINE_CONTENT = '<none>'
TRANSIENT_RETRY_START_LINE = '### RETRY INPUTS ###'

def get_s3_client(s3_role_arn=None, s3_session_vars=None, s3_client_params=None):
    if False:
        return 10
    from metaflow.plugins.aws.aws_client import get_aws_client
    return get_aws_client('s3', with_error=True, role_arn=s3_role_arn, session_vars=s3_session_vars if s3_session_vars else DATATOOLS_SESSION_VARS, client_params=s3_client_params if s3_client_params else DATATOOLS_CLIENT_PARAMS)

def aws_retry(f):
    if False:
        return 10

    def retry_wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        last_exc = None
        for i in range(S3_RETRY_COUNT + 1):
            try:
                ret = f(self, *args, **kwargs)
                if TEST_S3_RETRY and i == 0:
                    raise Exception('TEST_S3_RETRY env var set. Pretending that an S3 op failed. This is not a real failure.')
                else:
                    return ret
            except MetaflowException as ex:
                raise
            except Exception as ex:
                try:
                    function_name = f.func_name
                except AttributeError:
                    function_name = f.__name__
                if TEST_S3_RETRY and i == 0:
                    sys.stderr.write('[WARNING] S3 datastore operation %s failed (%s). Retrying %d more times..\n' % (function_name, ex, S3_RETRY_COUNT - i))
                if i + 1 > RETRY_WARNING_THRESHOLD:
                    sys.stderr.write('[WARNING] S3 datastore operation %s failed (%s). Retrying %d more times..\n' % (function_name, ex, S3_RETRY_COUNT - i))
                self.reset_client(hard_reset=True)
                last_exc = ex
                if not (TEST_S3_RETRY and i == 0):
                    time.sleep(2 ** i + random.randint(0, 5))
        raise last_exc
    return retry_wrapper

def read_in_chunks(dst, src, src_sz, max_chunk_size):
    if False:
        return 10
    remaining = src_sz
    while remaining > 0:
        buf = src.read(min(remaining, max_chunk_size))
        dst.write(buf)
        remaining -= len(buf)

def get_timestamp(dt):
    if False:
        while True:
            i = 10
    '\n    Python2 compatible way to compute the timestamp (seconds since 1/1/1970)\n    '
    return (dt.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()