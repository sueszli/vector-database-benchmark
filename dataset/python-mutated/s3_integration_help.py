import boto
import json
import os
import pytest
from boto import sts
from boto.s3.connection import Location
from wal_e.blobstore import s3
from wal_e.blobstore.s3 import calling_format
from wal_e.cmd import parse_boolean_envvar

def bucket_name_mangle(bn, delimiter='-'):
    if False:
        for i in range(10):
            print('nop')
    return bn + delimiter + os.getenv('AWS_ACCESS_KEY_ID').lower()

def no_real_s3_credentials():
    if False:
        while True:
            i = 10
    "Helps skip integration tests without live credentials.\n\n    Phrased in the negative to make it read better with 'skipif'.\n    "
    if parse_boolean_envvar(os.getenv('WALE_S3_INTEGRATION_TESTS')) is not True:
        return True
    for e_var in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'):
        if os.getenv(e_var) is None:
            return True
    return False

def prepare_s3_default_test_bucket():
    if False:
        while True:
            i = 10
    if no_real_s3_credentials():
        assert False
    bucket_name = bucket_name_mangle('waletdefwuy')
    creds = s3.Credentials(os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'), os.getenv('AWS_SECURITY_TOKEN'))
    cinfo = calling_format.from_store_name(bucket_name, region='us-west-1')
    conn = cinfo.connect(creds)

    def _clean():
        if False:
            while True:
                i = 10
        bucket = conn.get_bucket(bucket_name)
        bucket.delete_keys((key.name for key in bucket.list()))
    try:
        conn.create_bucket(bucket_name, location=Location.USWest)
    except boto.exception.S3CreateError as e:
        if e.status == 409:
            _clean()
        else:
            raise
    else:
        _clean()
    return bucket_name

@pytest.fixture(scope='session')
def default_test_bucket():
    if False:
        print('Hello World!')
    if not no_real_s3_credentials():
        os.putenv('AWS_REGION', 'us-east-1')
        ret = prepare_s3_default_test_bucket()
        os.unsetenv('AWS_REGION')
        return ret

def boto_supports_certs():
    if False:
        return 10
    return tuple((int(x) for x in boto.__version__.split('.'))) >= (2, 6, 0)

def make_policy(bucket_name, prefix, allow_get_location=False):
    if False:
        for i in range(10):
            print('nop')
    'Produces a S3 IAM text for selective access of data.\n\n    Only a prefix can be listed, gotten, or written to when a\n    credential is subject to this policy text.\n    '
    bucket_arn = 'arn:aws:s3:::' + bucket_name
    prefix_arn = 'arn:aws:s3:::{0}/{1}/*'.format(bucket_name, prefix)
    structure = {'Version': '2012-10-17', 'Statement': [{'Action': ['s3:ListBucket'], 'Effect': 'Allow', 'Resource': [bucket_arn], 'Condition': {'StringLike': {'s3:prefix': [prefix + '/*']}}}, {'Effect': 'Allow', 'Action': ['s3:PutObject', 's3:GetObject'], 'Resource': [prefix_arn]}]}
    if allow_get_location:
        structure['Statement'].append({'Action': ['s3:GetBucketLocation'], 'Effect': 'Allow', 'Resource': [bucket_arn]})
    return json.dumps(structure, indent=2)

@pytest.fixture
def sts_conn():
    if False:
        return 10
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    return sts.connect_to_region('us-east-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

def _delete_keys(bucket, keys):
    if False:
        print('Hello World!')
    for name in keys:
        while True:
            try:
                k = boto.s3.connection.Key(bucket, name)
                bucket.delete_key(k)
            except boto.exception.S3ResponseError as e:
                if e.status == 404:
                    break
                raise
            else:
                break

def apathetic_bucket_delete(bucket_name, keys, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    kwargs.setdefault('host', 's3.amazonaws.com')
    conn = boto.s3.connection.S3Connection(*args, **kwargs)
    bucket = conn.lookup(bucket_name)
    if bucket:
        _delete_keys(conn.lookup(bucket_name), keys)
    try:
        conn.delete_bucket(bucket_name)
    except boto.exception.S3ResponseError as e:
        if e.status == 404:
            pass
        else:
            raise
    return conn

def insistent_bucket_delete(conn, bucket_name, keys):
    if False:
        return 10
    bucket = conn.lookup(bucket_name)
    if bucket:
        _delete_keys(bucket, keys)
    while True:
        try:
            conn.delete_bucket(bucket_name)
        except boto.exception.S3ResponseError as e:
            if e.status == 404:
                continue
            else:
                raise
        break

def insistent_bucket_create(conn, bucket_name, *args, **kwargs):
    if False:
        while True:
            i = 10
    while True:
        try:
            bucket = conn.create_bucket(bucket_name, *args, **kwargs)
        except boto.exception.S3CreateError as e:
            if e.status == 409:
                continue
            raise
        return bucket

class FreshBucket(object):

    def __init__(self, bucket_name, keys=[], *args, **kwargs):
        if False:
            print('Hello World!')
        self.bucket_name = bucket_name
        self.keys = keys
        self.conn_args = args
        self.conn_kwargs = kwargs
        self.created_bucket = False

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if boto_supports_certs():
            self.conn_kwargs.setdefault('validate_certs', True)
        self.conn = apathetic_bucket_delete(self.bucket_name, self.keys, *self.conn_args, **self.conn_kwargs)
        return self

    def create(self, *args, **kwargs):
        if False:
            print('Hello World!')
        bucket = insistent_bucket_create(self.conn, self.bucket_name, *args, **kwargs)
        self.created_bucket = True
        return bucket

    def __exit__(self, typ, value, traceback):
        if False:
            i = 10
            return i + 15
        if not self.created_bucket:
            return False
        insistent_bucket_delete(self.conn, self.bucket_name, self.keys)
        return False