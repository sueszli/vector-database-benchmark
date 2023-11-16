from google.cloud import exceptions
from google.cloud import storage
import base64
import hmac
import json
import os
import pytest
from wal_e.cmd import parse_boolean_envvar
MANGLE_SUFFIX = None

def bucket_name_mangle(bn, delimiter='-'):
    if False:
        print('Hello World!')
    global MANGLE_SUFFIX
    if MANGLE_SUFFIX is None:
        MANGLE_SUFFIX = compute_mangle_suffix()
    return bn + delimiter + MANGLE_SUFFIX

def compute_mangle_suffix():
    if False:
        i = 10
        return i + 15
    with open(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')) as f:
        cj = json.load(f)
        dm = hmac.new(b'wal-e-tests')
        dm.update(cj['client_id'].encode('utf-8'))
        dg = dm.digest()
        return base64.b32encode(dg[:10]).decode('utf-8').lower()

def no_real_gs_credentials():
    if False:
        return 10
    "Helps skip integration tests without live credentials.\n\n    Phrased in the negative to make it read better with 'skipif'.\n    "
    if parse_boolean_envvar(os.getenv('WALE_GS_INTEGRATION_TESTS')) is not True:
        return True
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is None:
        return True
    return False

def prepare_gs_default_test_bucket():
    if False:
        while True:
            i = 10
    if no_real_gs_credentials():
        assert False
    bucket_name = bucket_name_mangle('waletdefwuy', delimiter='')
    conn = storage.Client()

    def _clean():
        if False:
            return 10
        bucket = conn.get_bucket(bucket_name)
        for blob in bucket.list_blobs():
            try:
                bucket.delete_blob(blob.path)
            except exceptions.NotFound:
                pass
    try:
        conn.create_bucket(bucket_name)
    except exceptions.Conflict:
        pass
    _clean()
    return bucket_name

@pytest.fixture(scope='session')
def default_test_gs_bucket():
    if False:
        print('Hello World!')
    if not no_real_gs_credentials():
        return prepare_gs_default_test_bucket()

def apathetic_bucket_delete(bucket_name, blobs, *args, **kwargs):
    if False:
        print('Hello World!')
    conn = storage.Client()
    bucket = storage.Bucket(conn, name=bucket_name)
    if bucket:
        bucket.delete_blobs(blobs)
    try:
        bucket.delete()
    except exceptions.NotFound:
        pass
    return conn

def insistent_bucket_delete(conn, bucket_name, blobs):
    if False:
        while True:
            i = 10
    bucket = conn.get_bucket(bucket_name)
    if bucket:
        bucket.delete_blobs(blobs)
    while True:
        try:
            bucket.delete()
        except exceptions.NotFound:
            continue
        break

def insistent_bucket_create(conn, bucket_name, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    while True:
        try:
            bucket = conn.create_bucket(bucket_name, *args, **kwargs)
        except exceptions.Conflict:
            continue
        return bucket

class FreshBucket(object):

    def __init__(self, bucket_name, blobs=[], *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.bucket_name = bucket_name
        self.blobs = blobs
        self.conn_args = args
        self.conn_kwargs = kwargs
        self.created_bucket = False

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.conn = apathetic_bucket_delete(self.bucket_name, self.blobs, *self.conn_args, **self.conn_kwargs)
        return self

    def create(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        bucket = insistent_bucket_create(self.conn, self.bucket_name, *args, **kwargs)
        self.created_bucket = True
        return bucket

    def __exit__(self, typ, value, traceback):
        if False:
            i = 10
            return i + 15
        if not self.created_bucket:
            return False
        insistent_bucket_delete(self.conn, self.bucket_name, self.blobs)
        return False