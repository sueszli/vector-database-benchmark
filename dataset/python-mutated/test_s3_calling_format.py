import boto
import inspect
import os
import pytest
import wal_e.exception
from boto.s3 import connection
from s3_integration_help import FreshBucket, bucket_name_mangle, no_real_s3_credentials
from wal_e.blobstore.s3 import Credentials
from wal_e.blobstore.s3 import calling_format
from wal_e.blobstore.s3.calling_format import _is_mostly_subdomain_compatible, _is_ipv4_like
SUBDOMAIN_BOGUS = ['1.2.3.4', 'myawsbucket.', 'myawsbucket-.', 'my.-awsbucket', '.myawsbucket', 'myawsbucket-', '-myawsbucket', 'my_awsbucket', 'my..examplebucket', 'sh', 'long' * 30]
SUBDOMAIN_OK = ['myawsbucket', 'my-aws-bucket', 'myawsbucket.1', 'my.aws.bucket']
no_real_s3_credentials = no_real_s3_credentials

def test_subdomain_detect():
    if False:
        i = 10
        return i + 15
    'Exercise subdomain compatible/incompatible bucket names.'
    for bn in SUBDOMAIN_OK:
        assert _is_mostly_subdomain_compatible(bn) is True
    for bn in SUBDOMAIN_BOGUS:
        assert _is_mostly_subdomain_compatible(bn) is False

def test_bogus_region(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setenv('AWS_REGION', 'not-a-valid-region-name')
    with pytest.raises(wal_e.exception.UserException) as e:
        calling_format.from_store_name('forces.OrdinaryCallingFormat')
    assert e.value.msg == 'Could not resolve host for AWS_REGION'
    assert e.value.detail == 'AWS_REGION is set to "not-a-valid-region-name".'
    monkeypatch.setenv('AWS_REGION', 'not-a-valid-region-name')
    calling_format.from_store_name('subdomain-format-acceptable')

def test_cert_validation_sensitivity(monkeypatch):
    if False:
        return 10
    'Test degradation of dotted bucket names to OrdinaryCallingFormat\n\n    Although legal bucket names with SubdomainCallingFormat, these\n    kinds of bucket names run afoul certification validation, and so\n    they are forced to fall back to OrdinaryCallingFormat.\n    '
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    for bn in SUBDOMAIN_OK:
        if '.' not in bn:
            cinfo = calling_format.from_store_name(bn)
            assert cinfo.calling_format == boto.s3.connection.SubdomainCallingFormat
        else:
            assert '.' in bn
            cinfo = calling_format.from_store_name(bn)
            assert cinfo.calling_format == connection.OrdinaryCallingFormat
            assert cinfo.region == 'us-east-1'
            assert cinfo.ordinary_endpoint == 's3.amazonaws.com'

@pytest.mark.skipif('no_real_s3_credentials()')
def test_subdomain_compatible():
    if False:
        return 10
    'Exercise a case where connecting is region-oblivious.'
    creds = Credentials(os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'))
    bucket_name = bucket_name_mangle('wal-e-test-us-west-1-no-dots')
    cinfo = calling_format.from_store_name(bucket_name)
    with FreshBucket(bucket_name, host='s3-us-west-1.amazonaws.com', calling_format=connection.OrdinaryCallingFormat()) as fb:
        fb.create(location='us-west-1')
        conn = cinfo.connect(creds)
        assert cinfo.region is None
        assert cinfo.calling_format is connection.SubdomainCallingFormat
        assert isinstance(conn.calling_format, connection.SubdomainCallingFormat)

def test_ipv4_detect():
    if False:
        while True:
            i = 10
    'IPv4 lookalikes are not valid SubdomainCallingFormat names\n\n    Even though they otherwise follow the bucket naming rules,\n    IPv4-alike names are called out as specifically banned.\n    '
    assert _is_ipv4_like('1.1.1.1') is True
    assert _is_ipv4_like('1.1.1.256') is True
    assert _is_ipv4_like('-1.1.1.1') is True
    assert _is_ipv4_like('1.1.1.hello') is False
    assert _is_ipv4_like('hello') is False
    assert _is_ipv4_like('-1.1.1') is False
    assert _is_ipv4_like('-1.1.1.') is False

def test_str_repr_call_info(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Ensure CallingInfo renders sensibly.\n\n    Try a few cases sensitive to the bucket name.\n    '
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    cinfo = calling_format.from_store_name('hello-world')
    assert repr(cinfo) == str(cinfo)
    assert repr(cinfo) == "CallingInfo(hello-world, <class 'boto.s3.connection.SubdomainCallingFormat'>, 'us-east-1', None)"
    cinfo = calling_format.from_store_name('hello.world')
    assert repr(cinfo) == str(cinfo)
    assert repr(cinfo) == "CallingInfo(hello.world, <class 'boto.s3.connection.OrdinaryCallingFormat'>, 'us-east-1', 's3.amazonaws.com')"
    cinfo = calling_format.from_store_name('Hello-World')
    assert repr(cinfo) == str(cinfo)
    assert repr(cinfo) == "CallingInfo(Hello-World, <class 'boto.s3.connection.OrdinaryCallingFormat'>, 'us-east-1', 's3.amazonaws.com')"

@pytest.mark.skipif('no_real_s3_credentials()')
def test_cipher_suites():
    if False:
        while True:
            i = 10
    from wal_e import cmd
    assert cmd
    creds = Credentials(os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'))
    cinfo = calling_format.from_store_name('irrelevant', region='us-east-1')
    conn = cinfo.connect(creds)
    conn.get_all_buckets()
    spec = inspect.getargspec(conn._pool.get_http_connection)
    kw = {'host': 's3.amazonaws.com', 'is_secure': True}
    if 'port' in spec.args:
        kw['port'] = 443
    htcon = conn.new_http_connection(**kw)
    htcon.connect()
    chosen_cipher_suite = htcon.sock.cipher()[0].split('-')
    acceptable = [['AES256', 'SHA'], ['AES128', 'SHA'], ['ECDHE', 'RSA', 'AES128', 'SHA'], ['ECDHE', 'RSA', 'AES128', 'GCM', 'SHA256']]
    assert chosen_cipher_suite in acceptable