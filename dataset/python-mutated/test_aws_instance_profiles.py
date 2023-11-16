import pytest
import boto
import boto.provider
from boto import utils
from wal_e.blobstore.s3 import s3_credentials
META_DATA_CREDENTIALS = {'Code': 'Success', 'LastUpdated': '2014-01-11T02:13:53Z', 'Type': 'AWS-HMAC', 'AccessKeyId': None, 'SecretAccessKey': None, 'Token': None, 'Expiration': '2014-01-11T08:16:59Z'}

def boto_flat_metadata():
    if False:
        while True:
            i = 10
    return tuple((int(x) for x in boto.__version__.split('.'))) >= (2, 9, 0)

@pytest.fixture()
def metadata(monkeypatch):
    if False:
        while True:
            i = 10
    m = dict(**META_DATA_CREDENTIALS)
    m['AccessKeyId'] = 'foo'
    m['SecretAccessKey'] = 'bar'
    m['Token'] = 'baz'
    monkeypatch.setattr(boto.provider.Provider, '_credentials_need_refresh', lambda self: False)
    if boto_flat_metadata():
        m = {'irrelevant': m}
    else:
        m = {'iam': {'security-credentials': {'irrelevant': m}}}
    monkeypatch.setattr(utils, 'get_instance_metadata', lambda *args, **kwargs: m)

def test_profile_provider(metadata):
    if False:
        print('Hello World!')
    ipp = s3_credentials.InstanceProfileCredentials()
    assert ipp.get_access_key() == 'foo'
    assert ipp.get_secret_key() == 'bar'
    assert ipp.get_security_token() == 'baz'