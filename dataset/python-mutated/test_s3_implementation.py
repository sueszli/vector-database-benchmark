import pytest
from boto.s3 import connection
from wal_e import exception
from wal_e.blobstore.s3 import calling_format

def test_https():
    if False:
        for i in range(10):
            print('nop')
    for proto in ['http', 'https']:
        for convention in ['virtualhost', 'path', 'subdomain']:
            opts = calling_format._s3connection_opts_from_uri('{0}+{1}://'.format(proto, convention))
            assert (proto == 'https') == opts['is_secure']
            assert (proto == 'http') == (not opts['is_secure'])
            cf = opts['calling_format']
            if convention == 'virtualhost':
                assert isinstance(cf, connection.VHostCallingFormat)
            elif convention == 'path':
                assert isinstance(cf, connection.OrdinaryCallingFormat)
            elif convention == 'subdomain':
                assert isinstance(cf, connection.SubdomainCallingFormat)
            else:
                assert False

def test_bad_proto():
    if False:
        return 10
    with pytest.raises(exception.UserException):
        calling_format._s3connection_opts_from_uri('nope+virtualhost://')

def test_bad_scheme():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(exception.UserException):
        calling_format._s3connection_opts_from_uri('https+nope://')

def test_port():
    if False:
        while True:
            i = 10
    opts = calling_format._s3connection_opts_from_uri('https+path://localhost:443')
    assert opts['port'] == 443

def test_reject_auth():
    if False:
        i = 10
        return i + 15
    for username in ['', 'hello']:
        for password in ['', 'world']:
            with pytest.raises(exception.UserException):
                calling_format._s3connection_opts_from_uri('https+path://{0}:{1}@localhost:443'.format(username, password))

def test_reject_path():
    if False:
        i = 10
        return i + 15
    with pytest.raises(exception.UserException):
        calling_format._s3connection_opts_from_uri('https+path://localhost/hello')

def test_reject_query():
    if False:
        i = 10
        return i + 15
    with pytest.raises(exception.UserException):
        calling_format._s3connection_opts_from_uri('https+path://localhost?q=world')

def test_reject_fragment():
    if False:
        return 10
    with pytest.raises(exception.UserException):
        print(calling_format._s3connection_opts_from_uri('https+path://localhost#hello'))