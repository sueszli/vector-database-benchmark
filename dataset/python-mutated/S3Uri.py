from __future__ import absolute_import, print_function
import os
import re
import sys
from .Utils import unicodise, deunicodise, check_bucket_name_dns_support
from . import Config
PY3 = sys.version_info >= (3, 0)

class S3Uri(object):
    type = None
    _subclasses = None

    def __new__(self, string):
        if False:
            for i in range(10):
                print('nop')
        if not self._subclasses:
            self._subclasses = []
            dict = sys.modules[__name__].__dict__
            for something in dict:
                if type(dict[something]) is not type(self):
                    continue
                if issubclass(dict[something], self) and dict[something] != self:
                    self._subclasses.append(dict[something])
        for subclass in self._subclasses:
            try:
                instance = object.__new__(subclass)
                instance.__init__(string)
                return instance
            except ValueError:
                continue
        raise ValueError('%s: not a recognized URI' % string)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if PY3:
            return self.uri()
        else:
            return deunicodise(self.uri())

    def __unicode__(self):
        if False:
            print('Hello World!')
        return self.uri()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr('<%s: %s>' % (self.__class__.__name__, self.__unicode__()))

    def public_url(self):
        if False:
            i = 10
            return i + 15
        raise ValueError('This S3 URI does not have Anonymous URL representation')

    def basename(self):
        if False:
            while True:
                i = 10
        return self.__unicode__().split('/')[-1]

class S3UriS3(S3Uri):
    type = 's3'
    _re = re.compile('^s3:///*([^/]*)/?(.*)', re.IGNORECASE | re.UNICODE)

    def __init__(self, string):
        if False:
            while True:
                i = 10
        match = self._re.match(string)
        if not match:
            raise ValueError('%s: not a S3 URI' % string)
        groups = match.groups()
        self._bucket = groups[0]
        self._object = groups[1]

    def bucket(self):
        if False:
            print('Hello World!')
        return self._bucket

    def object(self):
        if False:
            for i in range(10):
                print('nop')
        return self._object

    def has_bucket(self):
        if False:
            while True:
                i = 10
        return bool(self._bucket)

    def has_object(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self._object)

    def uri(self):
        if False:
            return 10
        return u'/'.join([u's3:/', self._bucket, self._object])

    def is_dns_compatible(self):
        if False:
            while True:
                i = 10
        return check_bucket_name_dns_support(Config.Config().host_bucket, self._bucket)

    def public_url(self):
        if False:
            return 10
        public_url_protocol = 'http'
        if Config.Config().public_url_use_https:
            public_url_protocol = 'https'
        if self.is_dns_compatible():
            return '%s://%s.%s/%s' % (public_url_protocol, self._bucket, Config.Config().host_base, self._object)
        else:
            return '%s://%s/%s/%s' % (public_url_protocol, Config.Config().host_base, self._bucket, self._object)

    def host_name(self):
        if False:
            print('Hello World!')
        if self.is_dns_compatible():
            return '%s.s3.amazonaws.com' % self._bucket
        else:
            return 's3.amazonaws.com'

    @staticmethod
    def compose_uri(bucket, object=''):
        if False:
            i = 10
            return i + 15
        return u's3://%s/%s' % (bucket, object)

    @staticmethod
    def httpurl_to_s3uri(http_url):
        if False:
            while True:
                i = 10
        m = re.match('(https?://)?([^/]+)/?(.*)', http_url, re.IGNORECASE | re.UNICODE)
        (hostname, object) = m.groups()[1:]
        hostname = hostname.lower()
        m = re.match('(.*\\.)?s3(?:\\-[^\\.]*)?(?:\\.dualstack)?(?:\\.[^\\.]*)?\\.amazonaws\\.com(?:\\.cn)?$', hostname, re.IGNORECASE | re.UNICODE)
        if not m:
            raise ValueError('Unable to parse URL: %s' % http_url)
        bucket = m.groups()[0]
        if not bucket:
            if '/' not in object:
                bucket = object
                object = ''
            else:
                (bucket, object) = object.split('/', 1)
        else:
            bucket = bucket.rstrip('.')
        return S3Uri(u's3://%(bucket)s/%(object)s' % {'bucket': bucket, 'object': object})

class S3UriS3FS(S3Uri):
    type = 's3fs'
    _re = re.compile('^s3fs:///*([^/]*)/?(.*)', re.IGNORECASE | re.UNICODE)

    def __init__(self, string):
        if False:
            return 10
        match = self._re.match(string)
        if not match:
            raise ValueError('%s: not a S3fs URI' % string)
        groups = match.groups()
        self._fsname = groups[0]
        self._path = groups[1].split('/')

    def fsname(self):
        if False:
            print('Hello World!')
        return self._fsname

    def path(self):
        if False:
            while True:
                i = 10
        return '/'.join(self._path)

    def uri(self):
        if False:
            while True:
                i = 10
        return u'/'.join([u's3fs:/', self._fsname, self.path()])

class S3UriFile(S3Uri):
    type = 'file'
    _re = re.compile('^(\\w+://)?(.*)', re.UNICODE)

    def __init__(self, string):
        if False:
            return 10
        match = self._re.match(string)
        groups = match.groups()
        if groups[0] not in (None, 'file://'):
            raise ValueError('%s: not a file:// URI' % string)
        if groups[0] is None:
            self._path = groups[1].split(os.sep)
        else:
            self._path = groups[1].split('/')

    def path(self):
        if False:
            i = 10
            return i + 15
        return os.sep.join(self._path)

    def uri(self):
        if False:
            while True:
                i = 10
        return u'/'.join([u'file:/'] + self._path)

    def isdir(self):
        if False:
            return 10
        return os.path.isdir(deunicodise(self.path()))

    def dirname(self):
        if False:
            for i in range(10):
                print('nop')
        return unicodise(os.path.dirname(deunicodise(self.path())))

    def basename(self):
        if False:
            for i in range(10):
                print('nop')
        return unicodise(os.path.basename(deunicodise(self.path())))

class S3UriCloudFront(S3Uri):
    type = 'cf'
    _re = re.compile('^cf://([^/]*)/*(.*)', re.IGNORECASE | re.UNICODE)

    def __init__(self, string):
        if False:
            i = 10
            return i + 15
        match = self._re.match(string)
        if not match:
            raise ValueError('%s: not a CloudFront URI' % string)
        groups = match.groups()
        self._dist_id = groups[0]
        self._request_id = groups[1] != '/' and groups[1] or None

    def dist_id(self):
        if False:
            i = 10
            return i + 15
        return self._dist_id

    def request_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._request_id

    def uri(self):
        if False:
            while True:
                i = 10
        uri = u'cf://' + self.dist_id()
        if self.request_id():
            uri += u'/' + self.request_id()
        return uri
if __name__ == '__main__':
    uri = S3Uri('s3://bucket/object')
    print('type()  =', type(uri))
    print('uri     =', uri)
    print('uri.type=', uri.type)
    print('bucket  =', uri.bucket())
    print('object  =', uri.object())
    print()
    uri = S3Uri('s3://bucket')
    print('type()  =', type(uri))
    print('uri     =', uri)
    print('uri.type=', uri.type)
    print('bucket  =', uri.bucket())
    print()
    uri = S3Uri('s3fs://filesystem1/path/to/remote/file.txt')
    print('type()  =', type(uri))
    print('uri     =', uri)
    print('uri.type=', uri.type)
    print('path    =', uri.path())
    print()
    uri = S3Uri('/path/to/local/file.txt')
    print('type()  =', type(uri))
    print('uri     =', uri)
    print('uri.type=', uri.type)
    print('path    =', uri.path())
    print()
    uri = S3Uri('cf://1234567890ABCD/')
    print('type()  =', type(uri))
    print('uri     =', uri)
    print('uri.type=', uri.type)
    print('dist_id =', uri.dist_id())
    print()