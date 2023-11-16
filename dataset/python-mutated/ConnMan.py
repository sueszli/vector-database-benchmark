from __future__ import absolute_import
import sys
if sys.version_info >= (3, 0):
    from .Custom_httplib3x import httplib
else:
    from .Custom_httplib27 import httplib
import ssl
from logging import debug
from threading import Semaphore
from time import time
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
from .Config import Config
from .Exceptions import ParameterError, S3SSLCertificateError
from .Utils import getBucketFromHostname
__all__ = ['ConnMan']

class http_connection(object):
    context = None
    context_set = False

    @staticmethod
    def _ssl_verified_context(cafile):
        if False:
            for i in range(10):
                print('nop')
        cfg = Config()
        context = None
        try:
            context = ssl.create_default_context(cafile=cafile)
        except AttributeError:
            pass
        if context and (not cfg.check_ssl_hostname):
            context.check_hostname = False
            debug(u'Disabling SSL certificate hostname checking')
        return context

    @staticmethod
    def _ssl_unverified_context(cafile):
        if False:
            while True:
                i = 10
        debug(u'Disabling SSL certificate checking')
        context = None
        try:
            context = ssl._create_unverified_context(cafile=cafile, cert_reqs=ssl.CERT_NONE)
        except AttributeError:
            pass
        return context

    @staticmethod
    def _ssl_client_auth_context(certfile, keyfile, check_server_cert, cafile):
        if False:
            for i in range(10):
                print('nop')
        context = None
        try:
            cert_reqs = ssl.CERT_REQUIRED if check_server_cert else ssl.CERT_NONE
            context = ssl._create_unverified_context(cafile=cafile, keyfile=keyfile, certfile=certfile, cert_reqs=cert_reqs)
        except AttributeError:
            pass
        return context

    @staticmethod
    def _ssl_context():
        if False:
            while True:
                i = 10
        if http_connection.context_set:
            return http_connection.context
        cfg = Config()
        cafile = cfg.ca_certs_file
        if cafile == '':
            cafile = None
        certfile = cfg.ssl_client_cert_file or None
        keyfile = cfg.ssl_client_key_file or None
        debug(u'Using ca_certs_file %s', cafile)
        debug(u'Using ssl_client_cert_file %s', certfile)
        debug(u'Using ssl_client_key_file %s', keyfile)
        if certfile is not None:
            context = http_connection._ssl_client_auth_context(certfile, keyfile, cfg.check_ssl_certificate, cafile)
        elif cfg.check_ssl_certificate:
            context = http_connection._ssl_verified_context(cafile)
        else:
            context = http_connection._ssl_unverified_context(cafile)
        http_connection.context = context
        http_connection.context_set = True
        return context

    def forgive_wildcard_cert(self, cert, hostname):
        if False:
            return 10
        '\n        Wildcard matching for *.s3.amazonaws.com and similar per region.\n\n        Per http://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html:\n        "We recommend that all bucket names comply with DNS naming conventions."\n\n        Per http://docs.aws.amazon.com/AmazonS3/latest/dev/VirtualHosting.html:\n        "When using virtual hosted-style buckets with SSL, the SSL\n        wild card certificate only matches buckets that do not contain\n        periods. To work around this, use HTTP or write your own\n        certificate verification logic."\n\n        Therefore, we need a custom validation routine that allows\n        mybucket.example.com.s3.amazonaws.com to be considered a valid\n        hostname for the *.s3.amazonaws.com wildcard cert, and for the\n        region-specific *.s3-[region].amazonaws.com wildcard cert.\n\n        We also forgive non-S3 wildcard certificates should the\n        hostname match, to allow compatibility with other S3\n        API-compatible storage providers.\n        '
        debug(u'checking SSL subjectAltName as forgiving wildcard cert')
        san = cert.get('subjectAltName', ())
        hostname = hostname.lower()
        cleaned_host_bucket_config = urlparse('https://' + Config.host_bucket).hostname
        for (key, value) in san:
            if key == 'DNS':
                value = value.lower()
                if value.startswith('*.s3') and (value.endswith('.amazonaws.com') and hostname.endswith('.amazonaws.com')) or (value.endswith('.amazonaws.com.cn') and hostname.endswith('.amazonaws.com.cn')):
                    return True
                elif value == cleaned_host_bucket_config % {'bucket': '*', 'location': Config.bucket_location.lower()} and hostname.endswith(cleaned_host_bucket_config % {'bucket': '', 'location': Config.bucket_location.lower()}):
                    return True
        return False

    def match_hostname(self):
        if False:
            return 10
        cert = self.c.sock.getpeercert()
        try:
            ssl.match_hostname(cert, self.hostname)
        except AttributeError:
            return
        except ValueError:
            return
        except S3CertificateError as e:
            if not self.forgive_wildcard_cert(cert, self.hostname):
                raise e

    @staticmethod
    def _https_connection(hostname, port=None):
        if False:
            print('Hello World!')
        try:
            context = http_connection._ssl_context()
            (bucket_name, success) = getBucketFromHostname(hostname)
            if success and '.' in bucket_name:
                debug(u'Bucket name contains "." character, disabling initial SSL hostname check')
                check_hostname = False
                if context:
                    context.check_hostname = False
            elif context:
                check_hostname = context.check_hostname
            else:
                check_hostname = True
            conn = httplib.HTTPSConnection(hostname, port, context=context, check_hostname=check_hostname)
            debug(u'httplib.HTTPSConnection() has both context and check_hostname')
        except TypeError:
            try:
                conn = httplib.HTTPSConnection(hostname, port, context=context)
                debug(u'httplib.HTTPSConnection() has only context')
            except TypeError:
                conn = httplib.HTTPSConnection(hostname, port)
                debug(u'httplib.HTTPSConnection() has neither context nor check_hostname')
        return conn

    def __init__(self, id, hostname, ssl, cfg):
        if False:
            for i in range(10):
                print('nop')
        self.ssl = ssl
        self.id = id
        self.counter = 0
        parsed_hostname = urlparse('https://' + hostname)
        self.hostname = parsed_hostname.hostname
        self.port = parsed_hostname.port
        if parsed_hostname.path and parsed_hostname.path != '/':
            self.path = parsed_hostname.path.rstrip('/')
            debug(u'endpoint path set to %s', self.path)
        else:
            self.path = None
        '\n        History note:\n        In a perfect world, or in the future:\n        - All http proxies would support CONNECT/tunnel, and so there would be no need\n        for using "absolute URIs" in format_uri.\n        - All s3-like servers would work well whether using relative or ABSOLUTE URIs.\n        But currently, what is currently common:\n        - Proxies without support for CONNECT for http, and so "absolute URIs" have to\n        be used.\n        - Proxies with support for CONNECT for httpS but s3-like servers having issues\n        with "absolute URIs", so relative one still have to be used as the requests will\n        pass as-is, through the proxy because of the CONNECT mode.\n        '
        if not cfg.proxy_host:
            if ssl:
                self.c = http_connection._https_connection(self.hostname, self.port)
                debug(u'non-proxied HTTPSConnection(%s, %s)', self.hostname, self.port)
            else:
                self.c = httplib.HTTPConnection(self.hostname, self.port)
                debug(u'non-proxied HTTPConnection(%s, %s)', self.hostname, self.port)
        elif ssl:
            self.c = http_connection._https_connection(cfg.proxy_host, cfg.proxy_port)
            debug(u'proxied HTTPSConnection(%s, %s)', cfg.proxy_host, cfg.proxy_port)
            port = self.port and self.port or 443
            self.c.set_tunnel(self.hostname, port)
            debug(u'tunnel to %s, %s', self.hostname, port)
        else:
            self.c = httplib.HTTPConnection(cfg.proxy_host, cfg.proxy_port)
            debug(u'proxied HTTPConnection(%s, %s)', cfg.proxy_host, cfg.proxy_port)
        self.last_used_time = time()

class ConnMan(object):
    _CS_REQ_SENT = httplib._CS_REQ_SENT
    CONTINUE = httplib.CONTINUE
    conn_pool_sem = Semaphore()
    conn_pool = {}
    conn_max_counter = 800

    @staticmethod
    def get(hostname, ssl=None):
        if False:
            while True:
                i = 10
        cfg = Config()
        if ssl is None:
            ssl = cfg.use_https
        conn = None
        if cfg.proxy_host != '':
            if ssl and sys.hexversion < 34013184:
                raise ParameterError("use_https=True can't be used with proxy on Python <2.7")
            conn_id = 'proxy://%s:%s' % (cfg.proxy_host, cfg.proxy_port)
        else:
            conn_id = 'http%s://%s' % (ssl and 's' or '', hostname)
        ConnMan.conn_pool_sem.acquire()
        if conn_id not in ConnMan.conn_pool:
            ConnMan.conn_pool[conn_id] = []
        while ConnMan.conn_pool[conn_id]:
            conn = ConnMan.conn_pool[conn_id].pop()
            cur_time = time()
            if cur_time < conn.last_used_time + cfg.connection_max_age and cur_time >= conn.last_used_time:
                debug('ConnMan.get(): re-using connection: %s#%d' % (conn.id, conn.counter))
                break
            debug('ConnMan.get(): closing expired connection')
            ConnMan.close(conn)
            conn = None
        ConnMan.conn_pool_sem.release()
        if not conn:
            debug('ConnMan.get(): creating new connection: %s' % conn_id)
            conn = http_connection(conn_id, hostname, ssl, cfg)
            conn.c.connect()
            if conn.ssl and cfg.check_ssl_certificate and cfg.check_ssl_hostname:
                conn.match_hostname()
        conn.counter += 1
        return conn

    @staticmethod
    def put(conn):
        if False:
            for i in range(10):
                print('nop')
        if conn.id.startswith('proxy://'):
            ConnMan.close(conn)
            debug('ConnMan.put(): closing proxy connection (keep-alive not yet supported)')
            return
        if conn.counter >= ConnMan.conn_max_counter:
            ConnMan.close(conn)
            debug('ConnMan.put(): closing over-used connection')
            return
        cfg = Config()
        if not cfg.connection_pooling:
            ConnMan.close(conn)
            debug('ConnMan.put(): closing connection (connection pooling disabled)')
            return
        conn.last_used_time = time()
        ConnMan.conn_pool_sem.acquire()
        ConnMan.conn_pool[conn.id].append(conn)
        ConnMan.conn_pool_sem.release()
        debug('ConnMan.put(): connection put back to pool (%s#%d)' % (conn.id, conn.counter))

    @staticmethod
    def close(conn):
        if False:
            print('Hello World!')
        if conn:
            conn.c.close()