from urllib.parse import urlparse
import gevent
import os
import socket
import traceback
import boto
from . import calling_format
from wal_e import files
from wal_e import log_help
from wal_e.exception import UserException
from wal_e.pipeline import get_download_pipeline
from wal_e.piper import PIPE
from wal_e.retries import retry, retry_with_count
logger = log_help.WalELogger(__name__)
if not boto.config.has_option('Boto', 'http_socket_timeout'):
    if not boto.config.has_section('Boto'):
        boto.config.add_section('Boto')
    boto.config.set('Boto', 'http_socket_timeout', '5')

def _uri_to_key(creds, uri, conn=None):
    if False:
        print('Hello World!')
    assert uri.startswith('s3://')
    url_tup = urlparse(uri)
    bucket_name = url_tup.netloc
    cinfo = calling_format.from_store_name(bucket_name)
    if conn is None:
        conn = cinfo.connect(creds)
    bucket = boto.s3.bucket.Bucket(connection=conn, name=bucket_name)
    return boto.s3.key.Key(bucket=bucket, name=url_tup.path)

def uri_put_file(creds, uri, fp, content_type=None, conn=None):
    if False:
        return 10
    assert fp.tell() == 0
    k = _uri_to_key(creds, uri, conn=conn)
    if content_type is not None:
        k.content_type = content_type
    storage_class = os.getenv('WALE_S3_STORAGE_CLASS', 'STANDARD')
    k.set_contents_from_file(fp, encrypt_key=True, headers={'x-amz-storage-class': storage_class})
    return k

def uri_get_file(creds, uri, conn=None):
    if False:
        while True:
            i = 10
    k = _uri_to_key(creds, uri, conn=conn)
    return k.get_contents_as_string()

def do_lzop_get(creds, url, path, decrypt, do_retry=True):
    if False:
        return 10
    '\n    Get and decompress a S3 URL\n\n    This streams the content directly to lzop; the compressed version\n    is never stored on disk.\n\n    '
    assert url.endswith('.lzo'), 'Expect an lzop-compressed file'

    def log_wal_fetch_failures_on_error(exc_tup, exc_processor_cxt):
        if False:
            while True:
                i = 10

        def standard_detail_message(prefix=''):
            if False:
                return 10
            return prefix + '  There have been {n} attempts to fetch wal file {url} so far.'.format(n=exc_processor_cxt, url=url)
        (typ, value, tb) = exc_tup
        del exc_tup
        if issubclass(typ, socket.error):
            socketmsg = value[1] if isinstance(value, tuple) else value
            logger.info(msg='Retrying fetch because of a socket error', detail=standard_detail_message("The socket error's message is '{0}'.".format(socketmsg)))
        elif issubclass(typ, boto.exception.S3ResponseError) and value.error_code == 'RequestTimeTooSkewed':
            logger.info(msg='Retrying fetch because of a Request Skew time', detail=standard_detail_message())
        else:
            logger.warning(msg='retrying WAL file fetch from unexpected exception', detail=standard_detail_message('The exception type is {etype} and its value is {evalue} and its traceback is {etraceback}'.format(etype=typ, evalue=value, etraceback=''.join(traceback.format_tb(tb)))))
        del tb

    def download():
        if False:
            for i in range(10):
                print('nop')
        with files.DeleteOnError(path) as decomp_out:
            key = _uri_to_key(creds, url)
            with get_download_pipeline(PIPE, decomp_out.f, decrypt) as pl:
                g = gevent.spawn(write_and_return_error, key, pl.stdin)
                try:
                    exc = g.get()
                    if exc is not None:
                        raise exc
                except boto.exception.S3ResponseError as e:
                    if e.status == 404:
                        pl.abort()
                        logger.info(msg='could no longer locate object while performing wal restore', detail='The absolute URI that could not be located is {url}.'.format(url=url), hint='This can be normal when Postgres is trying to detect what timelines are available during restoration.')
                        decomp_out.remove_regardless = True
                        return False
                    elif e.value.error_code == 'ExpiredToken':
                        pl.abort()
                        logger.info(msg='could no longer authenticate while performing wal restore', detail='The absolute URI that could not be accessed is {url}.'.format(url=url), hint='This can be normal when using STS credentials.')
                        decomp_out.remove_regardless = True
                        return False
                    else:
                        raise
            logger.info(msg='completed download and decompression', detail='Downloaded and decompressed "{url}" to "{path}"'.format(url=url, path=path))
        return True
    if do_retry:
        download = retry(retry_with_count(log_wal_fetch_failures_on_error))(download)
    return download()

def sigv4_check_apply():
    if False:
        print('Hello World!')
    region = os.getenv('AWS_REGION')
    endpoint = os.getenv('WALE_S3_ENDPOINT')
    if region and endpoint:
        logger.warning(msg='WALE_S3_ENDPOINT defined, ignoring AWS_REGION', hint='AWS_REGION is only intended for use with AWS S3, and not interface-compatible use cases supported by WALE_S3_ENDPOINT')
    elif region and (not endpoint):
        if not boto.config.has_option('s3', 'use-sigv4'):
            if not boto.config.has_section('s3'):
                boto.config.add_section('s3')
            boto.config.set('s3', 'use-sigv4', 'True')
    elif not region and endpoint:
        pass
    elif not region and (not endpoint):
        raise UserException(msg='must define one of AWS_REGION or WALE_S3_ENDPOINT', hint='AWS users will want to set AWS_REGION, those using alternative S3-compatible systems will want to use WALE_S3_ENDPOINT.')
    else:
        assert False

def write_and_return_error(key, stream):
    if False:
        for i in range(10):
            print('nop')
    try:
        key.get_contents_to_file(stream)
        stream.flush()
    except Exception as e:
        return e
    finally:
        stream.close()