import collections
import gevent
import io
import socket
import traceback
try:
    from azure.common import AzureMissingResourceHttpError
except ImportError:
    from azure import WindowsAzureMissingResourceError as AzureMissingResourceHttpError
from . import calling_format
from azure.storage.blob.blockblobservice import BlockBlobService
from azure.storage.blob.models import ContentSettings
from urllib.parse import urlparse
from wal_e import log_help
from wal_e import files
from wal_e.pipeline import get_download_pipeline
from wal_e.piper import PIPE
from wal_e.retries import retry, retry_with_count
assert calling_format
logger = log_help.WalELogger(__name__)
_Key = collections.namedtuple('_Key', ['size'])
WABS_CHUNK_SIZE = 4 * 1024 * 1024

def uri_put_file(creds, uri, fp, content_type=None):
    if False:
        return 10
    assert fp.tell() == 0
    assert uri.startswith('wabs://')

    def log_upload_failures_on_error(exc_tup, exc_processor_cxt):
        if False:
            while True:
                i = 10

        def standard_detail_message(prefix=''):
            if False:
                print('Hello World!')
            return prefix + '  There have been {n} attempts to upload  file {url} so far.'.format(n=exc_processor_cxt, url=uri)
        (typ, value, tb) = exc_tup
        del exc_tup
        if issubclass(typ, socket.error):
            socketmsg = value[1] if isinstance(value, tuple) else value
            logger.info(msg='Retrying upload because of a socket error', detail=standard_detail_message("The socket error's message is '{0}'.".format(socketmsg)))
        else:
            logger.warning(msg='retrying file upload from unexpected exception', detail=standard_detail_message('The exception type is {etype} and its value is {evalue} and its traceback is {etraceback}'.format(etype=typ, evalue=value, etraceback=''.join(traceback.format_tb(tb)))))
        del tb
    url_tup = urlparse(uri)
    kwargs = dict(content_settings=ContentSettings(content_type), validate_content=True)
    conn = BlockBlobService(creds.account_name, creds.account_key, sas_token=creds.access_token, protocol='https')
    conn.create_blob_from_bytes(url_tup.netloc, url_tup.path.lstrip('/'), fp.read(), **kwargs)
    return _Key(size=fp.tell())

def uri_get_file(creds, uri, conn=None):
    if False:
        while True:
            i = 10
    assert uri.startswith('wabs://')
    url_tup = urlparse(uri)
    if conn is None:
        conn = BlockBlobService(creds.account_name, creds.account_key, sas_token=creds.access_token, protocol='https')
    data = io.BytesIO()
    conn.get_blob_to_stream(url_tup.netloc, url_tup.path.lstrip('/'), data)
    return data.getvalue()

def do_lzop_get(creds, url, path, decrypt, do_retry=True):
    if False:
        return 10
    '\n    Get and decompress a WABS URL\n\n    This streams the content directly to lzop; the compressed version\n    is never stored on disk.\n\n    '
    assert url.endswith('.lzo'), 'Expect an lzop-compressed file'
    assert url.startswith('wabs://')
    conn = BlockBlobService(creds.account_name, creds.account_key, sas_token=creds.access_token, protocol='https')

    def log_wal_fetch_failures_on_error(exc_tup, exc_processor_cxt):
        if False:
            print('Hello World!')

        def standard_detail_message(prefix=''):
            if False:
                for i in range(10):
                    print('nop')
            return prefix + '  There have been {n} attempts to fetch wal file {url} so far.'.format(n=exc_processor_cxt, url=url)
        (typ, value, tb) = exc_tup
        del exc_tup
        if issubclass(typ, socket.error):
            socketmsg = value[1] if isinstance(value, tuple) else value
            logger.info(msg='Retrying fetch because of a socket error', detail=standard_detail_message("The socket error's message is '{0}'.".format(socketmsg)))
        else:
            logger.warning(msg='retrying WAL file fetch from unexpected exception', detail=standard_detail_message('The exception type is {etype} and its value is {evalue} and its traceback is {etraceback}'.format(etype=typ, evalue=value, etraceback=''.join(traceback.format_tb(tb)))))
        del tb

    def download():
        if False:
            while True:
                i = 10
        with files.DeleteOnError(path) as decomp_out:
            with get_download_pipeline(PIPE, decomp_out.f, decrypt) as pl:
                g = gevent.spawn(write_and_return_error, url, conn, pl.stdin)
                try:
                    exc = g.get()
                    if exc is not None:
                        raise exc
                except AzureMissingResourceHttpError:
                    pl.abort()
                    logger.warning(msg='could no longer locate object while performing wal restore', detail='The absolute URI that could not be located is {url}.'.format(url=url), hint='This can be normal when Postgres is trying to detect what timelines are available during restoration.')
                    decomp_out.remove_regardless = True
                    return False
            logger.info(msg='completed download and decompression', detail='Downloaded and decompressed "{url}" to "{path}"'.format(url=url, path=path))
        return True
    if do_retry:
        download = retry(retry_with_count(log_wal_fetch_failures_on_error))(download)
    return download()

def write_and_return_error(url, conn, stream):
    if False:
        while True:
            i = 10
    try:
        data = uri_get_file(None, url, conn=conn)
        stream.write(data)
        stream.flush()
    except Exception as e:
        return e
    finally:
        stream.close()