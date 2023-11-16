"""
Feed Exports extension

See documentation in docs/topics/feed-exports.rst
"""
import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse
from twisted.internet import defer, threads
from twisted.internet.defer import DeferredList
from w3lib.url import file_uri_to_path
from zope.interface import Interface, implementer
from scrapy import Spider, signals
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.extensions.postprocessing import PostProcessingManager
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.conf import feed_complete_default_values_from_settings
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import without_none_values
logger = logging.getLogger(__name__)
try:
    import boto3
    IS_BOTO3_AVAILABLE = True
except ImportError:
    IS_BOTO3_AVAILABLE = False

def build_storage(builder, uri, *args, feed_options=None, preargs=(), **kwargs):
    if False:
        for i in range(10):
            print('nop')
    kwargs['feed_options'] = feed_options
    return builder(*preargs, uri, *args, **kwargs)

class ItemFilter:
    """
    This will be used by FeedExporter to decide if an item should be allowed
    to be exported to a particular feed.

    :param feed_options: feed specific options passed from FeedExporter
    :type feed_options: dict
    """
    feed_options: Optional[dict]
    item_classes: Tuple

    def __init__(self, feed_options: Optional[dict]) -> None:
        if False:
            print('Hello World!')
        self.feed_options = feed_options
        if feed_options is not None:
            self.item_classes = tuple((load_object(item_class) for item_class in feed_options.get('item_classes') or ()))
        else:
            self.item_classes = tuple()

    def accepts(self, item: Any) -> bool:
        if False:
            return 10
        '\n        Return ``True`` if `item` should be exported or ``False`` otherwise.\n\n        :param item: scraped item which user wants to check if is acceptable\n        :type item: :ref:`Scrapy items <topics-items>`\n        :return: `True` if accepted, `False` otherwise\n        :rtype: bool\n        '
        if self.item_classes:
            return isinstance(item, self.item_classes)
        return True

class IFeedStorage(Interface):
    """Interface that all Feed Storages must implement"""

    def __init__(uri, *, feed_options=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the storage with the parameters given in the URI and the\n        feed-specific options (see :setting:`FEEDS`)'

    def open(spider):
        if False:
            print('Hello World!')
        'Open the storage for the given spider. It must return a file-like\n        object that will be used for the exporters'

    def store(file):
        if False:
            print('Hello World!')
        'Store the given file stream'

@implementer(IFeedStorage)
class BlockingFeedStorage:

    def open(self, spider):
        if False:
            for i in range(10):
                print('nop')
        path = spider.crawler.settings['FEED_TEMPDIR']
        if path and (not Path(path).is_dir()):
            raise OSError('Not a Directory: ' + str(path))
        return NamedTemporaryFile(prefix='feed-', dir=path)

    def store(self, file):
        if False:
            for i in range(10):
                print('nop')
        return threads.deferToThread(self._store_in_thread, file)

    def _store_in_thread(self, file):
        if False:
            while True:
                i = 10
        raise NotImplementedError

@implementer(IFeedStorage)
class StdoutFeedStorage:

    def __init__(self, uri, _stdout=None, *, feed_options=None):
        if False:
            for i in range(10):
                print('nop')
        if not _stdout:
            _stdout = sys.stdout.buffer
        self._stdout = _stdout
        if feed_options and feed_options.get('overwrite', False) is True:
            logger.warning('Standard output (stdout) storage does not support overwriting. To suppress this warning, remove the overwrite option from your FEEDS setting, or set it to False.')

    def open(self, spider):
        if False:
            print('Hello World!')
        return self._stdout

    def store(self, file):
        if False:
            while True:
                i = 10
        pass

@implementer(IFeedStorage)
class FileFeedStorage:

    def __init__(self, uri, *, feed_options=None):
        if False:
            print('Hello World!')
        self.path = file_uri_to_path(uri)
        feed_options = feed_options or {}
        self.write_mode = 'wb' if feed_options.get('overwrite', False) else 'ab'

    def open(self, spider) -> IO[Any]:
        if False:
            for i in range(10):
                print('nop')
        dirname = Path(self.path).parent
        if dirname and (not dirname.exists()):
            dirname.mkdir(parents=True)
        return Path(self.path).open(self.write_mode)

    def store(self, file):
        if False:
            print('Hello World!')
        file.close()

class S3FeedStorage(BlockingFeedStorage):

    def __init__(self, uri, access_key=None, secret_key=None, acl=None, endpoint_url=None, *, feed_options=None, session_token=None, region_name=None):
        if False:
            i = 10
            return i + 15
        if not is_botocore_available():
            raise NotConfigured('missing botocore library')
        u = urlparse(uri)
        self.bucketname = u.hostname
        self.access_key = u.username or access_key
        self.secret_key = u.password or secret_key
        self.session_token = session_token
        self.keyname = u.path[1:]
        self.acl = acl
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        if IS_BOTO3_AVAILABLE:
            import boto3.session
            session = boto3.session.Session()
            self.s3_client = session.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, aws_session_token=self.session_token, endpoint_url=self.endpoint_url, region_name=self.region_name)
        else:
            warnings.warn('`botocore` usage has been deprecated for S3 feed export, please use `boto3` to avoid problems', category=ScrapyDeprecationWarning)
            import botocore.session
            session = botocore.session.get_session()
            self.s3_client = session.create_client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, aws_session_token=self.session_token, endpoint_url=self.endpoint_url, region_name=self.region_name)
        if feed_options and feed_options.get('overwrite', True) is False:
            logger.warning('S3 does not support appending to files. To suppress this warning, remove the overwrite option from your FEEDS setting or set it to True.')

    @classmethod
    def from_crawler(cls, crawler, uri, *, feed_options=None):
        if False:
            return 10
        return build_storage(cls, uri, access_key=crawler.settings['AWS_ACCESS_KEY_ID'], secret_key=crawler.settings['AWS_SECRET_ACCESS_KEY'], session_token=crawler.settings['AWS_SESSION_TOKEN'], acl=crawler.settings['FEED_STORAGE_S3_ACL'] or None, endpoint_url=crawler.settings['AWS_ENDPOINT_URL'] or None, region_name=crawler.settings['AWS_REGION_NAME'] or None, feed_options=feed_options)

    def _store_in_thread(self, file):
        if False:
            return 10
        file.seek(0)
        if IS_BOTO3_AVAILABLE:
            kwargs = {'ExtraArgs': {'ACL': self.acl}} if self.acl else {}
            self.s3_client.upload_fileobj(Bucket=self.bucketname, Key=self.keyname, Fileobj=file, **kwargs)
        else:
            kwargs = {'ACL': self.acl} if self.acl else {}
            self.s3_client.put_object(Bucket=self.bucketname, Key=self.keyname, Body=file, **kwargs)
        file.close()

class GCSFeedStorage(BlockingFeedStorage):

    def __init__(self, uri, project_id, acl):
        if False:
            print('Hello World!')
        self.project_id = project_id
        self.acl = acl
        u = urlparse(uri)
        self.bucket_name = u.hostname
        self.blob_name = u.path[1:]

    @classmethod
    def from_crawler(cls, crawler, uri):
        if False:
            for i in range(10):
                print('nop')
        return cls(uri, crawler.settings['GCS_PROJECT_ID'], crawler.settings['FEED_STORAGE_GCS_ACL'] or None)

    def _store_in_thread(self, file):
        if False:
            while True:
                i = 10
        file.seek(0)
        from google.cloud.storage import Client
        client = Client(project=self.project_id)
        bucket = client.get_bucket(self.bucket_name)
        blob = bucket.blob(self.blob_name)
        blob.upload_from_file(file, predefined_acl=self.acl)

class FTPFeedStorage(BlockingFeedStorage):

    def __init__(self, uri: str, use_active_mode: bool=False, *, feed_options: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        u = urlparse(uri)
        if not u.hostname:
            raise ValueError(f'Got a storage URI without a hostname: {uri}')
        self.host: str = u.hostname
        self.port: int = int(u.port or '21')
        self.username: str = u.username or ''
        self.password: str = unquote(u.password or '')
        self.path: str = u.path
        self.use_active_mode: bool = use_active_mode
        self.overwrite: bool = not feed_options or feed_options.get('overwrite', True)

    @classmethod
    def from_crawler(cls, crawler, uri, *, feed_options=None):
        if False:
            return 10
        return build_storage(cls, uri, crawler.settings.getbool('FEED_STORAGE_FTP_ACTIVE'), feed_options=feed_options)

    def _store_in_thread(self, file):
        if False:
            for i in range(10):
                print('nop')
        ftp_store_file(path=self.path, file=file, host=self.host, port=self.port, username=self.username, password=self.password, use_active_mode=self.use_active_mode, overwrite=self.overwrite)

class FeedSlot:

    def __init__(self, storage, uri, format, store_empty, batch_id, uri_template, filter, feed_options, spider, exporters, settings, crawler):
        if False:
            for i in range(10):
                print('nop')
        self.file = None
        self.exporter = None
        self.storage = storage
        self.batch_id = batch_id
        self.format = format
        self.store_empty = store_empty
        self.uri_template = uri_template
        self.uri = uri
        self.filter = filter
        self.feed_options = feed_options
        self.spider = spider
        self.exporters = exporters
        self.settings = settings
        self.crawler = crawler
        self.itemcount = 0
        self._exporting = False
        self._fileloaded = False

    def start_exporting(self):
        if False:
            while True:
                i = 10
        if not self._fileloaded:
            self.file = self.storage.open(self.spider)
            if 'postprocessing' in self.feed_options:
                self.file = PostProcessingManager(self.feed_options['postprocessing'], self.file, self.feed_options)
            self.exporter = self._get_exporter(file=self.file, format=self.feed_options['format'], fields_to_export=self.feed_options['fields'], encoding=self.feed_options['encoding'], indent=self.feed_options['indent'], **self.feed_options['item_export_kwargs'])
            self._fileloaded = True
        if not self._exporting:
            self.exporter.start_exporting()
            self._exporting = True

    def _get_instance(self, objcls, *args, **kwargs):
        if False:
            print('Hello World!')
        return create_instance(objcls, self.settings, self.crawler, *args, **kwargs)

    def _get_exporter(self, file, format, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._get_instance(self.exporters[format], file, *args, **kwargs)

    def finish_exporting(self):
        if False:
            for i in range(10):
                print('nop')
        if self._exporting:
            self.exporter.finish_exporting()
            self._exporting = False
_FeedSlot = create_deprecated_class(name='_FeedSlot', new_class=FeedSlot)

class FeedExporter:
    _pending_deferreds: List[defer.Deferred] = []

    @classmethod
    def from_crawler(cls, crawler):
        if False:
            i = 10
            return i + 15
        exporter = cls(crawler)
        crawler.signals.connect(exporter.open_spider, signals.spider_opened)
        crawler.signals.connect(exporter.close_spider, signals.spider_closed)
        crawler.signals.connect(exporter.item_scraped, signals.item_scraped)
        return exporter

    def __init__(self, crawler):
        if False:
            while True:
                i = 10
        self.crawler = crawler
        self.settings = crawler.settings
        self.feeds = {}
        self.slots = []
        self.filters = {}
        if not self.settings['FEEDS'] and (not self.settings['FEED_URI']):
            raise NotConfigured
        if self.settings['FEED_URI']:
            warnings.warn('The `FEED_URI` and `FEED_FORMAT` settings have been deprecated in favor of the `FEEDS` setting. Please see the `FEEDS` setting docs for more details', category=ScrapyDeprecationWarning, stacklevel=2)
            uri = self.settings['FEED_URI']
            uri = str(uri) if not isinstance(uri, Path) else uri.absolute().as_uri()
            feed_options = {'format': self.settings.get('FEED_FORMAT', 'jsonlines')}
            self.feeds[uri] = feed_complete_default_values_from_settings(feed_options, self.settings)
            self.filters[uri] = self._load_filter(feed_options)
        for (uri, feed_options) in self.settings.getdict('FEEDS').items():
            uri = str(uri) if not isinstance(uri, Path) else uri.absolute().as_uri()
            self.feeds[uri] = feed_complete_default_values_from_settings(feed_options, self.settings)
            self.filters[uri] = self._load_filter(feed_options)
        self.storages = self._load_components('FEED_STORAGES')
        self.exporters = self._load_components('FEED_EXPORTERS')
        for (uri, feed_options) in self.feeds.items():
            if not self._storage_supported(uri, feed_options):
                raise NotConfigured
            if not self._settings_are_valid():
                raise NotConfigured
            if not self._exporter_supported(feed_options['format']):
                raise NotConfigured

    def open_spider(self, spider):
        if False:
            for i in range(10):
                print('nop')
        for (uri, feed_options) in self.feeds.items():
            uri_params = self._get_uri_params(spider, feed_options['uri_params'])
            self.slots.append(self._start_new_batch(batch_id=1, uri=uri % uri_params, feed_options=feed_options, spider=spider, uri_template=uri))

    async def close_spider(self, spider):
        for slot in self.slots:
            self._close_slot(slot, spider)
        if self._pending_deferreds:
            await maybe_deferred_to_future(DeferredList(self._pending_deferreds))
        await maybe_deferred_to_future(self.crawler.signals.send_catch_log_deferred(signals.feed_exporter_closed))

    def _close_slot(self, slot, spider):
        if False:
            while True:
                i = 10

        def get_file(slot_):
            if False:
                print('Hello World!')
            if isinstance(slot_.file, PostProcessingManager):
                slot_.file.close()
                return slot_.file.file
            return slot_.file
        if slot.itemcount:
            slot.finish_exporting()
        elif slot.store_empty and slot.batch_id == 1:
            slot.start_exporting()
            slot.finish_exporting()
        else:
            return None
        logmsg = f'{slot.format} feed ({slot.itemcount} items) in: {slot.uri}'
        d = defer.maybeDeferred(slot.storage.store, get_file(slot))
        d.addCallback(self._handle_store_success, logmsg, spider, type(slot.storage).__name__)
        d.addErrback(self._handle_store_error, logmsg, spider, type(slot.storage).__name__)
        self._pending_deferreds.append(d)
        d.addCallback(lambda _: self.crawler.signals.send_catch_log_deferred(signals.feed_slot_closed, slot=slot))
        d.addBoth(lambda _: self._pending_deferreds.remove(d))
        return d

    def _handle_store_error(self, f, logmsg, spider, slot_type):
        if False:
            while True:
                i = 10
        logger.error('Error storing %s', logmsg, exc_info=failure_to_exc_info(f), extra={'spider': spider})
        self.crawler.stats.inc_value(f'feedexport/failed_count/{slot_type}')

    def _handle_store_success(self, f, logmsg, spider, slot_type):
        if False:
            i = 10
            return i + 15
        logger.info('Stored %s', logmsg, extra={'spider': spider})
        self.crawler.stats.inc_value(f'feedexport/success_count/{slot_type}')

    def _start_new_batch(self, batch_id, uri, feed_options, spider, uri_template):
        if False:
            i = 10
            return i + 15
        '\n        Redirect the output data stream to a new file.\n        Execute multiple times if FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified\n        :param batch_id: sequence number of current batch\n        :param uri: uri of the new batch to start\n        :param feed_options: dict with parameters of feed\n        :param spider: user spider\n        :param uri_template: template of uri which contains %(batch_time)s or %(batch_id)d to create new uri\n        '
        storage = self._get_storage(uri, feed_options)
        slot = FeedSlot(storage=storage, uri=uri, format=feed_options['format'], store_empty=feed_options['store_empty'], batch_id=batch_id, uri_template=uri_template, filter=self.filters[uri_template], feed_options=feed_options, spider=spider, exporters=self.exporters, settings=self.settings, crawler=getattr(self, 'crawler', None))
        return slot

    def item_scraped(self, item, spider):
        if False:
            i = 10
            return i + 15
        slots = []
        for slot in self.slots:
            if not slot.filter.accepts(item):
                slots.append(slot)
                continue
            slot.start_exporting()
            slot.exporter.export_item(item)
            slot.itemcount += 1
            if self.feeds[slot.uri_template]['batch_item_count'] and slot.itemcount >= self.feeds[slot.uri_template]['batch_item_count']:
                uri_params = self._get_uri_params(spider, self.feeds[slot.uri_template]['uri_params'], slot)
                self._close_slot(slot, spider)
                slots.append(self._start_new_batch(batch_id=slot.batch_id + 1, uri=slot.uri_template % uri_params, feed_options=self.feeds[slot.uri_template], spider=spider, uri_template=slot.uri_template))
            else:
                slots.append(slot)
        self.slots = slots

    def _load_components(self, setting_prefix):
        if False:
            return 10
        conf = without_none_values(self.settings.getwithbase(setting_prefix))
        d = {}
        for (k, v) in conf.items():
            try:
                d[k] = load_object(v)
            except NotConfigured:
                pass
        return d

    def _exporter_supported(self, format):
        if False:
            i = 10
            return i + 15
        if format in self.exporters:
            return True
        logger.error('Unknown feed format: %(format)s', {'format': format})

    def _settings_are_valid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified uri has to contain\n        %(batch_time)s or %(batch_id)d to distinguish different files of partial output\n        '
        for (uri_template, values) in self.feeds.items():
            if values['batch_item_count'] and (not re.search('%\\(batch_time\\)s|%\\(batch_id\\)', uri_template)):
                logger.error('%%(batch_time)s or %%(batch_id)d must be in the feed URI (%s) if FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified and greater than 0. For more info see: https://docs.scrapy.org/en/latest/topics/feed-exports.html#feed-export-batch-item-count', uri_template)
                return False
        return True

    def _storage_supported(self, uri, feed_options):
        if False:
            while True:
                i = 10
        scheme = urlparse(uri).scheme
        if scheme in self.storages or PureWindowsPath(uri).drive:
            try:
                self._get_storage(uri, feed_options)
                return True
            except NotConfigured as e:
                logger.error('Disabled feed storage scheme: %(scheme)s. Reason: %(reason)s', {'scheme': scheme, 'reason': str(e)})
        else:
            logger.error('Unknown feed storage scheme: %(scheme)s', {'scheme': scheme})

    def _get_storage(self, uri, feed_options):
        if False:
            i = 10
            return i + 15
        'Fork of create_instance specific to feed storage classes\n\n        It supports not passing the *feed_options* parameters to classes that\n        do not support it, and issuing a deprecation warning instead.\n        '
        feedcls = self.storages.get(urlparse(uri).scheme, self.storages['file'])
        crawler = getattr(self, 'crawler', None)

        def build_instance(builder, *preargs):
            if False:
                return 10
            return build_storage(builder, uri, feed_options=feed_options, preargs=preargs)
        if crawler and hasattr(feedcls, 'from_crawler'):
            instance = build_instance(feedcls.from_crawler, crawler)
            method_name = 'from_crawler'
        elif hasattr(feedcls, 'from_settings'):
            instance = build_instance(feedcls.from_settings, self.settings)
            method_name = 'from_settings'
        else:
            instance = build_instance(feedcls)
            method_name = '__new__'
        if instance is None:
            raise TypeError(f'{feedcls.__qualname__}.{method_name} returned None')
        return instance

    def _get_uri_params(self, spider: Spider, uri_params_function: Optional[Union[str, Callable[[dict, Spider], dict]]], slot: Optional[FeedSlot]=None) -> dict:
        if False:
            return 10
        params = {}
        for k in dir(spider):
            params[k] = getattr(spider, k)
        utc_now = datetime.now(tz=timezone.utc)
        params['time'] = utc_now.replace(microsecond=0).isoformat().replace(':', '-')
        params['batch_time'] = utc_now.isoformat().replace(':', '-')
        params['batch_id'] = slot.batch_id + 1 if slot is not None else 1
        uripar_function = load_object(uri_params_function) if uri_params_function else lambda params, _: params
        new_params = uripar_function(params, spider)
        return new_params if new_params is not None else params

    def _load_filter(self, feed_options):
        if False:
            for i in range(10):
                print('nop')
        item_filter_class = load_object(feed_options.get('item_filter', ItemFilter))
        return item_filter_class(feed_options)