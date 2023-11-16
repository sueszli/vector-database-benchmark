import io
from typing import Optional
from testfixtures import LogCapture
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from twisted.trial import unittest
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.files import FileException
from scrapy.pipelines.images import ImagesPipeline
from scrapy.pipelines.media import MediaPipeline
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.signal import disconnect_all
from scrapy.utils.test import get_crawler
try:
    from PIL import Image
except ImportError:
    skip_pillow: Optional[str] = 'Missing Python Imaging Library, install https://pypi.python.org/pypi/Pillow'
else:
    skip_pillow = None

def _mocked_download_func(request, info):
    if False:
        return 10
    assert request.callback is NO_CALLBACK
    response = request.meta.get('response')
    return response() if callable(response) else response

class BaseMediaPipelineTestCase(unittest.TestCase):
    pipeline_class = MediaPipeline
    settings = None

    def setUp(self):
        if False:
            print('Hello World!')
        spider_cls = Spider
        self.spider = spider_cls('media.com')
        crawler = get_crawler(spider_cls, self.settings)
        self.pipe = self.pipeline_class.from_crawler(crawler)
        self.pipe.download_func = _mocked_download_func
        self.pipe.open_spider(self.spider)
        self.info = self.pipe.spiderinfo
        self.fingerprint = crawler.request_fingerprinter.fingerprint

    def tearDown(self):
        if False:
            print('Hello World!')
        for (name, signal) in vars(signals).items():
            if not name.startswith('_'):
                disconnect_all(signal)

    def test_default_media_to_download(self):
        if False:
            i = 10
            return i + 15
        request = Request('http://url')
        assert self.pipe.media_to_download(request, self.info) is None

    def test_default_get_media_requests(self):
        if False:
            return 10
        item = dict(name='name')
        assert self.pipe.get_media_requests(item, self.info) is None

    def test_default_media_downloaded(self):
        if False:
            i = 10
            return i + 15
        request = Request('http://url')
        response = Response('http://url', body=b'')
        assert self.pipe.media_downloaded(response, request, self.info) is response

    def test_default_media_failed(self):
        if False:
            i = 10
            return i + 15
        request = Request('http://url')
        fail = Failure(Exception())
        assert self.pipe.media_failed(fail, request, self.info) is fail

    def test_default_item_completed(self):
        if False:
            print('Hello World!')
        item = dict(name='name')
        assert self.pipe.item_completed([], item, self.info) is item
        fail = Failure(Exception())
        results = [(True, 1), (False, fail)]
        with LogCapture() as log:
            new_item = self.pipe.item_completed(results, item, self.info)
        assert new_item is item
        assert len(log.records) == 1
        record = log.records[0]
        assert record.levelname == 'ERROR'
        self.assertTupleEqual(record.exc_info, failure_to_exc_info(fail))
        self.pipe.LOG_FAILED_RESULTS = False
        with LogCapture() as log:
            new_item = self.pipe.item_completed(results, item, self.info)
        assert new_item is item
        assert len(log.records) == 0

    @inlineCallbacks
    def test_default_process_item(self):
        if False:
            i = 10
            return i + 15
        item = dict(name='name')
        new_item = (yield self.pipe.process_item(item, self.spider))
        assert new_item is item

    def test_modify_media_request(self):
        if False:
            while True:
                i = 10
        request = Request('http://url')
        self.pipe._modify_media_request(request)
        assert request.meta == {'handle_httpstatus_all': True}

    def test_should_remove_req_res_references_before_caching_the_results(self):
        if False:
            return 10
        "Regression test case to prevent a memory leak in the Media Pipeline.\n\n        The memory leak is triggered when an exception is raised when a Response\n        scheduled by the Media Pipeline is being returned. For example, when a\n        FileException('download-error') is raised because the Response status\n        code is not 200 OK.\n\n        It happens because we are keeping a reference to the Response object\n        inside the FileException context. This is caused by the way Twisted\n        return values from inline callbacks. It raises a custom exception\n        encapsulating the original return value.\n\n        The solution is to remove the exception context when this context is a\n        _DefGen_Return instance, the BaseException used by Twisted to pass the\n        returned value from those inline callbacks.\n\n        Maybe there's a better and more reliable way to test the case described\n        here, but it would be more complicated and involve running - or at least\n        mocking - some async steps from the Media Pipeline. The current test\n        case is simple and detects the problem very fast. On the other hand, it\n        would not detect another kind of leak happening due to old object\n        references being kept inside the Media Pipeline cache.\n\n        This problem does not occur in Python 2.7 since we don't have Exception\n        Chaining (https://www.python.org/dev/peps/pep-3134/).\n        "
        request = Request('http://url')
        response = Response('http://url', body=b'', request=request)
        try:
            raise StopIteration(response)
        except StopIteration as exc:
            def_gen_return_exc = exc
            try:
                raise FileException('download-error')
            except Exception as exc:
                file_exc = exc
                failure = Failure(file_exc)
        self.assertEqual(failure.value, file_exc)
        self.assertEqual(failure.value.__context__, def_gen_return_exc)
        fp = self.fingerprint(request)
        info = self.pipe.spiderinfo
        info.downloading.add(fp)
        info.waiting[fp] = []
        self.pipe._cache_result_and_execute_waiters(failure, fp, info)
        self.assertEqual(info.downloaded[fp], failure)
        self.assertEqual(info.downloaded[fp].value, file_exc)
        context = getattr(info.downloaded[fp].value, '__context__', None)
        self.assertIsNone(context)

class MockedMediaPipeline(MediaPipeline):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._mockcalled = []

    def download(self, request, info):
        if False:
            for i in range(10):
                print('nop')
        self._mockcalled.append('download')
        return super().download(request, info)

    def media_to_download(self, request, info, *, item=None):
        if False:
            return 10
        self._mockcalled.append('media_to_download')
        if 'result' in request.meta:
            return request.meta.get('result')
        return super().media_to_download(request, info)

    def get_media_requests(self, item, info):
        if False:
            for i in range(10):
                print('nop')
        self._mockcalled.append('get_media_requests')
        return item.get('requests')

    def media_downloaded(self, response, request, info, *, item=None):
        if False:
            return 10
        self._mockcalled.append('media_downloaded')
        return super().media_downloaded(response, request, info)

    def media_failed(self, failure, request, info):
        if False:
            print('Hello World!')
        self._mockcalled.append('media_failed')
        return super().media_failed(failure, request, info)

    def item_completed(self, results, item, info):
        if False:
            for i in range(10):
                print('nop')
        self._mockcalled.append('item_completed')
        item = super().item_completed(results, item, info)
        item['results'] = results
        return item

class MediaPipelineTestCase(BaseMediaPipelineTestCase):
    pipeline_class = MockedMediaPipeline

    def _callback(self, result):
        if False:
            i = 10
            return i + 15
        self.pipe._mockcalled.append('request_callback')
        return result

    def _errback(self, result):
        if False:
            for i in range(10):
                print('nop')
        self.pipe._mockcalled.append('request_errback')
        return result

    @inlineCallbacks
    def test_result_succeed(self):
        if False:
            i = 10
            return i + 15
        rsp = Response('http://url1')
        req = Request('http://url1', meta=dict(response=rsp), callback=self._callback, errback=self._errback)
        item = dict(requests=req)
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertEqual(new_item['results'], [(True, rsp)])
        self.assertEqual(self.pipe._mockcalled, ['get_media_requests', 'media_to_download', 'media_downloaded', 'request_callback', 'item_completed'])

    @inlineCallbacks
    def test_result_failure(self):
        if False:
            print('Hello World!')
        self.pipe.LOG_FAILED_RESULTS = False
        fail = Failure(Exception())
        req = Request('http://url1', meta=dict(response=fail), callback=self._callback, errback=self._errback)
        item = dict(requests=req)
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertEqual(new_item['results'], [(False, fail)])
        self.assertEqual(self.pipe._mockcalled, ['get_media_requests', 'media_to_download', 'media_failed', 'request_errback', 'item_completed'])

    @inlineCallbacks
    def test_mix_of_success_and_failure(self):
        if False:
            print('Hello World!')
        self.pipe.LOG_FAILED_RESULTS = False
        rsp1 = Response('http://url1')
        req1 = Request('http://url1', meta=dict(response=rsp1))
        fail = Failure(Exception())
        req2 = Request('http://url2', meta=dict(response=fail))
        item = dict(requests=[req1, req2])
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertEqual(new_item['results'], [(True, rsp1), (False, fail)])
        m = self.pipe._mockcalled
        self.assertEqual(m[0], 'get_media_requests')
        self.assertEqual(m.count('get_media_requests'), 1)
        self.assertEqual(m.count('item_completed'), 1)
        self.assertEqual(m[-1], 'item_completed')
        self.assertEqual(m.count('media_to_download'), 2)
        self.assertEqual(m.count('media_downloaded'), 1)
        self.assertEqual(m.count('media_failed'), 1)

    @inlineCallbacks
    def test_get_media_requests(self):
        if False:
            print('Hello World!')
        req = Request('http://url')
        item = dict(requests=req)
        new_item = (yield self.pipe.process_item(item, self.spider))
        assert new_item is item
        self.assertIn(self.fingerprint(req), self.info.downloaded)
        req1 = Request('http://url1')
        req2 = Request('http://url2')
        item = dict(requests=iter([req1, req2]))
        new_item = (yield self.pipe.process_item(item, self.spider))
        assert new_item is item
        assert self.fingerprint(req1) in self.info.downloaded
        assert self.fingerprint(req2) in self.info.downloaded

    @inlineCallbacks
    def test_results_are_cached_across_multiple_items(self):
        if False:
            for i in range(10):
                print('nop')
        rsp1 = Response('http://url1')
        req1 = Request('http://url1', meta=dict(response=rsp1))
        item = dict(requests=req1)
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertTrue(new_item is item)
        self.assertEqual(new_item['results'], [(True, rsp1)])
        req2 = Request(req1.url, meta=dict(response=Response('http://donot.download.me')))
        item = dict(requests=req2)
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertTrue(new_item is item)
        self.assertEqual(self.fingerprint(req1), self.fingerprint(req2))
        self.assertEqual(new_item['results'], [(True, rsp1)])

    @inlineCallbacks
    def test_results_are_cached_for_requests_of_single_item(self):
        if False:
            i = 10
            return i + 15
        rsp1 = Response('http://url1')
        req1 = Request('http://url1', meta=dict(response=rsp1))
        req2 = Request(req1.url, meta=dict(response=Response('http://donot.download.me')))
        item = dict(requests=[req1, req2])
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertTrue(new_item is item)
        self.assertEqual(new_item['results'], [(True, rsp1), (True, rsp1)])

    @inlineCallbacks
    def test_wait_if_request_is_downloading(self):
        if False:
            return 10

        def _check_downloading(response):
            if False:
                return 10
            fp = self.fingerprint(req1)
            self.assertTrue(fp in self.info.downloading)
            self.assertTrue(fp in self.info.waiting)
            self.assertTrue(fp not in self.info.downloaded)
            self.assertEqual(len(self.info.waiting[fp]), 2)
            return response
        rsp1 = Response('http://url')

        def rsp1_func():
            if False:
                return 10
            dfd = Deferred().addCallback(_check_downloading)
            reactor.callLater(0.1, dfd.callback, rsp1)
            return dfd

        def rsp2_func():
            if False:
                while True:
                    i = 10
            self.fail('it must cache rsp1 result and must not try to redownload')
        req1 = Request('http://url', meta=dict(response=rsp1_func))
        req2 = Request(req1.url, meta=dict(response=rsp2_func))
        item = dict(requests=[req1, req2])
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertEqual(new_item['results'], [(True, rsp1), (True, rsp1)])

    @inlineCallbacks
    def test_use_media_to_download_result(self):
        if False:
            i = 10
            return i + 15
        req = Request('http://url', meta=dict(result='ITSME', response=self.fail))
        item = dict(requests=req)
        new_item = (yield self.pipe.process_item(item, self.spider))
        self.assertEqual(new_item['results'], [(True, 'ITSME')])
        self.assertEqual(self.pipe._mockcalled, ['get_media_requests', 'media_to_download', 'item_completed'])

class MockedMediaPipelineDeprecatedMethods(ImagesPipeline):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._mockcalled = []

    def get_media_requests(self, item, info):
        if False:
            while True:
                i = 10
        item_url = item['image_urls'][0]
        output_img = io.BytesIO()
        img = Image.new('RGB', (60, 30), color='red')
        img.save(output_img, format='JPEG')
        return Request(item_url, meta={'response': Response(item_url, status=200, body=output_img.getvalue())})

    def inc_stats(self, *args, **kwargs):
        if False:
            return 10
        return True

    def media_to_download(self, request, info):
        if False:
            while True:
                i = 10
        self._mockcalled.append('media_to_download')
        return super().media_to_download(request, info)

    def media_downloaded(self, response, request, info):
        if False:
            print('Hello World!')
        self._mockcalled.append('media_downloaded')
        return super().media_downloaded(response, request, info)

    def file_downloaded(self, response, request, info):
        if False:
            return 10
        self._mockcalled.append('file_downloaded')
        return super().file_downloaded(response, request, info)

    def file_path(self, request, response=None, info=None):
        if False:
            i = 10
            return i + 15
        self._mockcalled.append('file_path')
        return super().file_path(request, response, info)

    def thumb_path(self, request, thumb_id, response=None, info=None):
        if False:
            for i in range(10):
                print('nop')
        self._mockcalled.append('thumb_path')
        return super().thumb_path(request, thumb_id, response, info)

    def get_images(self, response, request, info):
        if False:
            i = 10
            return i + 15
        self._mockcalled.append('get_images')
        return super().get_images(response, request, info)

    def image_downloaded(self, response, request, info):
        if False:
            for i in range(10):
                print('nop')
        self._mockcalled.append('image_downloaded')
        return super().image_downloaded(response, request, info)

class MediaPipelineAllowRedirectSettingsTestCase(unittest.TestCase):

    def _assert_request_no3xx(self, pipeline_class, settings):
        if False:
            while True:
                i = 10
        pipe = pipeline_class(settings=Settings(settings))
        request = Request('http://url')
        pipe._modify_media_request(request)
        self.assertIn('handle_httpstatus_list', request.meta)
        for (status, check) in [(200, True), (301, False), (302, False), (302, False), (307, False), (308, False), (400, True), (404, True), (500, True)]:
            if check:
                self.assertIn(status, request.meta['handle_httpstatus_list'])
            else:
                self.assertNotIn(status, request.meta['handle_httpstatus_list'])

    def test_standard_setting(self):
        if False:
            return 10
        self._assert_request_no3xx(MediaPipeline, {'MEDIA_ALLOW_REDIRECTS': True})

    def test_subclass_standard_setting(self):
        if False:
            while True:
                i = 10

        class UserDefinedPipeline(MediaPipeline):
            pass
        self._assert_request_no3xx(UserDefinedPipeline, {'MEDIA_ALLOW_REDIRECTS': True})

    def test_subclass_specific_setting(self):
        if False:
            print('Hello World!')

        class UserDefinedPipeline(MediaPipeline):
            pass
        self._assert_request_no3xx(UserDefinedPipeline, {'USERDEFINEDPIPELINE_MEDIA_ALLOW_REDIRECTS': True})