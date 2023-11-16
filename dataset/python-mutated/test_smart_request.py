"""Tests for smart server request infrastructure (bzrlib.smart.request)."""
import threading
from bzrlib import errors, transport
from bzrlib.bzrdir import BzrDir
from bzrlib.smart import request
from bzrlib.tests import TestCase, TestCaseWithMemoryTransport

class NoBodyRequest(request.SmartServerRequest):
    """A request that does not implement do_body."""

    def do(self):
        if False:
            i = 10
            return i + 15
        return request.SuccessfulSmartServerResponse(('ok',))

class DoErrorRequest(request.SmartServerRequest):
    """A request that raises an error from self.do()."""

    def do(self):
        if False:
            print('Hello World!')
        raise errors.NoSuchFile('xyzzy')

class DoUnexpectedErrorRequest(request.SmartServerRequest):
    """A request that encounters a generic error in self.do()"""

    def do(self):
        if False:
            for i in range(10):
                print('nop')
        dict()[1]

class ChunkErrorRequest(request.SmartServerRequest):
    """A request that raises an error from self.do_chunk()."""

    def do(self):
        if False:
            while True:
                i = 10
        'No-op.'
        pass

    def do_chunk(self, bytes):
        if False:
            i = 10
            return i + 15
        raise errors.NoSuchFile('xyzzy')

class EndErrorRequest(request.SmartServerRequest):
    """A request that raises an error from self.do_end()."""

    def do(self):
        if False:
            while True:
                i = 10
        'No-op.'
        pass

    def do_chunk(self, bytes):
        if False:
            while True:
                i = 10
        'No-op.'
        pass

    def do_end(self):
        if False:
            for i in range(10):
                print('nop')
        raise errors.NoSuchFile('xyzzy')

class CheckJailRequest(request.SmartServerRequest):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        request.SmartServerRequest.__init__(self, *args)
        self.jail_transports_log = []

    def do(self):
        if False:
            while True:
                i = 10
        self.jail_transports_log.append(request.jail_info.transports)

    def do_chunk(self, bytes):
        if False:
            while True:
                i = 10
        self.jail_transports_log.append(request.jail_info.transports)

    def do_end(self):
        if False:
            for i in range(10):
                print('nop')
        self.jail_transports_log.append(request.jail_info.transports)

class TestSmartRequest(TestCase):

    def test_request_class_without_do_body(self):
        if False:
            i = 10
            return i + 15
        "If a request has no body data, and the request's implementation does\n        not override do_body, then no exception is raised.\n        "
        handler = request.SmartServerRequestHandler(None, {'foo': NoBodyRequest}, '/')
        handler.args_received(('foo',))
        handler.end_received()

    def test_only_request_code_is_jailed(self):
        if False:
            print('Hello World!')
        transport = 'dummy transport'
        handler = request.SmartServerRequestHandler(transport, {'foo': CheckJailRequest}, '/')
        handler.args_received(('foo',))
        self.assertEqual(None, request.jail_info.transports)
        handler.accept_body('bytes')
        self.assertEqual(None, request.jail_info.transports)
        handler.end_received()
        self.assertEqual(None, request.jail_info.transports)
        self.assertEqual([[transport]] * 3, handler._command.jail_transports_log)

    def test_all_registered_requests_are_safety_qualified(self):
        if False:
            return 10
        unclassified_requests = []
        allowed_info = ('read', 'idem', 'mutate', 'semivfs', 'semi', 'stream')
        for key in request.request_handlers.keys():
            info = request.request_handlers.get_info(key)
            if info is None or info not in allowed_info:
                unclassified_requests.append(key)
        if unclassified_requests:
            self.fail('These requests were not categorized as safe/unsafe to retry: %s' % (unclassified_requests,))

class TestSmartRequestHandlerErrorTranslation(TestCase):
    """Tests that SmartServerRequestHandler will translate exceptions raised by
    a SmartServerRequest into FailedSmartServerResponses.
    """

    def assertNoResponse(self, handler):
        if False:
            print('Hello World!')
        self.assertEqual(None, handler.response)

    def assertResponseIsTranslatedError(self, handler):
        if False:
            return 10
        expected_translation = ('NoSuchFile', 'xyzzy')
        self.assertEqual(request.FailedSmartServerResponse(expected_translation), handler.response)

    def test_error_translation_from_args_received(self):
        if False:
            print('Hello World!')
        handler = request.SmartServerRequestHandler(None, {'foo': DoErrorRequest}, '/')
        handler.args_received(('foo',))
        self.assertResponseIsTranslatedError(handler)

    def test_error_translation_from_chunk_received(self):
        if False:
            while True:
                i = 10
        handler = request.SmartServerRequestHandler(None, {'foo': ChunkErrorRequest}, '/')
        handler.args_received(('foo',))
        self.assertNoResponse(handler)
        handler.accept_body('bytes')
        self.assertResponseIsTranslatedError(handler)

    def test_error_translation_from_end_received(self):
        if False:
            i = 10
            return i + 15
        handler = request.SmartServerRequestHandler(None, {'foo': EndErrorRequest}, '/')
        handler.args_received(('foo',))
        self.assertNoResponse(handler)
        handler.end_received()
        self.assertResponseIsTranslatedError(handler)

    def test_unexpected_error_translation(self):
        if False:
            i = 10
            return i + 15
        handler = request.SmartServerRequestHandler(None, {'foo': DoUnexpectedErrorRequest}, '/')
        handler.args_received(('foo',))
        self.assertEqual(request.FailedSmartServerResponse(('error', 'KeyError', '1')), handler.response)

class TestRequestHanderErrorTranslation(TestCase):
    """Tests for bzrlib.smart.request._translate_error."""

    def assertTranslationEqual(self, expected_tuple, error):
        if False:
            i = 10
            return i + 15
        self.assertEqual(expected_tuple, request._translate_error(error))

    def test_NoSuchFile(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTranslationEqual(('NoSuchFile', 'path'), errors.NoSuchFile('path'))

    def test_LockContention(self):
        if False:
            print('Hello World!')
        self.assertTranslationEqual(('LockContention',), errors.LockContention('lock', 'msg'))

    def test_TokenMismatch(self):
        if False:
            i = 10
            return i + 15
        self.assertTranslationEqual(('TokenMismatch', 'some-token', 'actual-token'), errors.TokenMismatch('some-token', 'actual-token'))

    def test_MemoryError(self):
        if False:
            print('Hello World!')
        self.assertTranslationEqual(('MemoryError',), MemoryError())

    def test_generic_Exception(self):
        if False:
            while True:
                i = 10
        self.assertTranslationEqual(('error', 'Exception', ''), Exception())

    def test_generic_BzrError(self):
        if False:
            return 10
        self.assertTranslationEqual(('error', 'BzrError', 'some text'), errors.BzrError(msg='some text'))

    def test_generic_zlib_error(self):
        if False:
            i = 10
            return i + 15
        from zlib import error
        msg = 'Error -3 while decompressing data: incorrect data check'
        self.assertTranslationEqual(('error', 'zlib.error', msg), error(msg))

class TestRequestJail(TestCaseWithMemoryTransport):

    def test_jail(self):
        if False:
            for i in range(10):
                print('nop')
        transport = self.get_transport('blah')
        req = request.SmartServerRequest(transport)
        self.assertEqual(None, request.jail_info.transports)
        req.setup_jail()
        self.assertEqual([transport], request.jail_info.transports)
        req.teardown_jail()
        self.assertEqual(None, request.jail_info.transports)

class TestJailHook(TestCaseWithMemoryTransport):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestJailHook, self).setUp()

        def clear_jail_info():
            if False:
                for i in range(10):
                    print('nop')
            request.jail_info.transports = None
        self.addCleanup(clear_jail_info)

    def test_jail_hook(self):
        if False:
            return 10
        request.jail_info.transports = None
        _pre_open_hook = request._pre_open_hook
        t = self.get_transport('foo')
        _pre_open_hook(t)
        request.jail_info.transports = [t]
        _pre_open_hook(t)
        _pre_open_hook(t.clone('child'))
        self.assertRaises(errors.JailBreak, _pre_open_hook, t.clone('..'))
        self.assertRaises(errors.JailBreak, _pre_open_hook, transport.get_transport_from_url('http://host/'))

    def test_open_bzrdir_in_non_main_thread(self):
        if False:
            while True:
                i = 10
        'Opening a bzrdir in a non-main thread should work ok.\n        \n        This makes sure that the globally-installed\n        bzrlib.smart.request._pre_open_hook, which uses a threading.local(),\n        works in a newly created thread.\n        '
        bzrdir = self.make_bzrdir('.')
        transport = bzrdir.root_transport
        thread_result = []

        def t():
            if False:
                return 10
            BzrDir.open_from_transport(transport)
            thread_result.append('ok')
        thread = threading.Thread(target=t)
        thread.start()
        thread.join()
        self.assertEqual(['ok'], thread_result)