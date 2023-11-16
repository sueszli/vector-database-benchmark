import os
import signal
import threading
import weakref
from bzrlib import tests, transport
from bzrlib.smart import client, medium, server, signals
SIGHUP = getattr(signal, 'SIGHUP', 1)

class TestSignalHandlers(tests.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestSignalHandlers, self).setUp()
        orig = signals._setup_on_hangup_dict()
        self.assertIs(None, orig)

        def cleanup():
            if False:
                i = 10
                return i + 15
            signals._on_sighup = None
        self.addCleanup(cleanup)

    def test_registered_callback_gets_called(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def call_me():
            if False:
                return 10
            calls.append('called')
        signals.register_on_hangup('myid', call_me)
        signals._sighup_handler(SIGHUP, None)
        self.assertEqual(['called'], calls)
        signals.unregister_on_hangup('myid')

    def test_unregister_not_present(self):
        if False:
            print('Hello World!')
        signals.unregister_on_hangup('no-such-id')
        log = self.get_log()
        self.assertContainsRe(log, 'Error occurred during unregister_on_hangup:')
        self.assertContainsRe(log, '(?s)Traceback.*KeyError')

    def test_failing_callback(self):
        if False:
            print('Hello World!')
        calls = []

        def call_me():
            if False:
                print('Hello World!')
            calls.append('called')

        def fail_me():
            if False:
                return 10
            raise RuntimeError('something bad happened')
        signals.register_on_hangup('myid', call_me)
        signals.register_on_hangup('otherid', fail_me)
        signals._sighup_handler(SIGHUP, None)
        signals.unregister_on_hangup('myid')
        signals.unregister_on_hangup('otherid')
        log = self.get_log()
        self.assertContainsRe(log, '(?s)Traceback.*RuntimeError')
        self.assertEqual(['called'], calls)

    def test_unregister_during_call(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def call_me_and_unregister():
            if False:
                return 10
            signals.unregister_on_hangup('myid')
            calls.append('called_and_unregistered')

        def call_me():
            if False:
                print('Hello World!')
            calls.append('called')
        signals.register_on_hangup('myid', call_me_and_unregister)
        signals.register_on_hangup('other', call_me)
        signals._sighup_handler(SIGHUP, None)

    def test_keyboard_interrupt_propagated(self):
        if False:
            while True:
                i = 10

        def call_me_and_raise():
            if False:
                return 10
            raise KeyboardInterrupt()
        signals.register_on_hangup('myid', call_me_and_raise)
        self.assertRaises(KeyboardInterrupt, signals._sighup_handler, SIGHUP, None)
        signals.unregister_on_hangup('myid')

    def test_weak_references(self):
        if False:
            print('Hello World!')
        self.assertIsInstance(signals._on_sighup, weakref.WeakValueDictionary)
        calls = []

        def call_me():
            if False:
                for i in range(10):
                    print('nop')
            calls.append('called')
        signals.register_on_hangup('myid', call_me)
        del call_me
        signals._sighup_handler(SIGHUP, None)
        self.assertEqual([], calls)

    def test_not_installed(self):
        if False:
            return 10
        signals._on_sighup = None
        calls = []

        def call_me():
            if False:
                print('Hello World!')
            calls.append('called')
        signals.register_on_hangup('myid', calls)
        signals._sighup_handler(SIGHUP, None)
        signals.unregister_on_hangup('myid')
        log = self.get_log()
        self.assertEqual('', log)

    def test_install_sighup_handler(self):
        if False:
            while True:
                i = 10
        signals._on_sighup = None
        orig = signals.install_sighup_handler()
        if getattr(signal, 'SIGHUP', None) is not None:
            cur = signal.getsignal(SIGHUP)
            self.assertEqual(signals._sighup_handler, cur)
        self.assertIsNot(None, signals._on_sighup)
        signals.restore_sighup_handler(orig)
        self.assertIs(None, signals._on_sighup)

class TestInetServer(tests.TestCase):

    def create_file_pipes(self):
        if False:
            i = 10
            return i + 15
        (r, w) = os.pipe()
        rf = os.fdopen(r, 'rb')
        wf = os.fdopen(w, 'wb')
        return (rf, wf)

    def test_inet_server_responds_to_sighup(self):
        if False:
            for i in range(10):
                print('nop')
        t = transport.get_transport('memory:///')
        content = 'a' * 1024 * 1024
        t.put_bytes('bigfile', content)
        factory = server.BzrServerFactory()
        (client_read, server_write) = self.create_file_pipes()
        (server_read, client_write) = self.create_file_pipes()
        factory._get_stdin_stdout = lambda : (server_read, server_write)
        factory.set_up(t, None, None, inet=True, timeout=4.0)
        self.addCleanup(factory.tear_down)
        started = threading.Event()
        stopped = threading.Event()

        def serving():
            if False:
                while True:
                    i = 10
            started.set()
            factory.smart_server.serve()
            stopped.set()
        server_thread = threading.Thread(target=serving)
        server_thread.start()
        started.wait()
        client_medium = medium.SmartSimplePipesClientMedium(client_read, client_write, 'base')
        client_client = client._SmartClient(client_medium)
        (resp, response_handler) = client_client.call_expecting_body('get', 'bigfile')
        signals._sighup_handler(SIGHUP, None)
        self.assertTrue(factory.smart_server.finished)
        v = response_handler.read_body_bytes()
        if v != content:
            self.fail('Got the wrong content back, expected 1M "a"')
        stopped.wait()
        server_thread.join()