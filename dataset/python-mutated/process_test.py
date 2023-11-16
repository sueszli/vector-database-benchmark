import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from tornado.httpclient import HTTPClient, HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.log import gen_log
from tornado.process import fork_processes, task_id, Subprocess
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import bind_unused_port, ExpectLog, AsyncTestCase, gen_test
from tornado.test.util import skipIfNonUnix
from tornado.web import RequestHandler, Application

@skipIfNonUnix
class ProcessTest(unittest.TestCase):

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')

        class ProcessHandler(RequestHandler):

            def get(self):
                if False:
                    i = 10
                    return i + 15
                if self.get_argument('exit', None):
                    os._exit(int(self.get_argument('exit')))
                if self.get_argument('signal', None):
                    os.kill(os.getpid(), int(self.get_argument('signal')))
                self.write(str(os.getpid()))
        return Application([('/', ProcessHandler)])

    def tearDown(self):
        if False:
            return 10
        if task_id() is not None:
            logging.error('aborting child process from tearDown')
            logging.shutdown()
            os._exit(1)
        signal.alarm(0)
        super().tearDown()

    def test_multi_process(self):
        if False:
            i = 10
            return i + 15
        with ExpectLog(gen_log, '(Starting .* processes|child .* exited|uncaught exception)'):
            (sock, port) = bind_unused_port()

            def get_url(path):
                if False:
                    while True:
                        i = 10
                return 'http://127.0.0.1:%d%s' % (port, path)
            signal.alarm(5)
            try:
                id = fork_processes(3, max_restarts=3)
                self.assertTrue(id is not None)
                signal.alarm(5)
            except SystemExit as e:
                self.assertEqual(e.code, 0)
                self.assertTrue(task_id() is None)
                sock.close()
                return
            try:
                if id in (0, 1):
                    self.assertEqual(id, task_id())

                    async def f():
                        server = HTTPServer(self.get_app())
                        server.add_sockets([sock])
                        await asyncio.Event().wait()
                    asyncio.run(f())
                elif id == 2:
                    self.assertEqual(id, task_id())
                    sock.close()
                    client = HTTPClient(SimpleAsyncHTTPClient)

                    def fetch(url, fail_ok=False):
                        if False:
                            print('Hello World!')
                        try:
                            return client.fetch(get_url(url))
                        except HTTPError as e:
                            if not (fail_ok and e.code == 599):
                                raise
                    fetch('/?exit=2', fail_ok=True)
                    fetch('/?exit=3', fail_ok=True)
                    int(fetch('/').body)
                    fetch('/?exit=0', fail_ok=True)
                    pid = int(fetch('/').body)
                    fetch('/?exit=4', fail_ok=True)
                    pid2 = int(fetch('/').body)
                    self.assertNotEqual(pid, pid2)
                    fetch('/?exit=0', fail_ok=True)
                    os._exit(0)
            except Exception:
                logging.error('exception in child process %d', id, exc_info=True)
                raise

@skipIfNonUnix
class SubprocessTest(AsyncTestCase):

    def term_and_wait(self, subproc):
        if False:
            while True:
                i = 10
        subproc.proc.terminate()
        subproc.proc.wait()

    @gen_test
    def test_subprocess(self):
        if False:
            print('Hello World!')
        if IOLoop.configured_class().__name__.endswith('LayeredTwistedIOLoop'):
            raise unittest.SkipTest('Subprocess tests not compatible with LayeredTwistedIOLoop')
        subproc = Subprocess([sys.executable, '-u', '-i'], stdin=Subprocess.STREAM, stdout=Subprocess.STREAM, stderr=subprocess.STDOUT)
        self.addCleanup(lambda : self.term_and_wait(subproc))
        self.addCleanup(subproc.stdout.close)
        self.addCleanup(subproc.stdin.close)
        yield subproc.stdout.read_until(b'>>> ')
        subproc.stdin.write(b"print('hello')\n")
        data = (yield subproc.stdout.read_until(b'\n'))
        self.assertEqual(data, b'hello\n')
        yield subproc.stdout.read_until(b'>>> ')
        subproc.stdin.write(b'raise SystemExit\n')
        data = (yield subproc.stdout.read_until_close())
        self.assertEqual(data, b'')

    @gen_test
    def test_close_stdin(self):
        if False:
            i = 10
            return i + 15
        subproc = Subprocess([sys.executable, '-u', '-i'], stdin=Subprocess.STREAM, stdout=Subprocess.STREAM, stderr=subprocess.STDOUT)
        self.addCleanup(lambda : self.term_and_wait(subproc))
        yield subproc.stdout.read_until(b'>>> ')
        subproc.stdin.close()
        data = (yield subproc.stdout.read_until_close())
        self.assertEqual(data, b'\n')

    @gen_test
    def test_stderr(self):
        if False:
            print('Hello World!')
        subproc = Subprocess([sys.executable, '-u', '-c', "import sys; sys.stderr.write('hello\\n')"], stderr=Subprocess.STREAM)
        self.addCleanup(lambda : self.term_and_wait(subproc))
        data = (yield subproc.stderr.read_until(b'\n'))
        self.assertEqual(data, b'hello\n')
        subproc.stderr.close()

    def test_sigchild(self):
        if False:
            print('Hello World!')
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'pass'])
        subproc.set_exit_callback(self.stop)
        ret = self.wait()
        self.assertEqual(ret, 0)
        self.assertEqual(subproc.returncode, ret)

    @gen_test
    def test_sigchild_future(self):
        if False:
            for i in range(10):
                print('nop')
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'pass'])
        ret = (yield subproc.wait_for_exit())
        self.assertEqual(ret, 0)
        self.assertEqual(subproc.returncode, ret)

    def test_sigchild_signal(self):
        if False:
            for i in range(10):
                print('nop')
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=Subprocess.STREAM)
        self.addCleanup(subproc.stdout.close)
        subproc.set_exit_callback(self.stop)
        time.sleep(0.1)
        os.kill(subproc.pid, signal.SIGTERM)
        try:
            ret = self.wait()
        except AssertionError:
            fut = subproc.stdout.read_until_close()
            fut.add_done_callback(lambda f: self.stop())
            try:
                self.wait()
            except AssertionError:
                raise AssertionError('subprocess failed to terminate')
            else:
                raise AssertionError('subprocess closed stdout but failed to get termination signal')
        self.assertEqual(subproc.returncode, ret)
        self.assertEqual(ret, -signal.SIGTERM)

    @gen_test
    def test_wait_for_exit_raise(self):
        if False:
            for i in range(10):
                print('nop')
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'import sys; sys.exit(1)'])
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            yield subproc.wait_for_exit()
        self.assertEqual(cm.exception.returncode, 1)

    @gen_test
    def test_wait_for_exit_raise_disabled(self):
        if False:
            return 10
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'import sys; sys.exit(1)'])
        ret = (yield subproc.wait_for_exit(raise_error=False))
        self.assertEqual(ret, 1)