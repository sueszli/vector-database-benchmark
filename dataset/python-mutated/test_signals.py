import asyncio
import signal
import subprocess
import sys
import time
from uvloop import _testbase as tb
DELAY = 0.1

class _TestSignal:
    NEW_LOOP = None

    @tb.silence_long_exec_warning()
    def test_signals_sigint_pycode_stop(self):
        if False:
            print('Hello World!')

        async def runner():
            PROG = "\\\nimport asyncio\nimport uvloop\nimport time\n\nfrom uvloop import _testbase as tb\n\nasync def worker():\n    print('READY', flush=True)\n    time.sleep(200)\n\n@tb.silence_long_exec_warning()\ndef run():\n    loop = " + self.NEW_LOOP + '\n    asyncio.set_event_loop(loop)\n    try:\n        loop.run_until_complete(worker())\n    finally:\n        loop.close()\n\nrun()\n'
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.stdout.readline()
            time.sleep(DELAY)
            proc.send_signal(signal.SIGINT)
            (out, err) = await proc.communicate()
            self.assertIn(b'KeyboardInterrupt', err)
            self.assertEqual(out, b'')
        self.loop.run_until_complete(runner())

    @tb.silence_long_exec_warning()
    def test_signals_sigint_pycode_continue(self):
        if False:
            for i in range(10):
                print('nop')

        async def runner():
            PROG = '\\\nimport asyncio\nimport uvloop\nimport time\n\nfrom uvloop import _testbase as tb\n\nasync def worker():\n    print(\'READY\', flush=True)\n    try:\n        time.sleep(200)\n    except KeyboardInterrupt:\n        print("oups")\n    await asyncio.sleep(0.5)\n    print(\'done\')\n\n@tb.silence_long_exec_warning()\ndef run():\n    loop = ' + self.NEW_LOOP + '\n    asyncio.set_event_loop(loop)\n    try:\n        loop.run_until_complete(worker())\n    finally:\n        loop.close()\n\nrun()\n'
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.stdout.readline()
            time.sleep(DELAY)
            proc.send_signal(signal.SIGINT)
            (out, err) = await proc.communicate()
            self.assertEqual(err, b'')
            self.assertEqual(out, b'oups\ndone\n')
        self.loop.run_until_complete(runner())

    @tb.silence_long_exec_warning()
    def test_signals_sigint_uvcode(self):
        if False:
            i = 10
            return i + 15

        async def runner():
            PROG = "\\\nimport asyncio\nimport uvloop\n\nsrv = None\n\nasync def worker():\n    global srv\n    cb = lambda *args: None\n    srv = await asyncio.start_server(cb, '127.0.0.1', 0)\n    print('READY', flush=True)\n\nloop = " + self.NEW_LOOP + '\nasyncio.set_event_loop(loop)\nloop.create_task(worker())\ntry:\n    loop.run_forever()\nfinally:\n    srv.close()\n    loop.run_until_complete(srv.wait_closed())\n    loop.close()\n'
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.stdout.readline()
            time.sleep(DELAY)
            proc.send_signal(signal.SIGINT)
            (out, err) = await proc.communicate()
            self.assertIn(b'KeyboardInterrupt', err)
        self.loop.run_until_complete(runner())

    @tb.silence_long_exec_warning()
    def test_signals_sigint_uvcode_two_loop_runs(self):
        if False:
            for i in range(10):
                print('nop')

        async def runner():
            PROG = "\\\nimport asyncio\nimport uvloop\n\nsrv = None\n\nasync def worker():\n    global srv\n    cb = lambda *args: None\n    srv = await asyncio.start_server(cb, '127.0.0.1', 0)\n\nloop = " + self.NEW_LOOP + "\nasyncio.set_event_loop(loop)\nloop.run_until_complete(worker())\nprint('READY', flush=True)\ntry:\n    loop.run_forever()\nfinally:\n    srv.close()\n    loop.run_until_complete(srv.wait_closed())\n    loop.close()\n"
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.stdout.readline()
            time.sleep(DELAY)
            proc.send_signal(signal.SIGINT)
            (out, err) = await proc.communicate()
            self.assertIn(b'KeyboardInterrupt', err)
        self.loop.run_until_complete(runner())

    @tb.silence_long_exec_warning()
    def test_signals_sigint_and_custom_handler(self):
        if False:
            while True:
                i = 10

        async def runner():
            PROG = "\\\nimport asyncio\nimport signal\nimport uvloop\n\nsrv = None\n\nasync def worker():\n    global srv\n    cb = lambda *args: None\n    srv = await asyncio.start_server(cb, '127.0.0.1', 0)\n    print('READY', flush=True)\n\ndef handler_sig(say):\n    print(say, flush=True)\n    exit()\n\ndef handler_hup(say):\n    print(say, flush=True)\n\nloop = " + self.NEW_LOOP + "\nloop.add_signal_handler(signal.SIGINT, handler_sig, '!s-int!')\nloop.add_signal_handler(signal.SIGHUP, handler_hup, '!s-hup!')\nasyncio.set_event_loop(loop)\nloop.create_task(worker())\ntry:\n    loop.run_forever()\nfinally:\n    srv.close()\n    loop.run_until_complete(srv.wait_closed())\n    loop.close()\n"
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.stdout.readline()
            time.sleep(DELAY)
            proc.send_signal(signal.SIGHUP)
            time.sleep(DELAY)
            proc.send_signal(signal.SIGINT)
            (out, err) = await proc.communicate()
            self.assertEqual(err, b'')
            self.assertIn(b'!s-hup!', out)
            self.assertIn(b'!s-int!', out)
        self.loop.run_until_complete(runner())

    @tb.silence_long_exec_warning()
    def test_signals_and_custom_handler_1(self):
        if False:
            for i in range(10):
                print('nop')

        async def runner():
            PROG = '\\\nimport asyncio\nimport signal\nimport uvloop\n\nsrv = None\n\nasync def worker():\n    global srv\n    cb = lambda *args: None\n    srv = await asyncio.start_server(cb, \'127.0.0.1\', 0)\n    print(\'READY\', flush=True)\n\ndef handler1():\n    print("GOTIT", flush=True)\n\ndef handler2():\n    assert loop.remove_signal_handler(signal.SIGUSR1)\n    print("REMOVED", flush=True)\n\ndef handler_hup():\n    exit()\n\nloop = ' + self.NEW_LOOP + '\nasyncio.set_event_loop(loop)\nloop.add_signal_handler(signal.SIGUSR1, handler1)\nloop.add_signal_handler(signal.SIGUSR2, handler2)\nloop.add_signal_handler(signal.SIGHUP, handler_hup)\nloop.create_task(worker())\ntry:\n    loop.run_forever()\nfinally:\n    srv.close()\n    loop.run_until_complete(srv.wait_closed())\n    loop.close()\n\n'
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await proc.stdout.readline()
            time.sleep(DELAY)
            proc.send_signal(signal.SIGUSR1)
            time.sleep(DELAY)
            proc.send_signal(signal.SIGUSR1)
            time.sleep(DELAY)
            proc.send_signal(signal.SIGUSR2)
            time.sleep(DELAY)
            proc.send_signal(signal.SIGUSR1)
            time.sleep(DELAY)
            proc.send_signal(signal.SIGUSR1)
            time.sleep(DELAY)
            proc.send_signal(signal.SIGHUP)
            (out, err) = await proc.communicate()
            self.assertEqual(err, b'')
            self.assertEqual(b'GOTIT\nGOTIT\nREMOVED\n', out)
        self.loop.run_until_complete(runner())

    def test_signals_invalid_signal(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'sig {} cannot be caught'.format(signal.SIGKILL)):
            self.loop.add_signal_handler(signal.SIGKILL, lambda *a: None)

    def test_signals_coro_callback(self):
        if False:
            for i in range(10):
                print('nop')

        async def coro():
            pass
        with self.assertRaisesRegex(TypeError, 'coroutines cannot be used'):
            self.loop.add_signal_handler(signal.SIGHUP, coro)

    def test_signals_wakeup_fd_unchanged(self):
        if False:
            for i in range(10):
                print('nop')

        async def runner():
            PROG = '\\\nimport uvloop\nimport signal\nimport asyncio\n\n\ndef get_wakeup_fd():\n    fd = signal.set_wakeup_fd(-1)\n    signal.set_wakeup_fd(fd)\n    return fd\n\nasync def f(): pass\n\nfd0 = get_wakeup_fd()\nloop = ' + self.NEW_LOOP + '\ntry:\n    asyncio.set_event_loop(loop)\n    loop.run_until_complete(f())\n    fd1 = get_wakeup_fd()\nfinally:\n    loop.close()\n\nprint(fd0 == fd1, flush=True)\n\n'
            proc = await asyncio.create_subprocess_exec(sys.executable, b'-W', b'ignore', b'-c', PROG, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = await proc.communicate()
            self.assertEqual(err, b'')
            self.assertIn(b'True', out)
        self.loop.run_until_complete(runner())

    def test_signals_fork_in_thread(self):
        if False:
            i = 10
            return i + 15
        PROG = "\\\nimport asyncio\nimport multiprocessing\nimport signal\nimport sys\nimport threading\nimport uvloop\n\nmultiprocessing.set_start_method('fork')\n\ndef subprocess():\n    loop = " + self.NEW_LOOP + '\n    loop.add_signal_handler(signal.SIGINT, lambda *a: None)\n\ndef run():\n    loop = ' + self.NEW_LOOP + '\n    loop.add_signal_handler(signal.SIGINT, lambda *a: None)\n    p = multiprocessing.Process(target=subprocess)\n    t = threading.Thread(target=p.start)\n    t.start()\n    t.join()\n    p.join()\n    sys.exit(p.exitcode)\n\nrun()\n'
        subprocess.check_call([sys.executable, b'-W', b'ignore', b'-c', PROG])

class Test_UV_Signals(_TestSignal, tb.UVTestCase):
    NEW_LOOP = 'uvloop.new_event_loop()'

    def test_signals_no_SIGCHLD(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'cannot add.*handler.*SIGCHLD'):
            self.loop.add_signal_handler(signal.SIGCHLD, lambda *a: None)

class Test_AIO_Signals(_TestSignal, tb.AIOTestCase):
    NEW_LOOP = 'asyncio.new_event_loop()'