import logging
import multiprocessing
import signal
import subprocess
import sys
import time
import pytest
from pytestshellutils.utils.processes import terminate_process
from salt._logging.handlers import DeferredStreamHandler
from salt.utils.nb_popen import NonBlockingPopen
from tests.support.helpers import CaptureOutput, dedent
from tests.support.runtests import RUNTIME_VARS
log = logging.getLogger(__name__)

def _sync_with_handlers_proc_target():
    if False:
        for i in range(10):
            print('nop')
    with CaptureOutput() as stds:
        handler = DeferredStreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        logger = logging.getLogger(__name__)
        logger.info('Foo')
        logger.info('Bar')
        logging.root.removeHandler(handler)
        assert not stds.stdout
        assert not stds.stderr
        stream_handler = logging.StreamHandler(sys.stderr)
        handler.sync_with_handlers([stream_handler])
        assert not stds.stdout
        assert stds.stderr == 'Foo\nBar\n'

def _deferred_write_on_flush_proc_target():
    if False:
        while True:
            i = 10
    with CaptureOutput() as stds:
        handler = DeferredStreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        logger = logging.getLogger(__name__)
        logger.info('Foo')
        logger.info('Bar')
        logging.root.removeHandler(handler)
        assert not stds.stdout
        assert not stds.stderr
        handler.flush()
        assert not stds.stdout
        assert stds.stderr == 'Foo\nBar\n'

def test_sync_with_handlers():
    if False:
        while True:
            i = 10
    proc = multiprocessing.Process(target=_sync_with_handlers_proc_target)
    proc.start()
    proc.join()
    assert proc.exitcode == 0

def test_deferred_write_on_flush():
    if False:
        for i in range(10):
            print('nop')
    proc = multiprocessing.Process(target=_deferred_write_on_flush_proc_target)
    proc.start()
    proc.join()
    assert proc.exitcode == 0

def test_deferred_write_on_atexit(tmp_path):
    if False:
        while True:
            i = 10
    pyscript = dedent("\n        import sys\n        import time\n        import logging\n\n        CODE_DIR = {!r}\n        if CODE_DIR in sys.path:\n            sys.path.remove(CODE_DIR)\n        sys.path.insert(0, CODE_DIR)\n\n        from salt._logging.handlers import DeferredStreamHandler\n        # Reset any logging handlers we might have already\n        logging.root.handlers[:] = []\n\n        handler = DeferredStreamHandler(sys.stderr)\n        handler.setLevel(logging.DEBUG)\n        logging.root.addHandler(handler)\n\n        log = logging.getLogger(__name__)\n        sys.stdout.write('STARTED\\n')\n        sys.stdout.flush()\n        log.debug('Foo')\n        sys.exit(0)\n    ".format(RUNTIME_VARS.CODE_DIR))
    script_path = tmp_path / 'atexit_deferred_logging_test.py'
    script_path.write_text(pyscript, encoding='utf-8')
    proc = NonBlockingPopen([sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = b''
    err = b''
    execution_time = 5
    max_time = time.time() + execution_time
    try:
        while True:
            if time.time() > max_time:
                pytest.fail("Script didn't exit after {} second".format(execution_time))
            time.sleep(0.125)
            _out = proc.recv()
            _err = proc.recv_err()
            if _out:
                out += _out
            if _err:
                err += _err
            if _out is None and _err is None:
                break
            if proc.poll() is not None:
                break
    finally:
        terminate_process(proc.pid, kill_children=True)
    if b'Foo' not in err:
        pytest.fail("'Foo' should be in stderr and it's not: {}".format(err))

@pytest.mark.skip_on_windows(reason='Windows does not support SIGINT')
def test_deferred_write_on_sigint(tmp_path):
    if False:
        print('Hello World!')
    pyscript = dedent("\n        import sys\n        import time\n        import signal\n        import logging\n\n        CODE_DIR = {!r}\n        if CODE_DIR in sys.path:\n            sys.path.remove(CODE_DIR)\n        sys.path.insert(0, CODE_DIR)\n\n        from salt._logging.handlers import DeferredStreamHandler\n        # Reset any logging handlers we might have already\n        logging.root.handlers[:] = []\n\n        handler = DeferredStreamHandler(sys.stderr)\n        handler.setLevel(logging.DEBUG)\n        logging.root.addHandler(handler)\n\n        if signal.getsignal(signal.SIGINT) != signal.default_int_handler:\n            # Looking at you Debian based distros :/\n            signal.signal(signal.SIGINT, signal.default_int_handler)\n\n        log = logging.getLogger(__name__)\n\n        start_printed = False\n        while True:\n            try:\n                log.debug('Foo')\n                if start_printed is False:\n                    sys.stdout.write('STARTED\\n')\n                    sys.stdout.write('SIGINT HANDLER: {{!r}}\\n'.format(signal.getsignal(signal.SIGINT)))\n                    sys.stdout.flush()\n                    start_printed = True\n                time.sleep(0.125)\n            except (KeyboardInterrupt, SystemExit):\n                log.info('KeyboardInterrupt caught')\n                sys.stdout.write('KeyboardInterrupt caught\\n')\n                sys.stdout.flush()\n                break\n        log.info('EXITING')\n        sys.stdout.write('EXITING\\n')\n        sys.stdout.flush()\n        sys.exit(0)\n        ".format(RUNTIME_VARS.CODE_DIR))
    script_path = tmp_path / 'sigint_deferred_logging_test.py'
    script_path.write_text(pyscript, encoding='utf-8')
    proc = NonBlockingPopen([sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = b''
    err = b''
    execution_time = 10
    start = time.time()
    max_time = time.time() + execution_time
    try:
        signalled = False
        log.info('Starting Loop')
        while True:
            time.sleep(0.125)
            _out = proc.recv()
            _err = proc.recv_err()
            if _out:
                out += _out
            if _err:
                err += _err
            if b'STARTED' in out and (not signalled):
                proc.send_signal(signal.SIGINT)
                signalled = True
                log.debug('Sent SIGINT after: %s', time.time() - start)
            if signalled is False:
                if out:
                    pytest.fail('We have stdout output when there should be none: {}'.format(out))
                if err:
                    pytest.fail('We have stderr output when there should be none: {}'.format(err))
            if _out is None and _err is None:
                log.info('_out and _err are None')
                if b'Foo' not in err:
                    pytest.fail("No more output and 'Foo' should be in stderr and it's not: {}".format(err))
                break
            if proc.poll() is not None:
                log.debug('poll() is not None')
                if b'Foo' not in err:
                    pytest.fail("Process terminated and 'Foo' should be in stderr and it's not: {}".format(err))
                break
            if time.time() > max_time:
                log.debug('Reached max time')
                if b'Foo' not in err:
                    pytest.fail("'Foo' should be in stderr and it's not:\n{0}\nSTDERR:\n{0}\n{1}\n{0}\nSTDOUT:\n{0}\n{2}\n{0}".format('-' * 80, err, out))
    finally:
        terminate_process(proc.pid, kill_children=True)
    log.debug('Test took %s seconds', time.time() - start)