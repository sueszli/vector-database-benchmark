import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger

class BaseSubprocessTransport(transports.SubprocessTransport):

    def __init__(self, loop, protocol, args, shell, stdin, stdout, stderr, bufsize, waiter=None, extra=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(extra)
        self._closed = False
        self._protocol = protocol
        self._loop = loop
        self._proc = None
        self._pid = None
        self._returncode = None
        self._exit_waiters = []
        self._pending_calls = collections.deque()
        self._pipes = {}
        self._finished = False
        if stdin == subprocess.PIPE:
            self._pipes[0] = None
        if stdout == subprocess.PIPE:
            self._pipes[1] = None
        if stderr == subprocess.PIPE:
            self._pipes[2] = None
        try:
            self._start(args=args, shell=shell, stdin=stdin, stdout=stdout, stderr=stderr, bufsize=bufsize, **kwargs)
        except:
            self.close()
            raise
        self._pid = self._proc.pid
        self._extra['subprocess'] = self._proc
        if self._loop.get_debug():
            if isinstance(args, (bytes, str)):
                program = args
            else:
                program = args[0]
            logger.debug('process %r created: pid %s', program, self._pid)
        self._loop.create_task(self._connect_pipes(waiter))

    def __repr__(self):
        if False:
            print('Hello World!')
        info = [self.__class__.__name__]
        if self._closed:
            info.append('closed')
        if self._pid is not None:
            info.append(f'pid={self._pid}')
        if self._returncode is not None:
            info.append(f'returncode={self._returncode}')
        elif self._pid is not None:
            info.append('running')
        else:
            info.append('not started')
        stdin = self._pipes.get(0)
        if stdin is not None:
            info.append(f'stdin={stdin.pipe}')
        stdout = self._pipes.get(1)
        stderr = self._pipes.get(2)
        if stdout is not None and stderr is stdout:
            info.append(f'stdout=stderr={stdout.pipe}')
        else:
            if stdout is not None:
                info.append(f'stdout={stdout.pipe}')
            if stderr is not None:
                info.append(f'stderr={stderr.pipe}')
        return '<{}>'.format(' '.join(info))

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def set_protocol(self, protocol):
        if False:
            print('Hello World!')
        self._protocol = protocol

    def get_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        return self._protocol

    def is_closing(self):
        if False:
            for i in range(10):
                print('nop')
        return self._closed

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self._closed:
            return
        self._closed = True
        for proto in self._pipes.values():
            if proto is None:
                continue
            proto.pipe.close()
        if self._proc is not None and self._returncode is None and (self._proc.poll() is None):
            if self._loop.get_debug():
                logger.warning('Close running child process: kill %r', self)
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    def __del__(self, _warn=warnings.warn):
        if False:
            return 10
        if not self._closed:
            _warn(f'unclosed transport {self!r}', ResourceWarning, source=self)
            self.close()

    def get_pid(self):
        if False:
            i = 10
            return i + 15
        return self._pid

    def get_returncode(self):
        if False:
            return 10
        return self._returncode

    def get_pipe_transport(self, fd):
        if False:
            i = 10
            return i + 15
        if fd in self._pipes:
            return self._pipes[fd].pipe
        else:
            return None

    def _check_proc(self):
        if False:
            return 10
        if self._proc is None:
            raise ProcessLookupError()

    def send_signal(self, signal):
        if False:
            while True:
                i = 10
        self._check_proc()
        self._proc.send_signal(signal)

    def terminate(self):
        if False:
            print('Hello World!')
        self._check_proc()
        self._proc.terminate()

    def kill(self):
        if False:
            while True:
                i = 10
        self._check_proc()
        self._proc.kill()

    async def _connect_pipes(self, waiter):
        try:
            proc = self._proc
            loop = self._loop
            if proc.stdin is not None:
                (_, pipe) = await loop.connect_write_pipe(lambda : WriteSubprocessPipeProto(self, 0), proc.stdin)
                self._pipes[0] = pipe
            if proc.stdout is not None:
                (_, pipe) = await loop.connect_read_pipe(lambda : ReadSubprocessPipeProto(self, 1), proc.stdout)
                self._pipes[1] = pipe
            if proc.stderr is not None:
                (_, pipe) = await loop.connect_read_pipe(lambda : ReadSubprocessPipeProto(self, 2), proc.stderr)
                self._pipes[2] = pipe
            assert self._pending_calls is not None
            loop.call_soon(self._protocol.connection_made, self)
            for (callback, data) in self._pending_calls:
                loop.call_soon(callback, *data)
            self._pending_calls = None
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if waiter is not None and (not waiter.cancelled()):
                waiter.set_exception(exc)
        else:
            if waiter is not None and (not waiter.cancelled()):
                waiter.set_result(None)

    def _call(self, cb, *data):
        if False:
            i = 10
            return i + 15
        if self._pending_calls is not None:
            self._pending_calls.append((cb, data))
        else:
            self._loop.call_soon(cb, *data)

    def _pipe_connection_lost(self, fd, exc):
        if False:
            for i in range(10):
                print('nop')
        self._call(self._protocol.pipe_connection_lost, fd, exc)
        self._try_finish()

    def _pipe_data_received(self, fd, data):
        if False:
            while True:
                i = 10
        self._call(self._protocol.pipe_data_received, fd, data)

    def _process_exited(self, returncode):
        if False:
            for i in range(10):
                print('nop')
        assert returncode is not None, returncode
        assert self._returncode is None, self._returncode
        if self._loop.get_debug():
            logger.info('%r exited with return code %r', self, returncode)
        self._returncode = returncode
        if self._proc.returncode is None:
            self._proc.returncode = returncode
        self._call(self._protocol.process_exited)
        self._try_finish()
        for waiter in self._exit_waiters:
            if not waiter.cancelled():
                waiter.set_result(returncode)
        self._exit_waiters = None

    async def _wait(self):
        """Wait until the process exit and return the process return code.

        This method is a coroutine."""
        if self._returncode is not None:
            return self._returncode
        waiter = self._loop.create_future()
        self._exit_waiters.append(waiter)
        return await waiter

    def _try_finish(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self._finished
        if self._returncode is None:
            return
        if all((p is not None and p.disconnected for p in self._pipes.values())):
            self._finished = True
            self._call(self._call_connection_lost, None)

    def _call_connection_lost(self, exc):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._protocol.connection_lost(exc)
        finally:
            self._loop = None
            self._proc = None
            self._protocol = None

class WriteSubprocessPipeProto(protocols.BaseProtocol):

    def __init__(self, proc, fd):
        if False:
            i = 10
            return i + 15
        self.proc = proc
        self.fd = fd
        self.pipe = None
        self.disconnected = False

    def connection_made(self, transport):
        if False:
            return 10
        self.pipe = transport

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<{self.__class__.__name__} fd={self.fd} pipe={self.pipe!r}>'

    def connection_lost(self, exc):
        if False:
            while True:
                i = 10
        self.disconnected = True
        self.proc._pipe_connection_lost(self.fd, exc)
        self.proc = None

    def pause_writing(self):
        if False:
            print('Hello World!')
        self.proc._protocol.pause_writing()

    def resume_writing(self):
        if False:
            while True:
                i = 10
        self.proc._protocol.resume_writing()

class ReadSubprocessPipeProto(WriteSubprocessPipeProto, protocols.Protocol):

    def data_received(self, data):
        if False:
            print('Hello World!')
        self.proc._pipe_data_received(self.fd, data)