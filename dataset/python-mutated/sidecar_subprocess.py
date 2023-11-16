from __future__ import print_function
import subprocess
import fcntl
import select
import os
import sys
import platform
from fcntl import F_SETFL
from os import O_NONBLOCK
from .sidecar_messages import Message, MessageTypes
from ..debug import debug
from metaflow.tracing import inject_tracing_vars
MUST_SEND_RETRY_TIMES = 4
MESSAGE_WRITE_TIMEOUT_IN_MS = 1000
NULL_SIDECAR_PREFIX = 'nullSidecar'
try:
    blockingError = BlockingIOError
except:
    blockingError = OSError

class PipeUnavailableError(Exception):
    """raised when unable to write to pipe given allotted time"""
    pass

class NullSidecarError(Exception):
    """raised when trying to poll or interact with the fake subprocess in the null sidecar"""
    pass

class MsgTimeoutError(Exception):
    """raised when trying unable to send message to sidecar in allocated time"""
    pass

class NullPoller(object):

    def poll(self, timeout):
        if False:
            i = 10
            return i + 15
        raise NullSidecarError()

class SidecarSubProcess(object):

    def __init__(self, worker_type):
        if False:
            return 10
        self._worker_type = worker_type
        self._process = None
        self._poller = None
        self._send_mustsend_remaining_tries = 0
        self._cached_mustsend = None
        self._prev_message_error = False
        self.start()

    def start(self):
        if False:
            return 10
        if self._worker_type is not None and self._worker_type.startswith(NULL_SIDECAR_PREFIX) or (platform.system() == 'Darwin' and sys.version_info < (3, 0)):
            self._poller = NullPoller()
            self._process = None
            self._logger('No sidecar started')
        else:
            self._starting = True
            from select import poll
            python_version = sys.executable
            cmdline = [python_version, '-u', os.path.dirname(__file__) + '/sidecar_worker.py', self._worker_type]
            self._logger('Starting sidecar')
            debug.sidecar_exec(cmdline)
            self._process = self._start_subprocess(cmdline)
            if self._process is not None:
                fcntl.fcntl(self._process.stdin, F_SETFL, O_NONBLOCK)
                self._poller = poll()
                self._poller.register(self._process.stdin.fileno(), select.POLLOUT)
            else:
                self._logger('Unable to start subprocess')
                self._poller = NullPoller()

    def kill(self):
        if False:
            i = 10
            return i + 15
        try:
            msg = Message(MessageTypes.SHUTDOWN, None)
            self._emit_msg(msg)
        except:
            pass

    def send(self, msg, retries=3):
        if False:
            while True:
                i = 10
        if msg.msg_type == MessageTypes.MUST_SEND:
            self._cached_mustsend = msg.payload
            self._send_mustsend_remaining_tries = MUST_SEND_RETRY_TIMES
            self._send_mustsend(retries)
        else:
            self._send_internal(msg, retries=retries)

    def _start_subprocess(self, cmdline):
        if False:
            print('Hello World!')
        for _ in range(3):
            try:
                env = os.environ.copy()
                inject_tracing_vars(env)
                return subprocess.Popen(cmdline, stdin=subprocess.PIPE, env=env, stdout=sys.stdout if debug.sidecar else subprocess.DEVNULL, stderr=sys.stderr if debug.sidecar else subprocess.DEVNULL, bufsize=0)
            except blockingError as be:
                self._logger('Sidecar popen failed: %s' % repr(be))
            except Exception as e:
                self._logger('Unknown popen error: %s' % repr(e))
                break

    def _send_internal(self, msg, retries=3):
        if False:
            i = 10
            return i + 15
        if self._process is None:
            return False
        try:
            if msg.msg_type == MessageTypes.BEST_EFFORT:
                if self._send_mustsend_remaining_tries == -1:
                    raise PipeUnavailableError()
                elif self._send_mustsend_remaining_tries > 0:
                    self._send_mustsend()
                if self._send_mustsend_remaining_tries == 0:
                    self._emit_msg(msg)
                    self._prev_message_error = False
                    return True
            else:
                self._emit_msg(msg)
                self._prev_message_error = False
                return True
            return False
        except MsgTimeoutError:
            self._logger('Unable to send message due to timeout')
            self._prev_message_error = True
        except Exception as ex:
            if isinstance(ex, (PipeUnavailableError, BrokenPipeError)):
                self._logger('Restarting sidecar due to broken/unavailable pipe')
                self.start()
                if self._cached_mustsend is not None:
                    self._send_mustsend_remaining_tries = MUST_SEND_RETRY_TIMES
            else:
                self._prev_message_error = True
            if retries > 0:
                self._logger('Retrying msg send to sidecar (due to %s)' % repr(ex))
                return self._send_internal(msg, retries - 1)
            else:
                self._logger('Error sending log message (exhausted retries): %s' % repr(ex))
        return False

    def _send_mustsend(self, retries=3):
        if False:
            print('Hello World!')
        if self._cached_mustsend is not None and self._send_mustsend_remaining_tries > 0:
            if self._send_internal(Message(MessageTypes.MUST_SEND, self._cached_mustsend), retries):
                self._cached_mustsend = None
                self._send_mustsend_remaining_tries = 0
                return True
            else:
                self._send_mustsend_remaining_tries -= 1
                if self._send_mustsend_remaining_tries == 0:
                    self._send_mustsend_remaining_tries = -1
                return False

    def _emit_msg(self, msg):
        if False:
            while True:
                i = 10
        msg = msg.serialize()
        if self._prev_message_error:
            msg = '\n' + msg
        msg_ser = msg.encode('utf-8')
        written_bytes = 0
        while written_bytes < len(msg_ser):
            try:
                fds = self._poller.poll(MESSAGE_WRITE_TIMEOUT_IN_MS)
                if fds is None or len(fds) == 0:
                    raise MsgTimeoutError('Poller timed out')
                for (fd, event) in fds:
                    if event & select.POLLERR:
                        raise PipeUnavailableError('Pipe unavailable')
                    f = os.write(fd, msg_ser[written_bytes:])
                    written_bytes += f
            except NullSidecarError:
                break

    def _logger(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if debug.sidecar:
            print('[sidecar:%s] %s' % (self._worker_type, msg), file=sys.stderr)