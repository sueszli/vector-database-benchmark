"""Per-test stdout/stderr capturing mechanism."""
import abc
import collections
import contextlib
import io
import os
import sys
from io import UnsupportedOperation
from tempfile import TemporaryFile
from types import TracebackType
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import SubRequest
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.reports import CollectReport
_CaptureMethod = Literal['fd', 'sys', 'no', 'tee-sys']

def pytest_addoption(parser: Parser) -> None:
    if False:
        while True:
            i = 10
    group = parser.getgroup('general')
    group._addoption('--capture', action='store', default='fd', metavar='method', choices=['fd', 'sys', 'no', 'tee-sys'], help='Per-test capturing method: one of fd|sys|no|tee-sys')
    group._addoption('-s', action='store_const', const='no', dest='capture', help='Shortcut for --capture=no')

def _colorama_workaround() -> None:
    if False:
        i = 10
        return i + 15
    'Ensure colorama is imported so that it attaches to the correct stdio\n    handles on Windows.\n\n    colorama uses the terminal on import time. So if something does the\n    first import of colorama while I/O capture is active, colorama will\n    fail in various ways.\n    '
    if sys.platform.startswith('win32'):
        try:
            import colorama
        except ImportError:
            pass

def _windowsconsoleio_workaround(stream: TextIO) -> None:
    if False:
        i = 10
        return i + 15
    'Workaround for Windows Unicode console handling.\n\n    Python 3.6 implemented Unicode console handling for Windows. This works\n    by reading/writing to the raw console handle using\n    ``{Read,Write}ConsoleW``.\n\n    The problem is that we are going to ``dup2`` over the stdio file\n    descriptors when doing ``FDCapture`` and this will ``CloseHandle`` the\n    handles used by Python to write to the console. Though there is still some\n    weirdness and the console handle seems to only be closed randomly and not\n    on the first call to ``CloseHandle``, or maybe it gets reopened with the\n    same handle value when we suspend capturing.\n\n    The workaround in this case will reopen stdio with a different fd which\n    also means a different handle by replicating the logic in\n    "Py_lifecycle.c:initstdio/create_stdio".\n\n    :param stream:\n        In practice ``sys.stdout`` or ``sys.stderr``, but given\n        here as parameter for unittesting purposes.\n\n    See https://github.com/pytest-dev/py/issues/103.\n    '
    if not sys.platform.startswith('win32') or hasattr(sys, 'pypy_version_info'):
        return
    if not hasattr(stream, 'buffer'):
        return
    buffered = hasattr(stream.buffer, 'raw')
    raw_stdout = stream.buffer.raw if buffered else stream.buffer
    if not isinstance(raw_stdout, io._WindowsConsoleIO):
        return

    def _reopen_stdio(f, mode):
        if False:
            i = 10
            return i + 15
        if not buffered and mode[0] == 'w':
            buffering = 0
        else:
            buffering = -1
        return io.TextIOWrapper(open(os.dup(f.fileno()), mode, buffering), f.encoding, f.errors, f.newlines, f.line_buffering)
    sys.stdin = _reopen_stdio(sys.stdin, 'rb')
    sys.stdout = _reopen_stdio(sys.stdout, 'wb')
    sys.stderr = _reopen_stdio(sys.stderr, 'wb')

@hookimpl(wrapper=True)
def pytest_load_initial_conftests(early_config: Config) -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    ns = early_config.known_args_namespace
    if ns.capture == 'fd':
        _windowsconsoleio_workaround(sys.stdout)
    _colorama_workaround()
    pluginmanager = early_config.pluginmanager
    capman = CaptureManager(ns.capture)
    pluginmanager.register(capman, 'capturemanager')
    early_config.add_cleanup(capman.stop_global_capturing)
    capman.start_global_capturing()
    try:
        try:
            yield
        finally:
            capman.suspend_global_capture()
    except BaseException:
        (out, err) = capman.read_global_capture()
        sys.stdout.write(out)
        sys.stderr.write(err)
        raise

class EncodedFile(io.TextIOWrapper):
    __slots__ = ()

    @property
    def name(self) -> str:
        if False:
            return 10
        return repr(self.buffer)

    @property
    def mode(self) -> str:
        if False:
            while True:
                i = 10
        return self.buffer.mode.replace('b', '')

class CaptureIO(io.TextIOWrapper):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(io.BytesIO(), encoding='UTF-8', newline='', write_through=True)

    def getvalue(self) -> str:
        if False:
            return 10
        assert isinstance(self.buffer, io.BytesIO)
        return self.buffer.getvalue().decode('UTF-8')

class TeeCaptureIO(CaptureIO):

    def __init__(self, other: TextIO) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._other = other
        super().__init__()

    def write(self, s: str) -> int:
        if False:
            i = 10
            return i + 15
        super().write(s)
        return self._other.write(s)

class DontReadFromInput(TextIO):

    @property
    def encoding(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return sys.__stdin__.encoding

    def read(self, size: int=-1) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise OSError('pytest: reading from stdin while output is captured!  Consider using `-s`.')
    readline = read

    def __next__(self) -> str:
        if False:
            return 10
        return self.readline()

    def readlines(self, hint: Optional[int]=-1) -> List[str]:
        if False:
            return 10
        raise OSError('pytest: reading from stdin while output is captured!  Consider using `-s`.')

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        return self

    def fileno(self) -> int:
        if False:
            return 10
        raise UnsupportedOperation('redirected stdin is pseudofile, has no fileno()')

    def flush(self) -> None:
        if False:
            i = 10
            return i + 15
        raise UnsupportedOperation('redirected stdin is pseudofile, has no flush()')

    def isatty(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def close(self) -> None:
        if False:
            return 10
        pass

    def readable(self) -> bool:
        if False:
            return 10
        return False

    def seek(self, offset: int, whence: int=0) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise UnsupportedOperation('redirected stdin is pseudofile, has no seek(int)')

    def seekable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def tell(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise UnsupportedOperation('redirected stdin is pseudofile, has no tell()')

    def truncate(self, size: Optional[int]=None) -> int:
        if False:
            while True:
                i = 10
        raise UnsupportedOperation('cannot truncate stdin')

    def write(self, data: str) -> int:
        if False:
            i = 10
            return i + 15
        raise UnsupportedOperation('cannot write to stdin')

    def writelines(self, lines: Iterable[str]) -> None:
        if False:
            i = 10
            return i + 15
        raise UnsupportedOperation('Cannot write to stdin')

    def writable(self) -> bool:
        if False:
            return 10
        return False

    def __enter__(self) -> 'DontReadFromInput':
        if False:
            return 10
        return self

    def __exit__(self, type: Optional[Type[BaseException]], value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @property
    def buffer(self) -> BinaryIO:
        if False:
            return 10
        return self

class CaptureBase(abc.ABC, Generic[AnyStr]):
    EMPTY_BUFFER: AnyStr

    @abc.abstractmethod
    def __init__(self, fd: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def suspend(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abc.abstractmethod
    def resume(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def writeorg(self, data: AnyStr) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abc.abstractmethod
    def snap(self) -> AnyStr:
        if False:
            print('Hello World!')
        raise NotImplementedError()
patchsysdict = {0: 'stdin', 1: 'stdout', 2: 'stderr'}

class NoCapture(CaptureBase[str]):
    EMPTY_BUFFER = ''

    def __init__(self, fd: int) -> None:
        if False:
            while True:
                i = 10
        pass

    def start(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def done(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def suspend(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def resume(self) -> None:
        if False:
            return 10
        pass

    def snap(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ''

    def writeorg(self, data: str) -> None:
        if False:
            i = 10
            return i + 15
        pass

class SysCaptureBase(CaptureBase[AnyStr]):

    def __init__(self, fd: int, tmpfile: Optional[TextIO]=None, *, tee: bool=False) -> None:
        if False:
            while True:
                i = 10
        name = patchsysdict[fd]
        self._old: TextIO = getattr(sys, name)
        self.name = name
        if tmpfile is None:
            if name == 'stdin':
                tmpfile = DontReadFromInput()
            else:
                tmpfile = CaptureIO() if not tee else TeeCaptureIO(self._old)
        self.tmpfile = tmpfile
        self._state = 'initialized'

    def repr(self, class_name: str) -> str:
        if False:
            while True:
                i = 10
        return '<{} {} _old={} _state={!r} tmpfile={!r}>'.format(class_name, self.name, hasattr(self, '_old') and repr(self._old) or '<UNSET>', self._state, self.tmpfile)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<{} {} _old={} _state={!r} tmpfile={!r}>'.format(self.__class__.__name__, self.name, hasattr(self, '_old') and repr(self._old) or '<UNSET>', self._state, self.tmpfile)

    def _assert_state(self, op: str, states: Tuple[str, ...]) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert self._state in states, 'cannot {} in state {!r}: expected one of {}'.format(op, self._state, ', '.join(states))

    def start(self) -> None:
        if False:
            return 10
        self._assert_state('start', ('initialized',))
        setattr(sys, self.name, self.tmpfile)
        self._state = 'started'

    def done(self) -> None:
        if False:
            return 10
        self._assert_state('done', ('initialized', 'started', 'suspended', 'done'))
        if self._state == 'done':
            return
        setattr(sys, self.name, self._old)
        del self._old
        self.tmpfile.close()
        self._state = 'done'

    def suspend(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._assert_state('suspend', ('started', 'suspended'))
        setattr(sys, self.name, self._old)
        self._state = 'suspended'

    def resume(self) -> None:
        if False:
            print('Hello World!')
        self._assert_state('resume', ('started', 'suspended'))
        if self._state == 'started':
            return
        setattr(sys, self.name, self.tmpfile)
        self._state = 'started'

class SysCaptureBinary(SysCaptureBase[bytes]):
    EMPTY_BUFFER = b''

    def snap(self) -> bytes:
        if False:
            i = 10
            return i + 15
        self._assert_state('snap', ('started', 'suspended'))
        self.tmpfile.seek(0)
        res = self.tmpfile.buffer.read()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: bytes) -> None:
        if False:
            return 10
        self._assert_state('writeorg', ('started', 'suspended'))
        self._old.flush()
        self._old.buffer.write(data)
        self._old.buffer.flush()

class SysCapture(SysCaptureBase[str]):
    EMPTY_BUFFER = ''

    def snap(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._assert_state('snap', ('started', 'suspended'))
        assert isinstance(self.tmpfile, CaptureIO)
        res = self.tmpfile.getvalue()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: str) -> None:
        if False:
            print('Hello World!')
        self._assert_state('writeorg', ('started', 'suspended'))
        self._old.write(data)
        self._old.flush()

class FDCaptureBase(CaptureBase[AnyStr]):

    def __init__(self, targetfd: int) -> None:
        if False:
            i = 10
            return i + 15
        self.targetfd = targetfd
        try:
            os.fstat(targetfd)
        except OSError:
            self.targetfd_invalid: Optional[int] = os.open(os.devnull, os.O_RDWR)
            os.dup2(self.targetfd_invalid, targetfd)
        else:
            self.targetfd_invalid = None
        self.targetfd_save = os.dup(targetfd)
        if targetfd == 0:
            self.tmpfile = open(os.devnull, encoding='utf-8')
            self.syscapture: CaptureBase[str] = SysCapture(targetfd)
        else:
            self.tmpfile = EncodedFile(TemporaryFile(buffering=0), encoding='utf-8', errors='replace', newline='', write_through=True)
            if targetfd in patchsysdict:
                self.syscapture = SysCapture(targetfd, self.tmpfile)
            else:
                self.syscapture = NoCapture(targetfd)
        self._state = 'initialized'

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<{} {} oldfd={} _state={!r} tmpfile={!r}>'.format(self.__class__.__name__, self.targetfd, self.targetfd_save, self._state, self.tmpfile)

    def _assert_state(self, op: str, states: Tuple[str, ...]) -> None:
        if False:
            i = 10
            return i + 15
        assert self._state in states, 'cannot {} in state {!r}: expected one of {}'.format(op, self._state, ', '.join(states))

    def start(self) -> None:
        if False:
            while True:
                i = 10
        'Start capturing on targetfd using memorized tmpfile.'
        self._assert_state('start', ('initialized',))
        os.dup2(self.tmpfile.fileno(), self.targetfd)
        self.syscapture.start()
        self._state = 'started'

    def done(self) -> None:
        if False:
            i = 10
            return i + 15
        'Stop capturing, restore streams, return original capture file,\n        seeked to position zero.'
        self._assert_state('done', ('initialized', 'started', 'suspended', 'done'))
        if self._state == 'done':
            return
        os.dup2(self.targetfd_save, self.targetfd)
        os.close(self.targetfd_save)
        if self.targetfd_invalid is not None:
            if self.targetfd_invalid != self.targetfd:
                os.close(self.targetfd)
            os.close(self.targetfd_invalid)
        self.syscapture.done()
        self.tmpfile.close()
        self._state = 'done'

    def suspend(self) -> None:
        if False:
            return 10
        self._assert_state('suspend', ('started', 'suspended'))
        if self._state == 'suspended':
            return
        self.syscapture.suspend()
        os.dup2(self.targetfd_save, self.targetfd)
        self._state = 'suspended'

    def resume(self) -> None:
        if False:
            print('Hello World!')
        self._assert_state('resume', ('started', 'suspended'))
        if self._state == 'started':
            return
        self.syscapture.resume()
        os.dup2(self.tmpfile.fileno(), self.targetfd)
        self._state = 'started'

class FDCaptureBinary(FDCaptureBase[bytes]):
    """Capture IO to/from a given OS-level file descriptor.

    snap() produces `bytes`.
    """
    EMPTY_BUFFER = b''

    def snap(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        self._assert_state('snap', ('started', 'suspended'))
        self.tmpfile.seek(0)
        res = self.tmpfile.buffer.read()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: bytes) -> None:
        if False:
            return 10
        'Write to original file descriptor.'
        self._assert_state('writeorg', ('started', 'suspended'))
        os.write(self.targetfd_save, data)

class FDCapture(FDCaptureBase[str]):
    """Capture IO to/from a given OS-level file descriptor.

    snap() produces text.
    """
    EMPTY_BUFFER = ''

    def snap(self) -> str:
        if False:
            i = 10
            return i + 15
        self._assert_state('snap', ('started', 'suspended'))
        self.tmpfile.seek(0)
        res = self.tmpfile.read()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: str) -> None:
        if False:
            while True:
                i = 10
        'Write to original file descriptor.'
        self._assert_state('writeorg', ('started', 'suspended'))
        os.write(self.targetfd_save, data.encode('utf-8'))
if sys.version_info >= (3, 11) or TYPE_CHECKING:

    @final
    class CaptureResult(NamedTuple, Generic[AnyStr]):
        """The result of :method:`CaptureFixture.readouterr`."""
        out: AnyStr
        err: AnyStr
else:

    class CaptureResult(collections.namedtuple('CaptureResult', ['out', 'err']), Generic[AnyStr]):
        """The result of :method:`CaptureFixture.readouterr`."""
        __slots__ = ()

class MultiCapture(Generic[AnyStr]):
    _state = None
    _in_suspended = False

    def __init__(self, in_: Optional[CaptureBase[AnyStr]], out: Optional[CaptureBase[AnyStr]], err: Optional[CaptureBase[AnyStr]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.in_: Optional[CaptureBase[AnyStr]] = in_
        self.out: Optional[CaptureBase[AnyStr]] = out
        self.err: Optional[CaptureBase[AnyStr]] = err

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '<MultiCapture out={!r} err={!r} in_={!r} _state={!r} _in_suspended={!r}>'.format(self.out, self.err, self.in_, self._state, self._in_suspended)

    def start_capturing(self) -> None:
        if False:
            return 10
        self._state = 'started'
        if self.in_:
            self.in_.start()
        if self.out:
            self.out.start()
        if self.err:
            self.err.start()

    def pop_outerr_to_orig(self) -> Tuple[AnyStr, AnyStr]:
        if False:
            return 10
        'Pop current snapshot out/err capture and flush to orig streams.'
        (out, err) = self.readouterr()
        if out:
            assert self.out is not None
            self.out.writeorg(out)
        if err:
            assert self.err is not None
            self.err.writeorg(err)
        return (out, err)

    def suspend_capturing(self, in_: bool=False) -> None:
        if False:
            print('Hello World!')
        self._state = 'suspended'
        if self.out:
            self.out.suspend()
        if self.err:
            self.err.suspend()
        if in_ and self.in_:
            self.in_.suspend()
            self._in_suspended = True

    def resume_capturing(self) -> None:
        if False:
            i = 10
            return i + 15
        self._state = 'started'
        if self.out:
            self.out.resume()
        if self.err:
            self.err.resume()
        if self._in_suspended:
            assert self.in_ is not None
            self.in_.resume()
            self._in_suspended = False

    def stop_capturing(self) -> None:
        if False:
            return 10
        'Stop capturing and reset capturing streams.'
        if self._state == 'stopped':
            raise ValueError('was already stopped')
        self._state = 'stopped'
        if self.out:
            self.out.done()
        if self.err:
            self.err.done()
        if self.in_:
            self.in_.done()

    def is_started(self) -> bool:
        if False:
            return 10
        'Whether actively capturing -- not suspended or stopped.'
        return self._state == 'started'

    def readouterr(self) -> CaptureResult[AnyStr]:
        if False:
            for i in range(10):
                print('nop')
        out = self.out.snap() if self.out else ''
        err = self.err.snap() if self.err else ''
        return CaptureResult(out, err)

def _get_multicapture(method: _CaptureMethod) -> MultiCapture[str]:
    if False:
        while True:
            i = 10
    if method == 'fd':
        return MultiCapture(in_=FDCapture(0), out=FDCapture(1), err=FDCapture(2))
    elif method == 'sys':
        return MultiCapture(in_=SysCapture(0), out=SysCapture(1), err=SysCapture(2))
    elif method == 'no':
        return MultiCapture(in_=None, out=None, err=None)
    elif method == 'tee-sys':
        return MultiCapture(in_=None, out=SysCapture(1, tee=True), err=SysCapture(2, tee=True))
    raise ValueError(f'unknown capturing method: {method!r}')

class CaptureManager:
    """The capture plugin.

    Manages that the appropriate capture method is enabled/disabled during
    collection and each test phase (setup, call, teardown). After each of
    those points, the captured output is obtained and attached to the
    collection/runtest report.

    There are two levels of capture:

    * global: enabled by default and can be suppressed by the ``-s``
      option. This is always enabled/disabled during collection and each test
      phase.

    * fixture: when a test function or one of its fixture depend on the
      ``capsys`` or ``capfd`` fixtures. In this case special handling is
      needed to ensure the fixtures take precedence over the global capture.
    """

    def __init__(self, method: _CaptureMethod) -> None:
        if False:
            while True:
                i = 10
        self._method: Final = method
        self._global_capturing: Optional[MultiCapture[str]] = None
        self._capture_fixture: Optional[CaptureFixture[Any]] = None

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<CaptureManager _method={!r} _global_capturing={!r} _capture_fixture={!r}>'.format(self._method, self._global_capturing, self._capture_fixture)

    def is_capturing(self) -> Union[str, bool]:
        if False:
            i = 10
            return i + 15
        if self.is_globally_capturing():
            return 'global'
        if self._capture_fixture:
            return 'fixture %s' % self._capture_fixture.request.fixturename
        return False

    def is_globally_capturing(self) -> bool:
        if False:
            print('Hello World!')
        return self._method != 'no'

    def start_global_capturing(self) -> None:
        if False:
            return 10
        assert self._global_capturing is None
        self._global_capturing = _get_multicapture(self._method)
        self._global_capturing.start_capturing()

    def stop_global_capturing(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._global_capturing is not None:
            self._global_capturing.pop_outerr_to_orig()
            self._global_capturing.stop_capturing()
            self._global_capturing = None

    def resume_global_capture(self) -> None:
        if False:
            print('Hello World!')
        if self._global_capturing is not None:
            self._global_capturing.resume_capturing()

    def suspend_global_capture(self, in_: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        if self._global_capturing is not None:
            self._global_capturing.suspend_capturing(in_=in_)

    def suspend(self, in_: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self.suspend_fixture()
        self.suspend_global_capture(in_)

    def resume(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.resume_global_capture()
        self.resume_fixture()

    def read_global_capture(self) -> CaptureResult[str]:
        if False:
            while True:
                i = 10
        assert self._global_capturing is not None
        return self._global_capturing.readouterr()

    def set_fixture(self, capture_fixture: 'CaptureFixture[Any]') -> None:
        if False:
            print('Hello World!')
        if self._capture_fixture:
            current_fixture = self._capture_fixture.request.fixturename
            requested_fixture = capture_fixture.request.fixturename
            capture_fixture.request.raiseerror('cannot use {} and {} at the same time'.format(requested_fixture, current_fixture))
        self._capture_fixture = capture_fixture

    def unset_fixture(self) -> None:
        if False:
            i = 10
            return i + 15
        self._capture_fixture = None

    def activate_fixture(self) -> None:
        if False:
            print('Hello World!')
        'If the current item is using ``capsys`` or ``capfd``, activate\n        them so they take precedence over the global capture.'
        if self._capture_fixture:
            self._capture_fixture._start()

    def deactivate_fixture(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Deactivate the ``capsys`` or ``capfd`` fixture of this item, if any.'
        if self._capture_fixture:
            self._capture_fixture.close()

    def suspend_fixture(self) -> None:
        if False:
            while True:
                i = 10
        if self._capture_fixture:
            self._capture_fixture._suspend()

    def resume_fixture(self) -> None:
        if False:
            print('Hello World!')
        if self._capture_fixture:
            self._capture_fixture._resume()

    @contextlib.contextmanager
    def global_and_fixture_disabled(self) -> Generator[None, None, None]:
        if False:
            print('Hello World!')
        'Context manager to temporarily disable global and current fixture capturing.'
        do_fixture = self._capture_fixture and self._capture_fixture._is_started()
        if do_fixture:
            self.suspend_fixture()
        do_global = self._global_capturing and self._global_capturing.is_started()
        if do_global:
            self.suspend_global_capture()
        try:
            yield
        finally:
            if do_global:
                self.resume_global_capture()
            if do_fixture:
                self.resume_fixture()

    @contextlib.contextmanager
    def item_capture(self, when: str, item: Item) -> Generator[None, None, None]:
        if False:
            return 10
        self.resume_global_capture()
        self.activate_fixture()
        try:
            yield
        finally:
            self.deactivate_fixture()
            self.suspend_global_capture(in_=False)
            (out, err) = self.read_global_capture()
            item.add_report_section(when, 'stdout', out)
            item.add_report_section(when, 'stderr', err)

    @hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: Collector) -> Generator[None, CollectReport, CollectReport]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(collector, File):
            self.resume_global_capture()
            try:
                rep = (yield)
            finally:
                self.suspend_global_capture()
            (out, err) = self.read_global_capture()
            if out:
                rep.sections.append(('Captured stdout', out))
            if err:
                rep.sections.append(('Captured stderr', err))
        else:
            rep = (yield)
        return rep

    @hookimpl(wrapper=True)
    def pytest_runtest_setup(self, item: Item) -> Generator[None, None, None]:
        if False:
            print('Hello World!')
        with self.item_capture('setup', item):
            return (yield)

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None, None, None]:
        if False:
            return 10
        with self.item_capture('call', item):
            return (yield)

    @hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item: Item) -> Generator[None, None, None]:
        if False:
            print('Hello World!')
        with self.item_capture('teardown', item):
            return (yield)

    @hookimpl(tryfirst=True)
    def pytest_keyboard_interrupt(self) -> None:
        if False:
            print('Hello World!')
        self.stop_global_capturing()

    @hookimpl(tryfirst=True)
    def pytest_internalerror(self) -> None:
        if False:
            i = 10
            return i + 15
        self.stop_global_capturing()

class CaptureFixture(Generic[AnyStr]):
    """Object returned by the :fixture:`capsys`, :fixture:`capsysbinary`,
    :fixture:`capfd` and :fixture:`capfdbinary` fixtures."""

    def __init__(self, captureclass: Type[CaptureBase[AnyStr]], request: SubRequest, *, _ispytest: bool=False) -> None:
        if False:
            while True:
                i = 10
        check_ispytest(_ispytest)
        self.captureclass: Type[CaptureBase[AnyStr]] = captureclass
        self.request = request
        self._capture: Optional[MultiCapture[AnyStr]] = None
        self._captured_out: AnyStr = self.captureclass.EMPTY_BUFFER
        self._captured_err: AnyStr = self.captureclass.EMPTY_BUFFER

    def _start(self) -> None:
        if False:
            print('Hello World!')
        if self._capture is None:
            self._capture = MultiCapture(in_=None, out=self.captureclass(1), err=self.captureclass(2))
            self._capture.start_capturing()

    def close(self) -> None:
        if False:
            print('Hello World!')
        if self._capture is not None:
            (out, err) = self._capture.pop_outerr_to_orig()
            self._captured_out += out
            self._captured_err += err
            self._capture.stop_capturing()
            self._capture = None

    def readouterr(self) -> CaptureResult[AnyStr]:
        if False:
            print('Hello World!')
        'Read and return the captured output so far, resetting the internal\n        buffer.\n\n        :returns:\n            The captured content as a namedtuple with ``out`` and ``err``\n            string attributes.\n        '
        (captured_out, captured_err) = (self._captured_out, self._captured_err)
        if self._capture is not None:
            (out, err) = self._capture.readouterr()
            captured_out += out
            captured_err += err
        self._captured_out = self.captureclass.EMPTY_BUFFER
        self._captured_err = self.captureclass.EMPTY_BUFFER
        return CaptureResult(captured_out, captured_err)

    def _suspend(self) -> None:
        if False:
            return 10
        "Suspend this fixture's own capturing temporarily."
        if self._capture is not None:
            self._capture.suspend_capturing()

    def _resume(self) -> None:
        if False:
            i = 10
            return i + 15
        "Resume this fixture's own capturing temporarily."
        if self._capture is not None:
            self._capture.resume_capturing()

    def _is_started(self) -> bool:
        if False:
            return 10
        'Whether actively capturing -- not disabled or closed.'
        if self._capture is not None:
            return self._capture.is_started()
        return False

    @contextlib.contextmanager
    def disabled(self) -> Generator[None, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Temporarily disable capturing while inside the ``with`` block.'
        capmanager: CaptureManager = self.request.config.pluginmanager.getplugin('capturemanager')
        with capmanager.global_and_fixture_disabled():
            yield

@fixture
def capsys(request: SubRequest) -> Generator[CaptureFixture[str], None, None]:
    if False:
        print('Hello World!')
    'Enable text capturing of writes to ``sys.stdout`` and ``sys.stderr``.\n\n    The captured output is made available via ``capsys.readouterr()`` method\n    calls, which return a ``(out, err)`` namedtuple.\n    ``out`` and ``err`` will be ``text`` objects.\n\n    Returns an instance of :class:`CaptureFixture[str] <pytest.CaptureFixture>`.\n\n    Example:\n\n    .. code-block:: python\n\n        def test_output(capsys):\n            print("hello")\n            captured = capsys.readouterr()\n            assert captured.out == "hello\\n"\n    '
    capman: CaptureManager = request.config.pluginmanager.getplugin('capturemanager')
    capture_fixture = CaptureFixture(SysCapture, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()

@fixture
def capsysbinary(request: SubRequest) -> Generator[CaptureFixture[bytes], None, None]:
    if False:
        i = 10
        return i + 15
    'Enable bytes capturing of writes to ``sys.stdout`` and ``sys.stderr``.\n\n    The captured output is made available via ``capsysbinary.readouterr()``\n    method calls, which return a ``(out, err)`` namedtuple.\n    ``out`` and ``err`` will be ``bytes`` objects.\n\n    Returns an instance of :class:`CaptureFixture[bytes] <pytest.CaptureFixture>`.\n\n    Example:\n\n    .. code-block:: python\n\n        def test_output(capsysbinary):\n            print("hello")\n            captured = capsysbinary.readouterr()\n            assert captured.out == b"hello\\n"\n    '
    capman: CaptureManager = request.config.pluginmanager.getplugin('capturemanager')
    capture_fixture = CaptureFixture(SysCaptureBinary, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()

@fixture
def capfd(request: SubRequest) -> Generator[CaptureFixture[str], None, None]:
    if False:
        for i in range(10):
            print('nop')
    'Enable text capturing of writes to file descriptors ``1`` and ``2``.\n\n    The captured output is made available via ``capfd.readouterr()`` method\n    calls, which return a ``(out, err)`` namedtuple.\n    ``out`` and ``err`` will be ``text`` objects.\n\n    Returns an instance of :class:`CaptureFixture[str] <pytest.CaptureFixture>`.\n\n    Example:\n\n    .. code-block:: python\n\n        def test_system_echo(capfd):\n            os.system(\'echo "hello"\')\n            captured = capfd.readouterr()\n            assert captured.out == "hello\\n"\n    '
    capman: CaptureManager = request.config.pluginmanager.getplugin('capturemanager')
    capture_fixture = CaptureFixture(FDCapture, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()

@fixture
def capfdbinary(request: SubRequest) -> Generator[CaptureFixture[bytes], None, None]:
    if False:
        for i in range(10):
            print('nop')
    'Enable bytes capturing of writes to file descriptors ``1`` and ``2``.\n\n    The captured output is made available via ``capfd.readouterr()`` method\n    calls, which return a ``(out, err)`` namedtuple.\n    ``out`` and ``err`` will be ``byte`` objects.\n\n    Returns an instance of :class:`CaptureFixture[bytes] <pytest.CaptureFixture>`.\n\n    Example:\n\n    .. code-block:: python\n\n        def test_system_echo(capfdbinary):\n            os.system(\'echo "hello"\')\n            captured = capfdbinary.readouterr()\n            assert captured.out == b"hello\\n"\n\n    '
    capman: CaptureManager = request.config.pluginmanager.getplugin('capturemanager')
    capture_fixture = CaptureFixture(FDCaptureBinary, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()