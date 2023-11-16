"""Misc. utilities related to Qt.

Module attributes:
    MAXVALS: A dictionary of C/Qt types (as string) mapped to their maximum
             value.
    MINVALS: A dictionary of C/Qt types (as string) mapped to their minimum
             value.
    MAX_WORLD_ID: The highest world ID allowed by QtWebEngine.
"""
import io
import enum
import pathlib
import operator
import contextlib
from typing import Any, AnyStr, TYPE_CHECKING, BinaryIO, IO, Iterator, Optional, Union, Tuple, Protocol, cast, TypeVar
from qutebrowser.qt import machinery, sip
from qutebrowser.qt.core import qVersion, QEventLoop, QDataStream, QByteArray, QIODevice, QFileDevice, QSaveFile, QT_VERSION_STR, PYQT_VERSION_STR, QObject, QUrl, QLibraryInfo
from qutebrowser.qt.gui import QColor
try:
    from qutebrowser.qt.webkit import qWebKitVersion
except ImportError:
    qWebKitVersion = None
if TYPE_CHECKING:
    from qutebrowser.qt.webkit import QWebHistory
    from qutebrowser.qt.webenginecore import QWebEngineHistory
    from typing_extensions import TypeGuard
from qutebrowser.misc import objects
from qutebrowser.utils import usertypes, utils
MAXVALS = {'int': 2 ** 31 - 1, 'int64': 2 ** 63 - 1}
MINVALS = {'int': -2 ** 31, 'int64': -2 ** 63}

class QtOSError(OSError):
    """An OSError triggered by a QIODevice.

    Attributes:
        qt_errno: The error attribute of the given QFileDevice, if applicable.
    """

    def __init__(self, dev: QIODevice, msg: str=None) -> None:
        if False:
            return 10
        if msg is None:
            msg = dev.errorString()
        self.qt_errno: Optional[QFileDevice.FileError] = None
        if isinstance(dev, QFileDevice):
            msg = self._init_filedev(dev, msg)
        super().__init__(msg)

    def _init_filedev(self, dev: QFileDevice, msg: str) -> str:
        if False:
            return 10
        self.qt_errno = dev.error()
        filename = dev.fileName()
        msg += ': {!r}'.format(filename)
        return msg

def version_check(version: str, exact: bool=False, compiled: bool=True) -> bool:
    if False:
        i = 10
        return i + 15
    "Check if the Qt runtime version is the version supplied or newer.\n\n    By default this function will check `version` against:\n\n    1. the runtime Qt version (from qVersion())\n    2. the Qt version that PyQt was compiled against (from QT_VERSION_STR)\n    3. the PyQt version (from PYQT_VERSION_STR)\n\n    With `compiled=False` only the runtime Qt version (1) is checked.\n\n    You can often run older PyQt versions against newer Qt versions, but you\n    won't be able to access any APIs that were only added in the newer Qt\n    version. So if you want to check if a new feature is supported, use the\n    default behavior. If you just want to check the underlying Qt version,\n    pass `compiled=False`.\n\n    Args:\n        version: The version to check against.\n        exact: if given, check with == instead of >=\n        compiled: Set to False to not check the compiled Qt version or the\n          PyQt version.\n    "
    if compiled and exact:
        raise ValueError("Can't use compiled=True with exact=True!")
    parsed = utils.VersionNumber.parse(version)
    op = operator.eq if exact else operator.ge
    qversion = qVersion()
    assert qversion is not None
    result = op(utils.VersionNumber.parse(qversion), parsed)
    if compiled and result:
        result = op(utils.VersionNumber.parse(QT_VERSION_STR), parsed)
    if compiled and result:
        result = op(utils.VersionNumber.parse(PYQT_VERSION_STR), parsed)
    return result
MAX_WORLD_ID = 256

def is_new_qtwebkit() -> bool:
    if False:
        while True:
            i = 10
    'Check if the given version is a new QtWebKit.'
    assert qWebKitVersion is not None
    return utils.VersionNumber.parse(qWebKitVersion()) > utils.VersionNumber.parse('538.1')

def is_single_process() -> bool:
    if False:
        print('Hello World!')
    'Check whether QtWebEngine is running in single-process mode.'
    if objects.backend == usertypes.Backend.QtWebKit:
        return False
    assert objects.backend == usertypes.Backend.QtWebEngine, objects.backend
    args = objects.qapp.arguments()
    return '--single-process' in args

def is_wayland() -> bool:
    if False:
        i = 10
        return i + 15
    'Check if we are running on Wayland.'
    return objects.qapp.platformName() in ['wayland', 'wayland-egl']

def check_overflow(arg: int, ctype: str, fatal: bool=True) -> int:
    if False:
        print('Hello World!')
    "Check if the given argument is in bounds for the given type.\n\n    Args:\n        arg: The argument to check\n        ctype: The C/Qt type to check as a string.\n        fatal: Whether to raise exceptions (True) or truncate values (False)\n\n    Return\n        The truncated argument if fatal=False\n        The original argument if it's in bounds.\n    "
    maxval = MAXVALS[ctype]
    minval = MINVALS[ctype]
    if arg > maxval:
        if fatal:
            raise OverflowError(arg)
        return maxval
    elif arg < minval:
        if fatal:
            raise OverflowError(arg)
        return minval
    else:
        return arg

class Validatable(Protocol):
    """An object with an isValid() method (e.g. QUrl)."""

    def isValid(self) -> bool:
        if False:
            while True:
                i = 10
        ...

def ensure_valid(obj: Validatable) -> None:
    if False:
        print('Hello World!')
    'Ensure a Qt object with an .isValid() method is valid.'
    if not obj.isValid():
        raise QtValueError(obj)

def check_qdatastream(stream: QDataStream) -> None:
    if False:
        return 10
    "Check the status of a QDataStream and raise OSError if it's not ok."
    status_to_str = {QDataStream.Status.Ok: 'The data stream is operating normally.', QDataStream.Status.ReadPastEnd: 'The data stream has read past the end of the data in the underlying device.', QDataStream.Status.ReadCorruptData: 'The data stream has read corrupt data.', QDataStream.Status.WriteFailed: 'The data stream cannot write to the underlying device.'}
    if stream.status() != QDataStream.Status.Ok:
        raise OSError(status_to_str[stream.status()])
_QtSerializableType = Union[QObject, QByteArray, QUrl, 'QWebEngineHistory', 'QWebHistory']

def serialize(obj: _QtSerializableType) -> QByteArray:
    if False:
        for i in range(10):
            print('nop')
    'Serialize an object into a QByteArray.'
    data = QByteArray()
    stream = QDataStream(data, QIODevice.OpenModeFlag.WriteOnly)
    serialize_stream(stream, obj)
    return data

def deserialize(data: QByteArray, obj: _QtSerializableType) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deserialize an object from a QByteArray.'
    stream = QDataStream(data, QIODevice.OpenModeFlag.ReadOnly)
    deserialize_stream(stream, obj)

def serialize_stream(stream: QDataStream, obj: _QtSerializableType) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Serialize an object into a QDataStream.'
    check_qdatastream(stream)
    stream << obj
    check_qdatastream(stream)

def deserialize_stream(stream: QDataStream, obj: _QtSerializableType) -> None:
    if False:
        i = 10
        return i + 15
    'Deserialize a QDataStream into an object.'
    check_qdatastream(stream)
    stream >> obj
    check_qdatastream(stream)

@contextlib.contextmanager
def savefile_open(filename: str, binary: bool=False, encoding: str='utf-8') -> Iterator[IO[AnyStr]]:
    if False:
        while True:
            i = 10
    'Context manager to easily use a QSaveFile.'
    f = QSaveFile(filename)
    cancelled = False
    try:
        open_ok = f.open(QIODevice.OpenModeFlag.WriteOnly)
        if not open_ok:
            raise QtOSError(f)
        dev = cast(BinaryIO, PyQIODevice(f))
        if binary:
            new_f: IO[Any] = dev
        else:
            new_f = io.TextIOWrapper(dev, encoding=encoding)
        yield new_f
        new_f.flush()
    except:
        f.cancelWriting()
        cancelled = True
        raise
    finally:
        commit_ok = f.commit()
        if not commit_ok and (not cancelled):
            raise QtOSError(f, msg='Commit failed!')

def qcolor_to_qsscolor(c: QColor) -> str:
    if False:
        i = 10
        return i + 15
    'Convert a QColor to a string that can be used in a QStyleSheet.'
    ensure_valid(c)
    return 'rgba({}, {}, {}, {})'.format(c.red(), c.green(), c.blue(), c.alpha())

class PyQIODevice(io.BufferedIOBase):
    """Wrapper for a QIODevice which provides a python interface.

    Attributes:
        dev: The underlying QIODevice.
    """

    def __init__(self, dev: QIODevice) -> None:
        if False:
            return 10
        super().__init__()
        self.dev = dev

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return self.dev.size()

    def _check_open(self) -> None:
        if False:
            print('Hello World!')
        'Check if the device is open, raise ValueError if not.'
        if not self.dev.isOpen():
            raise ValueError('IO operation on closed device!')

    def _check_random(self) -> None:
        if False:
            print('Hello World!')
        'Check if the device supports random access, raise OSError if not.'
        if not self.seekable():
            raise OSError('Random access not allowed!')

    def _check_readable(self) -> None:
        if False:
            return 10
        'Check if the device is readable, raise OSError if not.'
        if not self.dev.isReadable():
            raise OSError('Trying to read unreadable file!')

    def _check_writable(self) -> None:
        if False:
            while True:
                i = 10
        'Check if the device is writable, raise OSError if not.'
        if not self.writable():
            raise OSError('Trying to write to unwritable file!')

    def open(self, mode: QIODevice.OpenModeFlag) -> contextlib.closing:
        if False:
            return 10
        'Open the underlying device and ensure opening succeeded.\n\n        Raises OSError if opening failed.\n\n        Args:\n            mode: QIODevice::OpenMode flags.\n\n        Return:\n            A contextlib.closing() object so this can be used as\n            contextmanager.\n        '
        ok = self.dev.open(mode)
        if not ok:
            raise QtOSError(self.dev)
        return contextlib.closing(self)

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        'Close the underlying device.'
        self.dev.close()

    def fileno(self) -> int:
        if False:
            return 10
        raise io.UnsupportedOperation

    def seek(self, offset: int, whence: int=io.SEEK_SET) -> int:
        if False:
            return 10
        self._check_open()
        self._check_random()
        if whence == io.SEEK_SET:
            ok = self.dev.seek(offset)
        elif whence == io.SEEK_CUR:
            ok = self.dev.seek(self.tell() + offset)
        elif whence == io.SEEK_END:
            ok = self.dev.seek(len(self) + offset)
        else:
            raise io.UnsupportedOperation('whence = {} is not supported!'.format(whence))
        if not ok:
            raise QtOSError(self.dev, msg='seek failed!')
        return self.dev.pos()

    def truncate(self, size: int=None) -> int:
        if False:
            print('Hello World!')
        raise io.UnsupportedOperation

    @property
    def closed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self.dev.isOpen()

    def flush(self) -> None:
        if False:
            print('Hello World!')
        self._check_open()
        self.dev.waitForBytesWritten(-1)

    def isatty(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        self._check_open()
        return False

    def readable(self) -> bool:
        if False:
            return 10
        return self.dev.isReadable()

    def readline(self, size: Optional[int]=-1) -> bytes:
        if False:
            return 10
        self._check_open()
        self._check_readable()
        if size is None or size < 0:
            qt_size = None
        elif size == 0:
            return b''
        else:
            qt_size = size + 1
        buf: Union[QByteArray, bytes, None] = None
        if self.dev.canReadLine():
            if qt_size is None:
                buf = self.dev.readLine()
            else:
                buf = self.dev.readLine(qt_size)
        elif size is None or size < 0:
            buf = self.dev.readAll()
        else:
            buf = self.dev.read(size)
        if buf is None:
            raise QtOSError(self.dev)
        if isinstance(buf, QByteArray):
            buf = buf.data()
        return buf

    def seekable(self) -> bool:
        if False:
            return 10
        return not self.dev.isSequential()

    def tell(self) -> int:
        if False:
            print('Hello World!')
        self._check_open()
        self._check_random()
        return self.dev.pos()

    def writable(self) -> bool:
        if False:
            return 10
        return self.dev.isWritable()

    def write(self, data: Union[bytes, bytearray]) -> int:
        if False:
            for i in range(10):
                print('nop')
        self._check_open()
        self._check_writable()
        num = self.dev.write(data)
        if num == -1 or num < len(data):
            raise QtOSError(self.dev)
        return num

    def read(self, size: Optional[int]=None) -> bytes:
        if False:
            print('Hello World!')
        self._check_open()
        self._check_readable()
        buf: Union[QByteArray, bytes, None] = None
        if size in [None, -1]:
            buf = self.dev.readAll()
        else:
            assert size is not None
            buf = self.dev.read(size)
        if buf is None:
            raise QtOSError(self.dev)
        if isinstance(buf, QByteArray):
            buf = buf.data()
        return buf

class QtValueError(ValueError):
    """Exception which gets raised by ensure_valid."""

    def __init__(self, obj: Validatable) -> None:
        if False:
            print('Hello World!')
        try:
            self.reason = obj.errorString()
        except AttributeError:
            self.reason = None
        err = '{} is not valid'.format(obj)
        if self.reason:
            err += ': {}'.format(self.reason)
        super().__init__(err)
if machinery.IS_QT6:
    _ProcessEventFlagType = QEventLoop.ProcessEventsFlag
else:
    _ProcessEventFlagType = Union[QEventLoop.ProcessEventsFlag, QEventLoop.ProcessEventsFlags]

class EventLoop(QEventLoop):
    """A thin wrapper around QEventLoop.

    Raises an exception when doing exec() multiple times.
    """

    def __init__(self, parent: QObject=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._executing = False

    def exec(self, flags: _ProcessEventFlagType=QEventLoop.ProcessEventsFlag.AllEvents) -> int:
        if False:
            return 10
        'Override exec_ to raise an exception when re-running.'
        if self._executing:
            raise AssertionError('Eventloop is already running!')
        self._executing = True
        if machinery.IS_QT5:
            flags = cast(QEventLoop.ProcessEventsFlags, flags)
        status = super().exec(flags)
        self._executing = False
        return status

def _get_color_percentage(x1: int, y1: int, z1: int, a1: int, x2: int, y2: int, z2: int, a2: int, percent: int) -> Tuple[int, int, int, int]:
    if False:
        print('Hello World!')
    'Get a color which is percent% interpolated between start and end.\n\n    Args:\n        x1, y1, z1, a1 : Start color components (R, G, B, A / H, S, V, A / H, S, L, A)\n        x2, y2, z2, a2 : End color components (R, G, B, A / H, S, V, A / H, S, L, A)\n        percent: Percentage to interpolate, 0-100.\n                 0: Start color will be returned.\n                 100: End color will be returned.\n\n    Return:\n        A (x, y, z, alpha) tuple with the interpolated color components.\n    '
    if not 0 <= percent <= 100:
        raise ValueError('percent needs to be between 0 and 100!')
    x = round(x1 + (x2 - x1) * percent / 100)
    y = round(y1 + (y2 - y1) * percent / 100)
    z = round(z1 + (z2 - z1) * percent / 100)
    a = round(a1 + (a2 - a1) * percent / 100)
    return (x, y, z, a)

def interpolate_color(start: QColor, end: QColor, percent: int, colorspace: Optional[QColor.Spec]=QColor.Spec.Rgb) -> QColor:
    if False:
        i = 10
        return i + 15
    'Get an interpolated color value.\n\n    Args:\n        start: The start color.\n        end: The end color.\n        percent: Which value to get (0 - 100)\n        colorspace: The desired interpolation color system,\n                    QColor::{Rgb,Hsv,Hsl} (from QColor::Spec enum)\n                    If None, start is used except when percent is 100.\n\n    Return:\n        The interpolated QColor, with the same spec as the given start color.\n    '
    ensure_valid(start)
    ensure_valid(end)
    if colorspace is None:
        if percent == 100:
            (r, g, b, a) = end.getRgb()
            assert r is not None
            assert g is not None
            assert b is not None
            assert a is not None
            return QColor(r, g, b, a)
        else:
            (r, g, b, a) = start.getRgb()
            assert r is not None
            assert g is not None
            assert b is not None
            assert a is not None
            return QColor(r, g, b, a)
    out = QColor()
    if colorspace == QColor.Spec.Rgb:
        (r1, g1, b1, a1) = start.getRgb()
        (r2, g2, b2, a2) = end.getRgb()
        assert r1 is not None
        assert g1 is not None
        assert b1 is not None
        assert a1 is not None
        assert r2 is not None
        assert g2 is not None
        assert b2 is not None
        assert a2 is not None
        components = _get_color_percentage(r1, g1, b1, a1, r2, g2, b2, a2, percent)
        out.setRgb(*components)
    elif colorspace == QColor.Spec.Hsv:
        (h1, s1, v1, a1) = start.getHsv()
        (h2, s2, v2, a2) = end.getHsv()
        assert h1 is not None
        assert s1 is not None
        assert v1 is not None
        assert a1 is not None
        assert h2 is not None
        assert s2 is not None
        assert v2 is not None
        assert a2 is not None
        components = _get_color_percentage(h1, s1, v1, a1, h2, s2, v2, a2, percent)
        out.setHsv(*components)
    elif colorspace == QColor.Spec.Hsl:
        (h1, s1, l1, a1) = start.getHsl()
        (h2, s2, l2, a2) = end.getHsl()
        assert h1 is not None
        assert s1 is not None
        assert l1 is not None
        assert a1 is not None
        assert h2 is not None
        assert s2 is not None
        assert l2 is not None
        assert a2 is not None
        components = _get_color_percentage(h1, s1, l1, a1, h2, s2, l2, a2, percent)
        out.setHsl(*components)
    else:
        raise ValueError('Invalid colorspace!')
    out = out.convertTo(start.spec())
    ensure_valid(out)
    return out

class LibraryPath(enum.Enum):
    """A path to be passed to QLibraryInfo.

    Should mirror QLibraryPath (Qt 5) and QLibraryLocation (Qt 6).
    Values are the respective Qt names.
    """
    prefix = 'PrefixPath'
    documentation = 'DocumentationPath'
    headers = 'HeadersPath'
    libraries = 'LibrariesPath'
    library_executables = 'LibraryExecutablesPath'
    binaries = 'BinariesPath'
    plugins = 'PluginsPath'
    qml2_imports = 'Qml2ImportsPath'
    arch_data = 'ArchDataPath'
    data = 'DataPath'
    translations = 'TranslationsPath'
    examples = 'ExamplesPath'
    tests = 'TestsPath'
    settings = 'SettingsPath'

def library_path(which: LibraryPath) -> pathlib.Path:
    if False:
        while True:
            i = 10
    'Wrapper around QLibraryInfo.path / .location.'
    if machinery.IS_QT6:
        val = getattr(QLibraryInfo.LibraryPath, which.value)
        ret = QLibraryInfo.path(val)
    else:
        val = getattr(QLibraryInfo.LibraryLocation, which.value)
        ret = QLibraryInfo.location(val)
    assert ret
    return pathlib.Path(ret)

def extract_enum_val(val: Union[sip.simplewrapper, int, enum.Enum]) -> int:
    if False:
        return 10
    'Extract an int value from a Qt enum value.\n\n    For Qt 5, enum values are basically Python integers.\n    For Qt 6, they are usually enum.Enum instances, with the value set to the\n    integer.\n    '
    if isinstance(val, enum.Enum):
        return val.value
    elif isinstance(val, sip.simplewrapper):
        return int(val)
    return val

def qobj_repr(obj: Optional[QObject]) -> str:
    if False:
        i = 10
        return i + 15
    'Show nicer debug information for a QObject.'
    py_repr = repr(obj)
    if obj is None:
        return py_repr
    try:
        object_name = obj.objectName()
        meta_object = obj.metaObject()
    except AttributeError:
        return py_repr
    class_name = '' if meta_object is None else meta_object.className()
    if py_repr.startswith('<') and py_repr.endswith('>'):
        py_repr = py_repr[1:-1]
    parts = [py_repr]
    if object_name:
        parts.append(f'objectName={object_name!r}')
    if class_name and f'.{class_name} object at 0x' not in py_repr:
        parts.append(f'className={class_name!r}')
    return f"<{', '.join(parts)}>"
_T = TypeVar('_T')
if machinery.IS_QT5:

    def remove_optional(obj: Optional[_T]) -> _T:
        if False:
            while True:
                i = 10
        return cast(_T, obj)

    def add_optional(obj: _T) -> Optional[_T]:
        if False:
            i = 10
            return i + 15
        return cast(Optional[_T], obj)
    QT_NONE: Any = None
else:

    def remove_optional(obj: Optional[_T]) -> Optional[_T]:
        if False:
            for i in range(10):
                print('nop')
        return obj

    def add_optional(obj: Optional[_T]) -> Optional[_T]:
        if False:
            return 10
        return obj
    QT_NONE = None