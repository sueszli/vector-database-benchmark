"""Tests for qutebrowser.utils.qtutils."""
import io
import os
import pathlib
import dataclasses
import unittest
import unittest.mock
import pytest
from qutebrowser.qt.core import QDataStream, QPoint, QUrl, QByteArray, QIODevice, QTimer, QBuffer, QFile, QProcess, QFileDevice, QLibraryInfo, Qt, QObject
from qutebrowser.qt.gui import QColor
from qutebrowser.qt import sip
from qutebrowser.utils import qtutils, utils, usertypes
import overflow_test_cases
from helpers import testutils
if utils.is_linux:
    try:
        from test import test_file
    except ImportError:
        test_file = None
else:
    test_file = None

@pytest.mark.parametrize('qversion, compiled, pyqt, version, exact, expected', [('5.14.0', None, None, '5.14.0', False, True), ('5.14.0', None, None, '5.14.0', True, True), ('5.14.0', None, None, '5.14', True, True), ('5.14.1', None, None, '5.14', False, True), ('5.14.1', None, None, '5.14', True, False), ('5.13.2', None, None, '5.14', False, False), ('5.13.0', None, None, '5.13.2', False, False), ('5.13.0', None, None, '5.13.2', True, False), ('5.14.0', '5.13.0', '5.14.0', '5.14.0', False, False), ('5.14.0', '5.14.0', '5.13.0', '5.14.0', False, False), ('5.14.0', '5.14.0', '5.14.0', '5.14.0', False, True), ('5.15.1', '5.15.1', '5.15.2.dev2009281246', '5.15.0', False, True)])
def test_version_check(monkeypatch, qversion, compiled, pyqt, version, exact, expected):
    if False:
        for i in range(10):
            print('nop')
    'Test for version_check().\n\n    Args:\n        monkeypatch: The pytest monkeypatch fixture.\n        qversion: The version to set as fake qVersion().\n        compiled: The value for QT_VERSION_STR (set compiled=False)\n        pyqt: The value for PYQT_VERSION_STR (set compiled=False)\n        version: The version to compare with.\n        exact: Use exact comparing (==)\n        expected: The expected result.\n    '
    monkeypatch.setattr(qtutils, 'qVersion', lambda : qversion)
    if compiled is not None:
        monkeypatch.setattr(qtutils, 'QT_VERSION_STR', compiled)
        monkeypatch.setattr(qtutils, 'PYQT_VERSION_STR', pyqt)
        compiled_arg = True
    else:
        compiled_arg = False
    actual = qtutils.version_check(version, exact, compiled=compiled_arg)
    assert actual == expected

def test_version_check_compiled_and_exact():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        qtutils.version_check('1.2.3', exact=True, compiled=True)

@pytest.mark.parametrize('version, is_new', [('537.21', False), ('538.1', False), ('602.1', True)])
def test_is_new_qtwebkit(monkeypatch, version, is_new):
    if False:
        print('Hello World!')
    monkeypatch.setattr(qtutils, 'qWebKitVersion', lambda : version)
    assert qtutils.is_new_qtwebkit() == is_new

@pytest.mark.parametrize('backend, arguments, single_process', [(usertypes.Backend.QtWebKit, ['--single-process'], False), (usertypes.Backend.QtWebEngine, ['--single-process'], True), (usertypes.Backend.QtWebEngine, [], False)])
def test_is_single_process(monkeypatch, stubs, backend, arguments, single_process):
    if False:
        print('Hello World!')
    qapp = stubs.FakeQApplication(arguments=arguments)
    monkeypatch.setattr(qtutils.objects, 'qapp', qapp)
    monkeypatch.setattr(qtutils.objects, 'backend', backend)
    assert qtutils.is_single_process() == single_process

@pytest.mark.parametrize('platform, is_wayland', [('wayland', True), ('wayland-egl', True), ('xcb', False)])
def test_is_wayland(monkeypatch, stubs, platform, is_wayland):
    if False:
        while True:
            i = 10
    qapp = stubs.FakeQApplication(platform_name=platform)
    monkeypatch.setattr(qtutils.objects, 'qapp', qapp)
    assert qtutils.is_wayland() == is_wayland

class TestCheckOverflow:
    """Test check_overflow."""

    @pytest.mark.parametrize('ctype, val', overflow_test_cases.good_values())
    def test_good_values(self, ctype, val):
        if False:
            i = 10
            return i + 15
        'Test values which are inside bounds.'
        qtutils.check_overflow(val, ctype)

    @pytest.mark.parametrize('ctype, val', [(ctype, val) for (ctype, val, _) in overflow_test_cases.bad_values()])
    def test_bad_values_fatal(self, ctype, val):
        if False:
            while True:
                i = 10
        'Test values which are outside bounds with fatal=True.'
        with pytest.raises(OverflowError):
            qtutils.check_overflow(val, ctype)

    @pytest.mark.parametrize('ctype, val, repl', overflow_test_cases.bad_values())
    def test_bad_values_nonfatal(self, ctype, val, repl):
        if False:
            print('Hello World!')
        'Test values which are outside bounds with fatal=False.'
        newval = qtutils.check_overflow(val, ctype, fatal=False)
        assert newval == repl

class QtObject:
    """Fake Qt object for test_ensure."""

    def __init__(self, valid=True, null=False, error=None):
        if False:
            for i in range(10):
                print('nop')
        self._valid = valid
        self._null = null
        self._error = error

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<QtObject>'

    def errorString(self):
        if False:
            i = 10
            return i + 15
        'Get the fake error, or raise AttributeError if set to None.'
        if self._error is None:
            raise AttributeError
        return self._error

    def isValid(self):
        if False:
            print('Hello World!')
        return self._valid

    def isNull(self):
        if False:
            return 10
        return self._null

@pytest.mark.parametrize('obj, raising, exc_reason, exc_str', [(QtObject(valid=True, null=True), False, None, None), (QtObject(valid=True, null=False), False, None, None), (QtObject(valid=False, null=True), True, None, '<QtObject> is not valid'), (QtObject(valid=False, null=False), True, None, '<QtObject> is not valid'), (QtObject(valid=False, null=True, error='Test'), True, 'Test', '<QtObject> is not valid: Test')])
def test_ensure_valid(obj, raising, exc_reason, exc_str):
    if False:
        i = 10
        return i + 15
    'Test ensure_valid.\n\n    Args:\n        obj: The object to test with.\n        raising: Whether QtValueError is expected to be raised.\n        exc_reason: The expected .reason attribute of the exception.\n        exc_str: The expected string of the exception.\n    '
    if raising:
        with pytest.raises(qtutils.QtValueError) as excinfo:
            qtutils.ensure_valid(obj)
        assert excinfo.value.reason == exc_reason
        assert str(excinfo.value) == exc_str
    else:
        qtutils.ensure_valid(obj)

@pytest.mark.parametrize('status, raising, message', [(QDataStream.Status.Ok, False, None), (QDataStream.Status.ReadPastEnd, True, 'The data stream has read past the end of the data in the underlying device.'), (QDataStream.Status.ReadCorruptData, True, 'The data stream has read corrupt data.'), (QDataStream.Status.WriteFailed, True, 'The data stream cannot write to the underlying device.')])
def test_check_qdatastream(status, raising, message):
    if False:
        i = 10
        return i + 15
    'Test check_qdatastream.\n\n    Args:\n        status: The status to set on the QDataStream we test with.\n        raising: Whether check_qdatastream is expected to raise OSError.\n        message: The expected exception string.\n    '
    stream = QDataStream()
    stream.setStatus(status)
    if raising:
        with pytest.raises(OSError, match=message):
            qtutils.check_qdatastream(stream)
    else:
        qtutils.check_qdatastream(stream)

def test_qdatastream_status_count():
    if False:
        print('Hello World!')
    'Make sure no new members are added to QDataStream.Status.'
    status_vals = testutils.enum_members(QDataStream, QDataStream.Status)
    assert len(status_vals) == 4

@pytest.mark.parametrize('color, expected', [(QColor('red'), 'rgba(255, 0, 0, 255)'), (QColor('blue'), 'rgba(0, 0, 255, 255)'), (QColor(1, 3, 5, 7), 'rgba(1, 3, 5, 7)')])
def test_qcolor_to_qsscolor(color, expected):
    if False:
        return 10
    assert qtutils.qcolor_to_qsscolor(color) == expected

def test_qcolor_to_qsscolor_invalid():
    if False:
        i = 10
        return i + 15
    with pytest.raises(qtutils.QtValueError):
        qtutils.qcolor_to_qsscolor(QColor())

@pytest.mark.parametrize('obj', [QPoint(23, 42), QUrl('http://www.qutebrowser.org/')])
def test_serialize(obj):
    if False:
        i = 10
        return i + 15
    'Test a serialize/deserialize round trip.\n\n    Args:\n        obj: The object to test with.\n    '
    new_obj = type(obj)()
    qtutils.deserialize(qtutils.serialize(obj), new_obj)
    assert new_obj == obj

class TestSerializeStream:
    """Tests for serialize_stream and deserialize_stream."""

    def _set_status(self, stream, status):
        if False:
            for i in range(10):
                print('nop')
        'Helper function so mocks can set an error status when used.'
        stream.status.return_value = status

    @pytest.fixture
    def stream_mock(self):
        if False:
            i = 10
            return i + 15
        'Fixture providing a QDataStream-like mock.'
        m = unittest.mock.MagicMock(spec=QDataStream)
        m.status.return_value = QDataStream.Status.Ok
        return m

    def test_serialize_pre_error_mock(self, stream_mock):
        if False:
            print('Hello World!')
        'Test serialize_stream with an error already set.'
        stream_mock.status.return_value = QDataStream.Status.ReadCorruptData
        with pytest.raises(OSError, match='The data stream has read corrupt data.'):
            qtutils.serialize_stream(stream_mock, QPoint())
        assert not stream_mock.__lshift__.called

    def test_serialize_post_error_mock(self, stream_mock):
        if False:
            i = 10
            return i + 15
        'Test serialize_stream with an error while serializing.'
        obj = QPoint()
        stream_mock.__lshift__.side_effect = lambda _other: self._set_status(stream_mock, QDataStream.Status.ReadCorruptData)
        with pytest.raises(OSError, match='The data stream has read corrupt data.'):
            qtutils.serialize_stream(stream_mock, obj)
        stream_mock.__lshift__.assert_called_once_with(obj)

    def test_deserialize_pre_error_mock(self, stream_mock):
        if False:
            return 10
        'Test deserialize_stream with an error already set.'
        stream_mock.status.return_value = QDataStream.Status.ReadCorruptData
        with pytest.raises(OSError, match='The data stream has read corrupt data.'):
            qtutils.deserialize_stream(stream_mock, QPoint())
        assert not stream_mock.__rshift__.called

    def test_deserialize_post_error_mock(self, stream_mock):
        if False:
            for i in range(10):
                print('nop')
        'Test deserialize_stream with an error while deserializing.'
        obj = QPoint()
        stream_mock.__rshift__.side_effect = lambda _other: self._set_status(stream_mock, QDataStream.Status.ReadCorruptData)
        with pytest.raises(OSError, match='The data stream has read corrupt data.'):
            qtutils.deserialize_stream(stream_mock, obj)
        stream_mock.__rshift__.assert_called_once_with(obj)

    def test_round_trip_real_stream(self):
        if False:
            i = 10
            return i + 15
        'Test a round trip with a real QDataStream.'
        src_obj = QPoint(23, 42)
        dest_obj = QPoint()
        data = QByteArray()
        write_stream = QDataStream(data, QIODevice.OpenModeFlag.WriteOnly)
        qtutils.serialize_stream(write_stream, src_obj)
        read_stream = QDataStream(data, QIODevice.OpenModeFlag.ReadOnly)
        qtutils.deserialize_stream(read_stream, dest_obj)
        assert src_obj == dest_obj

    @pytest.mark.qt_log_ignore('^QIODevice::write.*: ReadOnly device')
    def test_serialize_readonly_stream(self):
        if False:
            for i in range(10):
                print('nop')
        'Test serialize_stream with a read-only stream.'
        data = QByteArray()
        stream = QDataStream(data, QIODevice.OpenModeFlag.ReadOnly)
        with pytest.raises(OSError, match='The data stream cannot write to the underlying device.'):
            qtutils.serialize_stream(stream, QPoint())

    @pytest.mark.qt_log_ignore('QIODevice::read.*: WriteOnly device')
    def test_deserialize_writeonly_stream(self):
        if False:
            i = 10
            return i + 15
        'Test deserialize_stream with a write-only stream.'
        data = QByteArray()
        obj = QPoint()
        stream = QDataStream(data, QIODevice.OpenModeFlag.WriteOnly)
        with pytest.raises(OSError, match='The data stream has read past the end of the data in the underlying device.'):
            qtutils.deserialize_stream(stream, obj)

class SavefileTestException(Exception):
    """Exception raised in TestSavefileOpen for testing."""

@pytest.mark.usefixtures('qapp')
class TestSavefileOpen:
    """Tests for savefile_open."""

    @pytest.fixture
    def qsavefile_mock(self, mocker):
        if False:
            while True:
                i = 10
        'Mock for QSaveFile.'
        m = mocker.patch('qutebrowser.utils.qtutils.QSaveFile')
        instance = m()
        yield instance
        instance.commit.assert_called_once_with()

    def test_mock_open_error(self, qsavefile_mock):
        if False:
            while True:
                i = 10
        'Test with a mock and a failing open().'
        qsavefile_mock.open.return_value = False
        qsavefile_mock.errorString.return_value = 'Hello World'
        with pytest.raises(OSError, match='Hello World'):
            with qtutils.savefile_open('filename'):
                pass
        qsavefile_mock.open.assert_called_once_with(QIODevice.OpenModeFlag.WriteOnly)
        qsavefile_mock.cancelWriting.assert_called_once_with()

    def test_mock_exception(self, qsavefile_mock):
        if False:
            print('Hello World!')
        'Test with a mock and an exception in the block.'
        qsavefile_mock.open.return_value = True
        with pytest.raises(SavefileTestException):
            with qtutils.savefile_open('filename'):
                raise SavefileTestException
        qsavefile_mock.open.assert_called_once_with(QIODevice.OpenModeFlag.WriteOnly)
        qsavefile_mock.cancelWriting.assert_called_once_with()

    def test_mock_commit_failed(self, qsavefile_mock):
        if False:
            i = 10
            return i + 15
        'Test with a mock and an exception in the block.'
        qsavefile_mock.open.return_value = True
        qsavefile_mock.commit.return_value = False
        with pytest.raises(OSError, match='Commit failed!'):
            with qtutils.savefile_open('filename'):
                pass
        qsavefile_mock.open.assert_called_once_with(QIODevice.OpenModeFlag.WriteOnly)
        assert not qsavefile_mock.cancelWriting.called
        assert not qsavefile_mock.errorString.called

    def test_mock_successful(self, qsavefile_mock):
        if False:
            i = 10
            return i + 15
        'Test with a mock and a successful write.'
        qsavefile_mock.open.return_value = True
        qsavefile_mock.errorString.return_value = 'Hello World'
        qsavefile_mock.commit.return_value = True
        qsavefile_mock.write.side_effect = len
        qsavefile_mock.isOpen.return_value = True
        with qtutils.savefile_open('filename') as f:
            f.write('Hello World')
        qsavefile_mock.open.assert_called_once_with(QIODevice.OpenModeFlag.WriteOnly)
        assert not qsavefile_mock.cancelWriting.called
        qsavefile_mock.write.assert_called_once_with(b'Hello World')

    @pytest.mark.parametrize('data', ['Hello World', 'Snowman! ☃'])
    def test_utf8(self, data, tmp_path):
        if False:
            return 10
        'Test with UTF8 data.'
        filename = tmp_path / 'foo'
        filename.write_text('Old data', encoding='utf-8')
        with qtutils.savefile_open(str(filename)) as f:
            f.write(data)
        assert list(tmp_path.iterdir()) == [filename]
        assert filename.read_text(encoding='utf-8') == data

    def test_binary(self, tmp_path):
        if False:
            return 10
        'Test with binary data.'
        filename = tmp_path / 'foo'
        with qtutils.savefile_open(str(filename), binary=True) as f:
            f.write(b'\xde\xad\xbe\xef')
        assert list(tmp_path.iterdir()) == [filename]
        assert filename.read_bytes() == b'\xde\xad\xbe\xef'

    def test_exception(self, tmp_path):
        if False:
            while True:
                i = 10
        'Test with an exception in the block.'
        filename = tmp_path / 'foo'
        filename.write_text('Old content', encoding='utf-8')
        with pytest.raises(SavefileTestException):
            with qtutils.savefile_open(str(filename)) as f:
                f.write('Hello World!')
                raise SavefileTestException
        assert list(tmp_path.iterdir()) == [filename]
        assert filename.read_text(encoding='utf-8') == 'Old content'

    def test_existing_dir(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'Test with the filename already occupied by a directory.'
        filename = tmp_path / 'foo'
        filename.mkdir()
        with pytest.raises(OSError) as excinfo:
            with qtutils.savefile_open(str(filename)):
                pass
        msg = 'Filename refers to a directory: {!r}'.format(str(filename))
        assert str(excinfo.value) == msg
        assert list(tmp_path.iterdir()) == [filename]

    def test_failing_flush(self, tmp_path):
        if False:
            return 10
        'Test with the file being closed before flushing.'
        filename = tmp_path / 'foo'
        with pytest.raises(ValueError, match='IO operation on closed device!'):
            with qtutils.savefile_open(str(filename), binary=True) as f:
                f.write(b'Hello')
                f.dev.commit()
        assert list(tmp_path.iterdir()) == [filename]

    def test_failing_commit(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'Test with the file being closed before committing.'
        filename = tmp_path / 'foo'
        with pytest.raises(OSError, match='Commit failed!'):
            with qtutils.savefile_open(str(filename), binary=True) as f:
                f.write(b'Hello')
                f.dev.cancelWriting()
        assert list(tmp_path.iterdir()) == []

    def test_line_endings(self, tmp_path):
        if False:
            print('Hello World!')
        'Make sure line endings are translated correctly.\n\n        See https://github.com/qutebrowser/qutebrowser/issues/309\n        '
        filename = tmp_path / 'foo'
        with qtutils.savefile_open(str(filename)) as f:
            f.write('foo\nbar\nbaz')
        data = filename.read_bytes()
        if utils.is_windows:
            assert data == b'foo\r\nbar\r\nbaz'
        else:
            assert data == b'foo\nbar\nbaz'
if test_file is not None:

    @pytest.fixture(scope='session', autouse=True)
    def clean_up_python_testfile():
        if False:
            print('Hello World!')
        "Clean up the python testfile after tests if tests didn't."
        yield
        try:
            pathlib.Path(test_file.TESTFN).unlink()
        except FileNotFoundError:
            pass

    class PyIODeviceTestMixin:
        """Some helper code to run Python's tests with PyQIODevice.

        Attributes:
            _data: A QByteArray containing the data in memory.
            f: The opened PyQIODevice.
        """

        def setUp(self):
            if False:
                return 10
            'Set up self.f using a PyQIODevice instead of a real file.'
            self._data = QByteArray()
            self.f = self.open(test_file.TESTFN, 'wb')

        def open(self, _fname, mode):
            if False:
                i = 10
                return i + 15
            'Open an in-memory PyQIODevice instead of a real file.'
            modes = {'wb': QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Truncate, 'w': QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Text | QIODevice.OpenModeFlag.Truncate, 'rb': QIODevice.OpenModeFlag.ReadOnly, 'r': QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text}
            try:
                qt_mode = modes[mode]
            except KeyError:
                raise ValueError('Invalid mode {}!'.format(mode))
            f = QBuffer(self._data)
            f.open(qt_mode)
            qiodev = qtutils.PyQIODevice(f)
            qiodev.name = test_file.TESTFN
            qiodev.mode = mode
            with open(test_file.TESTFN, 'w', encoding='utf-8'):
                pass
            return qiodev

    class PyAutoFileTests(PyIODeviceTestMixin, test_file.AutoFileTests, unittest.TestCase):
        """Unittest testcase to run Python's AutoFileTests."""

        def testReadinto_text(self):
            if False:
                while True:
                    i = 10
            'Skip this test as BufferedIOBase seems to fail it.'

    class PyOtherFileTests(PyIODeviceTestMixin, test_file.OtherFileTests, unittest.TestCase):
        """Unittest testcase to run Python's OtherFileTests."""

        def testSetBufferSize(self):
            if False:
                while True:
                    i = 10
            'Skip this test as setting buffer size is unsupported.'

        def testTruncateOnWindows(self):
            if False:
                for i in range(10):
                    print('nop')
            'Skip this test truncating is unsupported.'

class FailingQIODevice(QIODevice):
    """A fake QIODevice where reads/writes fail."""

    def isOpen(self):
        if False:
            while True:
                i = 10
        return True

    def isReadable(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def isWritable(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def write(self, _data):
        if False:
            i = 10
            return i + 15
        'Simulate failed write.'
        self.setErrorString('Writing failed')
        return -1

    def read(self, _maxsize):
        if False:
            for i in range(10):
                print('nop')
        'Simulate failed read.'
        self.setErrorString('Reading failed')
        return None

    def readAll(self):
        if False:
            return 10
        return self.read(0)

    def readLine(self, maxsize):
        if False:
            i = 10
            return i + 15
        return self.read(maxsize)

class TestPyQIODevice:
    """Tests for PyQIODevice."""

    @pytest.fixture
    def pyqiodev(self):
        if False:
            for i in range(10):
                print('nop')
        'Fixture providing a PyQIODevice with a QByteArray to test.'
        data = QByteArray()
        f = QBuffer(data)
        qiodev = qtutils.PyQIODevice(f)
        yield qiodev
        qiodev.close()

    @pytest.fixture
    def pyqiodev_failing(self):
        if False:
            while True:
                i = 10
        'Fixture providing a PyQIODevice with a FailingQIODevice to test.'
        failing = FailingQIODevice()
        return qtutils.PyQIODevice(failing)

    @pytest.mark.parametrize('method, args', [('seek', [0]), ('flush', []), ('isatty', []), ('readline', []), ('tell', []), ('write', [b'']), ('read', [])])
    def test_closed_device(self, pyqiodev, method, args):
        if False:
            for i in range(10):
                print('nop')
        'Test various methods with a closed device.\n\n        Args:\n            method: The name of the method to call.\n            args: The arguments to pass.\n        '
        func = getattr(pyqiodev, method)
        with pytest.raises(ValueError, match='IO operation on closed device!'):
            func(*args)

    @pytest.mark.parametrize('method', ['readline', 'read'])
    def test_unreadable(self, pyqiodev, method):
        if False:
            for i in range(10):
                print('nop')
        'Test methods with an unreadable device.\n\n        Args:\n            method: The name of the method to call.\n        '
        pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly)
        func = getattr(pyqiodev, method)
        with pytest.raises(OSError, match='Trying to read unreadable file!'):
            func()

    def test_unwritable(self, pyqiodev):
        if False:
            i = 10
            return i + 15
        'Test writing with a read-only device.'
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        with pytest.raises(OSError, match='Trying to write to unwritable file!'):
            pyqiodev.write(b'')

    @pytest.mark.parametrize('data', [b'12345', b''])
    def test_len(self, pyqiodev, data):
        if False:
            while True:
                i = 10
        'Test len()/__len__.\n\n        Args:\n            data: The data to write before checking if the length equals\n                  len(data).\n        '
        pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly)
        pyqiodev.write(data)
        assert len(pyqiodev) == len(data)

    def test_failing_open(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        "Test open() which fails (because it's an existent directory)."
        qf = QFile(str(tmp_path))
        dev = qtutils.PyQIODevice(qf)
        with pytest.raises(qtutils.QtOSError) as excinfo:
            dev.open(QIODevice.OpenModeFlag.WriteOnly)
        assert excinfo.value.qt_errno == QFileDevice.FileError.OpenError
        assert dev.closed

    def test_fileno(self, pyqiodev):
        if False:
            i = 10
            return i + 15
        with pytest.raises(io.UnsupportedOperation):
            pyqiodev.fileno()

    @pytest.mark.qt_log_ignore('^QBuffer::seek: Invalid pos:')
    @pytest.mark.parametrize('offset, whence, pos, data, raising', [(0, io.SEEK_SET, 0, b'1234567890', False), (42, io.SEEK_SET, 0, b'1234567890', True), (8, io.SEEK_CUR, 8, b'90', False), (-5, io.SEEK_CUR, 0, b'1234567890', True), (-2, io.SEEK_END, 8, b'90', False), (2, io.SEEK_END, 0, b'1234567890', True), (0, io.SEEK_END, 10, b'', False)])
    def test_seek_tell(self, pyqiodev, offset, whence, pos, data, raising):
        if False:
            for i in range(10):
                print('nop')
        'Test seek() and tell().\n\n        The initial position when these tests run is 0.\n\n        Args:\n            offset: The offset to pass to .seek().\n            whence: The whence argument to pass to .seek().\n            pos: The expected position after seeking.\n            data: The expected data to read after seeking.\n            raising: Whether seeking should raise OSError.\n        '
        with pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly) as f:
            f.write(b'1234567890')
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        if raising:
            with pytest.raises(OSError, match='seek failed!'):
                pyqiodev.seek(offset, whence)
        else:
            pyqiodev.seek(offset, whence)
        assert pyqiodev.tell() == pos
        assert pyqiodev.read() == data

    def test_seek_unsupported(self, pyqiodev):
        if False:
            print('Hello World!')
        'Test seeking with unsupported whence arguments.'
        if hasattr(os, 'SEEK_HOLE'):
            whence = os.SEEK_HOLE
        elif hasattr(os, 'SEEK_DATA'):
            whence = os.SEEK_DATA
        else:
            pytest.skip('Needs os.SEEK_HOLE or os.SEEK_DATA available.')
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        with pytest.raises(io.UnsupportedOperation):
            pyqiodev.seek(0, whence)

    @pytest.mark.flaky
    def test_qprocess(self, py_proc):
        if False:
            return 10
        'Test PyQIODevice with a QProcess which is non-sequential.\n\n        This also verifies seek() and tell() behave as expected.\n        '
        proc = QProcess()
        proc.start(*py_proc('print("Hello World")'))
        dev = qtutils.PyQIODevice(proc)
        assert not dev.closed
        with pytest.raises(OSError, match='Random access not allowed!'):
            dev.seek(0)
        with pytest.raises(OSError, match='Random access not allowed!'):
            dev.tell()
        proc.waitForFinished(1000)
        proc.kill()
        assert bytes(dev.read()).rstrip() == b'Hello World'

    def test_truncate(self, pyqiodev):
        if False:
            while True:
                i = 10
        with pytest.raises(io.UnsupportedOperation):
            pyqiodev.truncate()

    def test_closed(self, pyqiodev):
        if False:
            while True:
                i = 10
        'Test the closed attribute.'
        assert pyqiodev.closed
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        assert not pyqiodev.closed
        pyqiodev.close()
        assert pyqiodev.closed

    def test_contextmanager(self, pyqiodev):
        if False:
            while True:
                i = 10
        'Make sure using the PyQIODevice as context manager works.'
        assert pyqiodev.closed
        with pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly) as f:
            assert not f.closed
            assert f is pyqiodev
        assert pyqiodev.closed

    def test_flush(self, pyqiodev):
        if False:
            for i in range(10):
                print('nop')
        "Make sure flushing doesn't raise an exception."
        pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly)
        pyqiodev.write(b'test')
        pyqiodev.flush()

    @pytest.mark.parametrize('method, ret', [('isatty', False), ('seekable', True)])
    def test_bools(self, method, ret, pyqiodev):
        if False:
            return 10
        'Make sure simple bool arguments return the right thing.\n\n        Args:\n            method: The name of the method to call.\n            ret: The return value we expect.\n        '
        pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly)
        func = getattr(pyqiodev, method)
        assert func() == ret

    @pytest.mark.parametrize('mode, readable, writable', [(QIODevice.OpenModeFlag.ReadOnly, True, False), (QIODevice.OpenModeFlag.ReadWrite, True, True), (QIODevice.OpenModeFlag.WriteOnly, False, True)])
    def test_readable_writable(self, mode, readable, writable, pyqiodev):
        if False:
            return 10
        'Test readable() and writable().\n\n        Args:\n            mode: The mode to open the PyQIODevice in.\n            readable:  Whether the device should be readable.\n            writable:  Whether the device should be writable.\n        '
        assert not pyqiodev.readable()
        assert not pyqiodev.writable()
        pyqiodev.open(mode)
        assert pyqiodev.readable() == readable
        assert pyqiodev.writable() == writable

    @pytest.mark.parametrize('size, chunks', [(-1, [b'one\n', b'two\n', b'three', b'']), (0, [b'', b'', b'', b'']), (2, [b'on', b'e\n', b'tw', b'o\n', b'th', b're', b'e']), (10, [b'one\n', b'two\n', b'three', b''])])
    def test_readline(self, size, chunks, pyqiodev):
        if False:
            i = 10
            return i + 15
        'Test readline() with different sizes.\n\n        Args:\n            size: The size to pass to readline()\n            chunks: A list of expected chunks to read.\n        '
        with pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly) as f:
            f.write(b'one\ntwo\nthree')
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        for (i, chunk) in enumerate(chunks, start=1):
            print('Expecting chunk {}: {!r}'.format(i, chunk))
            assert pyqiodev.readline(size) == chunk

    def test_write(self, pyqiodev):
        if False:
            return 10
        'Make sure writing and re-reading works.'
        with pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly) as f:
            f.write(b'foo\n')
            f.write(b'bar\n')
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        assert pyqiodev.read() == b'foo\nbar\n'

    def test_write_error(self, pyqiodev_failing):
        if False:
            while True:
                i = 10
        'Test writing with FailingQIODevice.'
        with pytest.raises(OSError, match='Writing failed'):
            pyqiodev_failing.write(b'x')

    @pytest.mark.posix
    @pytest.mark.skipif(not pathlib.Path('/dev/full').exists(), reason='Needs /dev/full.')
    def test_write_error_real(self):
        if False:
            i = 10
            return i + 15
        'Test a real write error with /dev/full on supported systems.'
        qf = QFile('/dev/full')
        qf.open(QIODevice.OpenModeFlag.WriteOnly | QIODevice.OpenModeFlag.Unbuffered)
        dev = qtutils.PyQIODevice(qf)
        with pytest.raises(OSError, match='No space left on device'):
            dev.write(b'foo')
        qf.close()

    @pytest.mark.parametrize('size, chunks', [(-1, [b'1234567890']), (0, [b'']), (3, [b'123', b'456', b'789', b'0']), (20, [b'1234567890'])])
    def test_read(self, size, chunks, pyqiodev):
        if False:
            print('Hello World!')
        'Test reading with different sizes.\n\n        Args:\n            size: The size to pass to read()\n            chunks: A list of expected data chunks.\n        '
        with pyqiodev.open(QIODevice.OpenModeFlag.WriteOnly) as f:
            f.write(b'1234567890')
        pyqiodev.open(QIODevice.OpenModeFlag.ReadOnly)
        for (i, chunk) in enumerate(chunks):
            print('Expecting chunk {}: {!r}'.format(i, chunk))
            assert pyqiodev.read(size) == chunk

    @pytest.mark.parametrize('method, args', [('read', []), ('read', [5]), ('readline', []), ('readline', [5])])
    def test_failing_reads(self, method, args, pyqiodev_failing):
        if False:
            i = 10
            return i + 15
        'Test reading with a FailingQIODevice.\n\n        Args:\n            method: The name of the method to call.\n            args: A list of arguments to pass.\n        '
        func = getattr(pyqiodev_failing, method)
        with pytest.raises(OSError, match='Reading failed'):
            func(*args)

@pytest.mark.usefixtures('qapp')
class TestEventLoop:
    """Tests for EventLoop.

    Attributes:
        loop: The EventLoop we're testing.
    """

    def _assert_executing(self):
        if False:
            print('Hello World!')
        'Slot which gets called from timers to be sure the loop runs.'
        assert self.loop._executing

    def _double_exec(self):
        if False:
            print('Hello World!')
        'Slot which gets called from timers to assert double-exec fails.'
        with pytest.raises(AssertionError):
            self.loop.exec()

    def test_normal_exec(self):
        if False:
            for i in range(10):
                print('nop')
        'Test exec_ without double-executing.'
        self.loop = qtutils.EventLoop()
        QTimer.singleShot(100, self._assert_executing)
        QTimer.singleShot(200, self.loop.quit)
        self.loop.exec()
        assert not self.loop._executing

    def test_double_exec(self):
        if False:
            i = 10
            return i + 15
        'Test double-executing.'
        self.loop = qtutils.EventLoop()
        QTimer.singleShot(100, self._assert_executing)
        QTimer.singleShot(200, self._double_exec)
        QTimer.singleShot(300, self._assert_executing)
        QTimer.singleShot(400, self.loop.quit)
        self.loop.exec()
        assert not self.loop._executing

class TestInterpolateColor:

    @dataclasses.dataclass
    class Colors:
        white: testutils.Color
        black: testutils.Color

    @pytest.fixture
    def colors(self):
        if False:
            for i in range(10):
                print('nop')
        'Example colors to be used.'
        return self.Colors(testutils.Color('white'), testutils.Color('black'))

    def test_invalid_start(self, colors):
        if False:
            for i in range(10):
                print('nop')
        'Test an invalid start color.'
        with pytest.raises(qtutils.QtValueError):
            qtutils.interpolate_color(testutils.Color(), colors.white, 0)

    def test_invalid_end(self, colors):
        if False:
            for i in range(10):
                print('nop')
        'Test an invalid end color.'
        with pytest.raises(qtutils.QtValueError):
            qtutils.interpolate_color(colors.white, testutils.Color(), 0)

    @pytest.mark.parametrize('perc', [-1, 101])
    def test_invalid_percentage(self, colors, perc):
        if False:
            for i in range(10):
                print('nop')
        'Test an invalid percentage.'
        with pytest.raises(ValueError):
            qtutils.interpolate_color(colors.white, colors.white, perc)

    def test_invalid_colorspace(self, colors):
        if False:
            while True:
                i = 10
        'Test an invalid colorspace.'
        with pytest.raises(ValueError):
            qtutils.interpolate_color(colors.white, colors.black, 10, QColor.Spec.Cmyk)

    @pytest.mark.parametrize('colorspace', [QColor.Spec.Rgb, QColor.Spec.Hsv, QColor.Spec.Hsl])
    def test_0_100(self, colors, colorspace):
        if False:
            for i in range(10):
                print('nop')
        'Test 0% and 100% in different colorspaces.'
        white = qtutils.interpolate_color(colors.white, colors.black, 0, colorspace)
        black = qtutils.interpolate_color(colors.white, colors.black, 100, colorspace)
        assert testutils.Color(white) == colors.white
        assert testutils.Color(black) == colors.black

    def test_interpolation_rgb(self):
        if False:
            return 10
        'Test an interpolation in the RGB colorspace.'
        color = qtutils.interpolate_color(testutils.Color(0, 40, 100), testutils.Color(0, 20, 200), 50, QColor.Spec.Rgb)
        assert testutils.Color(color) == testutils.Color(0, 30, 150)

    def test_interpolation_hsv(self):
        if False:
            while True:
                i = 10
        'Test an interpolation in the HSV colorspace.'
        start = testutils.Color()
        stop = testutils.Color()
        start.setHsv(0, 40, 100)
        stop.setHsv(0, 20, 200)
        color = qtutils.interpolate_color(start, stop, 50, QColor.Spec.Hsv)
        expected = testutils.Color()
        expected.setHsv(0, 30, 150)
        assert testutils.Color(color) == expected

    def test_interpolation_hsl(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an interpolation in the HSL colorspace.'
        start = testutils.Color()
        stop = testutils.Color()
        start.setHsl(0, 40, 100)
        stop.setHsl(0, 20, 200)
        color = qtutils.interpolate_color(start, stop, 50, QColor.Spec.Hsl)
        expected = testutils.Color()
        expected.setHsl(0, 30, 150)
        assert testutils.Color(color) == expected

    @pytest.mark.parametrize('colorspace', [QColor.Spec.Rgb, QColor.Spec.Hsv, QColor.Spec.Hsl])
    def test_interpolation_alpha(self, colorspace):
        if False:
            return 10
        "Test interpolation of colorspace's alpha."
        start = testutils.Color(0, 0, 0, 30)
        stop = testutils.Color(0, 0, 0, 100)
        color = qtutils.interpolate_color(start, stop, 50, colorspace)
        expected = testutils.Color(0, 0, 0, 65)
        assert testutils.Color(color) == expected

    @pytest.mark.parametrize('percentage, expected', [(0, (0, 0, 0)), (99, (0, 0, 0)), (100, (255, 255, 255))])
    def test_interpolation_none(self, percentage, expected):
        if False:
            while True:
                i = 10
        'Test an interpolation with a gradient turned off.'
        color = qtutils.interpolate_color(testutils.Color(0, 0, 0), testutils.Color(255, 255, 255), percentage, None)
        assert isinstance(color, QColor)
        assert testutils.Color(color) == testutils.Color(*expected)

class TestLibraryPath:

    def test_simple(self):
        if False:
            return 10
        try:
            path = QLibraryInfo.path(QLibraryInfo.LibraryPath.DataPath)
        except AttributeError:
            path = QLibraryInfo.location(QLibraryInfo.LibraryLocation.DataPath)
        assert path
        assert qtutils.library_path(qtutils.LibraryPath.data).as_posix() == path

    @pytest.mark.parametrize('which', list(qtutils.LibraryPath))
    def test_all(self, which):
        if False:
            return 10
        if utils.is_windows and which == qtutils.LibraryPath.settings:
            pytest.skip('Settings path not supported on Windows')
        qtutils.library_path(which)

    def test_values_match_qt(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            enumtype = QLibraryInfo.LibraryPath
        except AttributeError:
            enumtype = QLibraryInfo.LibraryLocation
        our_names = {member.value for member in qtutils.LibraryPath}
        qt_names = set(testutils.enum_members(QLibraryInfo, enumtype))
        qt_names.discard('ImportsPath')
        assert qt_names == our_names

def test_extract_enum_val():
    if False:
        for i in range(10):
            print('nop')
    value = qtutils.extract_enum_val(Qt.KeyboardModifier.ShiftModifier)
    assert value == 33554432

class TestQObjRepr:

    @pytest.mark.parametrize('obj', [QObject(), object(), None])
    def test_simple(self, obj):
        if False:
            for i in range(10):
                print('nop')
        assert qtutils.qobj_repr(obj) == repr(obj)

    def _py_repr(self, obj):
        if False:
            while True:
                i = 10
        'Get the original repr of an object, with <> stripped off.\n\n        We do this in code instead of recreating it in tests because of output\n        differences between PyQt5/PyQt6 and between operating systems.\n        '
        r = repr(obj)
        if r.startswith('<') and r.endswith('>'):
            return r[1:-1]
        return r

    def test_object_name(self):
        if False:
            i = 10
            return i + 15
        obj = QObject()
        obj.setObjectName('Tux')
        expected = f"<{self._py_repr(obj)}, objectName='Tux'>"
        assert qtutils.qobj_repr(obj) == expected

    def test_class_name(self):
        if False:
            i = 10
            return i + 15
        obj = QTimer()
        hidden = sip.cast(obj, QObject)
        expected = f"<{self._py_repr(hidden)}, className='QTimer'>"
        assert qtutils.qobj_repr(hidden) == expected

    def test_both(self):
        if False:
            for i in range(10):
                print('nop')
        obj = QTimer()
        obj.setObjectName('Pomodoro')
        hidden = sip.cast(obj, QObject)
        expected = f"<{self._py_repr(hidden)}, objectName='Pomodoro', className='QTimer'>"
        assert qtutils.qobj_repr(hidden) == expected

    def test_rich_repr(self):
        if False:
            i = 10
            return i + 15

        class RichRepr(QObject):

            def __repr__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'RichRepr()'
        obj = RichRepr()
        assert repr(obj) == 'RichRepr()'
        expected = "<RichRepr(), className='RichRepr'>"
        assert qtutils.qobj_repr(obj) == expected