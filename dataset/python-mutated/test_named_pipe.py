import threading
from unittest.mock import Mock, call
import pytest
from streamlink.utils.named_pipe import NamedPipe, NamedPipeBase, NamedPipePosix, NamedPipeWindows
try:
    from ctypes import byref, c_ulong, create_string_buffer, windll
except ImportError:
    pass
GENERIC_READ = 2147483648
OPEN_EXISTING = 3

class ReadNamedPipeThread(threading.Thread):

    def __init__(self, pipe: NamedPipeBase):
        if False:
            while True:
                i = 10
        super().__init__(daemon=True)
        self.path = str(pipe.path)
        self.error = None
        self.data = b''
        self.done = threading.Event()

    def run(self):
        if False:
            print('Hello World!')
        try:
            self.read()
        except OSError as err:
            self.error = err
        self.done.set()

    def read(self):
        if False:
            return 10
        raise NotImplementedError

class ReadNamedPipeThreadPosix(ReadNamedPipeThread):

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.path, 'rb') as file:
            while True:
                data = file.read(-1)
                if len(data) == 0:
                    break
                self.data += data

class ReadNamedPipeThreadWindows(ReadNamedPipeThread):

    def read(self):
        if False:
            i = 10
            return i + 15
        handle = windll.kernel32.CreateFileW(self.path, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
        try:
            while True:
                data = create_string_buffer(NamedPipeWindows.bufsize)
                read = c_ulong(0)
                if not windll.kernel32.ReadFile(handle, data, NamedPipeWindows.bufsize, byref(read), None):
                    raise OSError(f'Failed reading pipe: {windll.kernel32.GetLastError()}')
                self.data += data.value
                if read.value != len(data.value):
                    break
        finally:
            windll.kernel32.CloseHandle(handle)

class TestNamedPipe:

    def test_name(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        if False:
            for i in range(10):
                print('nop')
        caplog.set_level(1, 'streamlink')
        monkeypatch.setattr('streamlink.utils.named_pipe._id', 0)
        monkeypatch.setattr('streamlink.utils.named_pipe.os.getpid', Mock(return_value=12345))
        monkeypatch.setattr('streamlink.utils.named_pipe.random.randint', Mock(return_value=67890))
        monkeypatch.setattr('streamlink.utils.named_pipe.NamedPipe._create', Mock(return_value=None))
        NamedPipe()
        NamedPipe()
        assert [(record.name, record.levelname, record.message) for record in caplog.records] == [('streamlink.utils.named_pipe', 'info', 'Creating pipe streamlinkpipe-12345-1-67890'), ('streamlink.utils.named_pipe', 'info', 'Creating pipe streamlinkpipe-12345-2-67890')]

@pytest.mark.posix_only()
class TestNamedPipePosix:

    def test_export(self):
        if False:
            for i in range(10):
                print('nop')
        assert NamedPipe is NamedPipePosix

    def test_create(self, monkeypatch: pytest.MonkeyPatch):
        if False:
            for i in range(10):
                print('nop')
        mock_mkfifo = Mock(side_effect=OSError)
        monkeypatch.setattr('streamlink.utils.named_pipe.os.mkfifo', mock_mkfifo)
        with pytest.raises(OSError):
            NamedPipePosix()
        assert mock_mkfifo.call_args[0][1:] == (432,)

    def test_close_before_open(self):
        if False:
            i = 10
            return i + 15
        pipe = NamedPipePosix()
        assert pipe.path.is_fifo()
        pipe.close()
        assert not pipe.path.is_fifo()
        pipe.close()

    def test_close_error(self, monkeypatch: pytest.MonkeyPatch):
        if False:
            print('Hello World!')
        mock_fd_close = Mock(side_effect=OSError)
        mock_fd = Mock(close=mock_fd_close)
        monkeypatch.setattr('builtins.open', Mock(return_value=mock_fd))
        pipe = NamedPipePosix()
        assert pipe.path.is_fifo()
        pipe.open()
        assert mock_fd_close.call_args_list == []
        with pytest.raises(OSError):
            pipe.close()
        assert mock_fd_close.call_args_list == [call()]
        assert not pipe.path.is_fifo()

    def test_write_before_open(self):
        if False:
            i = 10
            return i + 15
        pipe = NamedPipePosix()
        assert pipe.path.is_fifo()
        with pytest.raises(AttributeError):
            pipe.write(b'foo')
        pipe.close()

    def test_named_pipe(self):
        if False:
            return 10
        pipe = NamedPipePosix()
        assert pipe.path.is_fifo()
        reader = ReadNamedPipeThreadPosix(pipe)
        reader.start()
        pipe.open()
        assert pipe.write(b'foo') == 3
        assert pipe.write(b'bar') == 3
        pipe.close()
        assert not pipe.path.is_fifo()
        reader.done.wait(4000)
        assert reader.error is None
        assert reader.data == b'foobar'
        assert not reader.is_alive()

@pytest.mark.windows_only()
class TestNamedPipeWindows:

    def test_export(self):
        if False:
            for i in range(10):
                print('nop')
        assert NamedPipe is NamedPipeWindows

    def test_create(self, monkeypatch: pytest.MonkeyPatch):
        if False:
            i = 10
            return i + 15
        mock_kernel32 = Mock()
        mock_kernel32.CreateNamedPipeW.return_value = NamedPipeWindows.INVALID_HANDLE_VALUE
        mock_kernel32.GetLastError.return_value = 12345
        monkeypatch.setattr('streamlink.utils.named_pipe.windll.kernel32', mock_kernel32)
        with pytest.raises(OSError, match='^Named pipe error code 0x00003039$'):
            NamedPipeWindows()
        assert mock_kernel32.CreateNamedPipeW.call_args[0][1:] == (2, 0, 255, 8192, 8192, 0, None)

    def test_close_before_open(self):
        if False:
            print('Hello World!')
        pipe = NamedPipeWindows()
        handle = windll.kernel32.CreateFileW(str(pipe.path), GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
        assert handle != NamedPipeWindows.INVALID_HANDLE_VALUE
        windll.kernel32.CloseHandle(handle)
        pipe.close()
        handle = windll.kernel32.CreateFileW(str(pipe.path), GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
        assert handle == NamedPipeWindows.INVALID_HANDLE_VALUE
        pipe.close()

    @pytest.mark.parametrize('method', ['DisconnectNamedPipe', 'CloseHandle'])
    def test_close_error(self, monkeypatch: pytest.MonkeyPatch, method: str):
        if False:
            return 10
        mock_method = Mock(side_effect=OSError)
        mock_kernel32 = Mock(**{method: mock_method})
        monkeypatch.setattr('streamlink.utils.named_pipe.windll.kernel32', mock_kernel32)
        pipe = NamedPipeWindows()
        mock_pipe = pipe.pipe
        assert mock_pipe is not None
        pipe.open()
        assert mock_method.call_args_list == []
        with pytest.raises(OSError):
            pipe.close()
        assert mock_method.call_args_list == [call(mock_pipe)]
        assert pipe.pipe is None

    def test_named_pipe(self):
        if False:
            for i in range(10):
                print('nop')
        pipe = NamedPipeWindows()
        reader = ReadNamedPipeThreadWindows(pipe)
        reader.start()
        pipe.open()
        assert pipe.write(b'foo') == 3
        assert pipe.write(b'bar') == 3
        assert pipe.write(b'\x00') == 1
        reader.done.wait(4000)
        assert reader.error is None
        assert reader.data == b'foobar'
        assert not reader.is_alive()
        pipe.close()