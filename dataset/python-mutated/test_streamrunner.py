import asyncio
import errno
import sys
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Callable, Deque, List, Union
from unittest.mock import Mock, patch
import pytest
from streamlink.stream.stream import StreamIO
from streamlink_cli.output import FileOutput, HTTPOutput, PlayerOutput
from streamlink_cli.streamrunner import PlayerPollThread, StreamRunner, log as streamrunnerlogger
from streamlink_cli.utils.progress import Progress
from tests.testutils.handshake import Handshake
TIMEOUT_AWAIT_HANDSHAKE = 1
TIMEOUT_AWAIT_THREADJOIN = 1

class EventedPlayerPollThread(PlayerPollThread):
    POLLING_INTERVAL = 0

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.handshake = Handshake()

    def poll(self):
        if False:
            i = 10
            return i + 15
        with self.handshake():
            return super().poll()

    def close(self):
        if False:
            i = 10
            return i + 15
        super().close()
        self.handshake.go()

class FakeStream(StreamIO):
    """Fake stream implementation, for feeding sample data to the stream runner and simulating read pauses and read errors"""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.handshake = Handshake()
        self.data: Deque[Union[bytes, Callable]] = deque()

    def read(self, *args):
        if False:
            for i in range(10):
                print('nop')
        with self.handshake():
            if not self.data:
                return b''
            data = self.data.popleft()
            return data() if callable(data) else data

class FakeOutput:
    """Common output/http-server/progress interface, for caching all write() calls and simulating write errors"""

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.handshake = Handshake()
        self.data: List[bytes] = []

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        with self.handshake():
            return self._write(data)

    def _write(self, data):
        if False:
            i = 10
            return i + 15
        self.data.append(data)

class FakePlayerOutput(FakeOutput, PlayerOutput):

    def open(self):
        if False:
            return 10
        with patch('streamlink_cli.output.player.which', side_effect=lambda path: path):
            return super().open()

class FakeFileOutput(FakeOutput, FileOutput):
    pass

class FakeHTTPOutput(FakeOutput, HTTPOutput):
    pass

class FakeProgress(FakeOutput, Progress):
    update = print_end = lambda *_, **__: None

class FakeStreamRunner(StreamRunner):
    playerpoller: EventedPlayerPollThread
    progress: FakeProgress

@pytest.fixture(autouse=True)
def _logging(caplog: pytest.LogCaptureFixture):
    if False:
        while True:
            i = 10
    assert streamrunnerlogger.name == 'streamlink.cli'
    caplog.set_level(1, 'streamlink')

@pytest.fixture()
def stream():
    if False:
        return 10
    stream = FakeStream()
    yield stream
    assert stream.closed

@pytest.fixture()
def runnerthread(request: pytest.FixtureRequest, stream_runner: StreamRunner):
    if False:
        i = 10
        return i + 15

    class RunnerThread(Thread):
        exception = None

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            try:
                super().run()
            except BaseException as err:
                self.exception = err
    thread = RunnerThread(daemon=True, name='Runner thread', target=stream_runner.run, args=(b'prebuffer',))
    yield thread
    assert_thread_termination(thread, 'Runner thread has terminated')
    exception = getattr(request, 'param', {}).get('exception', None)
    assert isinstance(thread.exception, type(exception))
    assert str(thread.exception) == str(exception)

async def assert_handshake_steps(*items):
    """
    Run handshake steps concurrently, to not be dependent too much on implementation details and the order of handshakes.
    For example, concurrently await one read(), one write() and one progress() call.
    """
    steps = asyncio.gather(*(item.handshake.asyncstep(TIMEOUT_AWAIT_HANDSHAKE) for item in items), return_exceptions=True)
    assert await steps == [True for _ in items]

def assert_thread_termination(thread: Thread, assertion: str):
    if False:
        return 10
    thread.join(TIMEOUT_AWAIT_THREADJOIN)
    assert not thread.is_alive(), assertion

class TestPlayerOutput:

    @pytest.fixture()
    def player_process(self):
        if False:
            return 10
        player_process = Mock()
        player_process.poll = Mock(return_value=None)
        return player_process

    @pytest.fixture()
    def output(self, player_process: Mock):
        if False:
            while True:
                i = 10
        with patch('subprocess.Popen') as mock_popen, patch('streamlink_cli.output.player.sleep'):
            mock_popen.return_value = player_process
            output = FakePlayerOutput(Path('mocked'))
            output.open()
            yield output
            output.close()

    @pytest.fixture()
    def stream_runner(self, stream: FakeStream, output: FakePlayerOutput):
        if False:
            while True:
                i = 10
        with patch('streamlink_cli.streamrunner.PlayerPollThread', EventedPlayerPollThread):
            stream_runner = StreamRunner(stream, output)
            assert isinstance(stream_runner.playerpoller, EventedPlayerPollThread)
            assert not stream_runner.playerpoller.is_alive()
            assert not stream_runner.is_http
            assert not stream_runner.progress
            yield stream_runner
            assert not stream_runner.playerpoller.is_alive()

    @pytest.mark.asyncio()
    async def test_read_write(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakePlayerOutput):
        stream.data.extend((b'foo', b'bar'))
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer']
        await assert_handshake_steps(stream, output)
        assert output.data == [b'prebuffer', b'foo']
        await assert_handshake_steps(stream_runner.playerpoller)
        assert stream_runner.playerpoller.is_alive()
        await assert_handshake_steps(stream, output)
        assert output.data == [b'prebuffer', b'foo', b'bar']
        assert not stream.closed, 'Stream is not closed'
        await assert_handshake_steps(stream)
        assert output.data == [b'prebuffer', b'foo', b'bar']
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Stream ended')]

    @pytest.mark.asyncio()
    async def test_paused(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakePlayerOutput):
        delayed = Handshake()

        def item():
            if False:
                i = 10
                return i + 15
            with delayed():
                return b'delayed'
        stream.data.append(item)
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer']
        assert not delayed.wait_ready(0), 'Delayed chunk has not been read yet'
        stream.handshake.go()
        assert delayed.wait_ready(TIMEOUT_AWAIT_HANDSHAKE), 'read() call of delayed chunk is paused'
        assert output.data == [b'prebuffer']
        assert not stream.closed, 'Stream is not closed'
        await assert_handshake_steps(stream_runner.playerpoller)
        assert stream_runner.playerpoller.is_alive()
        delayed.go()
        assert stream.handshake.wait_done(TIMEOUT_AWAIT_HANDSHAKE), 'Delayed chunk has successfully been read'
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer', b'delayed']
        assert not stream.closed, 'Stream is not closed'
        await assert_handshake_steps(stream)
        assert output.data == [b'prebuffer', b'delayed']
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Stream ended')]

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(('writeerror', 'runnerthread'), [pytest.param(OSError(errno.EPIPE, 'Broken pipe'), {}, id='Acceptable error: EPIPE'), pytest.param(OSError(errno.EINVAL, 'Invalid argument'), {}, id='Acceptable error: EINVAL'), pytest.param(OSError(errno.ECONNRESET, 'Connection reset'), {}, id='Acceptable error: ECONNRESET'), pytest.param(OSError('Unknown error'), {'exception': OSError('Error when writing to output: Unknown error, exiting')}, id='Non-acceptable error')], indirect=['runnerthread'])
    async def test_player_close(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakePlayerOutput, player_process: Mock, writeerror: Exception):
        stream.data.extend((b'foo', b'bar'))
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer']
        await assert_handshake_steps(stream_runner.playerpoller)
        assert stream_runner.playerpoller.is_alive()
        await assert_handshake_steps(stream, output)
        assert output.data == [b'prebuffer', b'foo']
        assert not stream.closed, 'Stream is not closed yet'
        with patch.object(output, '_write', side_effect=writeerror):
            player_process.poll.return_value = 0
            await assert_handshake_steps(stream_runner.playerpoller)
            assert_thread_termination(stream_runner.playerpoller, 'Polling has stopped after player process terminated')
            assert stream.closed, 'Stream got closed after the player was closed'
            await assert_handshake_steps(stream, output)
            assert output.data == [b'prebuffer', b'foo']
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Player closed'), ('streamrunner', 'info', 'Stream ended')]

    @pytest.mark.asyncio()
    async def test_player_close_paused(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakePlayerOutput, player_process: Mock):
        delayed = Handshake()

        def item():
            if False:
                for i in range(10):
                    print('nop')
            with delayed():
                return b''
        stream.data.append(item)
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer']
        assert not delayed.wait_ready(0), 'Delayed chunk has not been read yet'
        await assert_handshake_steps(stream_runner.playerpoller)
        assert stream_runner.playerpoller.is_alive()
        stream.handshake.go()
        assert delayed.wait_ready(TIMEOUT_AWAIT_HANDSHAKE), 'read() call of delayed chunk is paused'
        assert output.data == [b'prebuffer']
        assert not stream.closed, 'Stream is not closed yet'
        player_process.poll.return_value = 0
        await assert_handshake_steps(stream_runner.playerpoller)
        assert_thread_termination(stream_runner.playerpoller, 'Polling has stopped after player process terminated')
        assert stream.closed, 'Stream got closed after the player was closed, even if the stream was paused'
        delayed.go()
        assert stream.handshake.wait_done(TIMEOUT_AWAIT_HANDSHAKE), 'Delayed chunk has successfully been read'
        assert output.data == [b'prebuffer']
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Player closed'), ('streamrunner', 'info', 'Stream ended')]

    @pytest.mark.asyncio()
    @pytest.mark.parametrize('runnerthread', [{'exception': OSError('Error when reading from stream: Read timeout, exiting')}], indirect=['runnerthread'])
    async def test_readerror(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakePlayerOutput):
        stream.data.append(Mock(side_effect=OSError('Read timeout')))
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer']
        await assert_handshake_steps(stream_runner.playerpoller)
        assert stream_runner.playerpoller.is_alive()
        await assert_handshake_steps(stream)
        await assert_handshake_steps(stream_runner.playerpoller)
        assert_thread_termination(stream_runner.playerpoller, 'Polling has stopped on read error')
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Stream ended')]

class TestHTTPServer:

    @pytest.fixture()
    def output(self):
        if False:
            print('Hello World!')
        return FakeHTTPOutput()

    @pytest.fixture()
    def stream_runner(self, stream: FakeStream, output: FakeHTTPOutput):
        if False:
            return 10
        stream_runner = StreamRunner(stream, output)
        assert not stream_runner.playerpoller
        assert not stream_runner.progress
        assert stream_runner.is_http
        return stream_runner

    @pytest.mark.asyncio()
    async def test_read_write(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakeHTTPOutput):
        stream.data.extend((b'foo', b'bar'))
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output)
        assert output.data == [b'prebuffer']
        await assert_handshake_steps(stream, output)
        assert output.data == [b'prebuffer', b'foo']
        await assert_handshake_steps(stream, output)
        assert output.data == [b'prebuffer', b'foo', b'bar']
        assert not stream.closed, 'Stream is not closed'
        await assert_handshake_steps(stream)
        assert output.data == [b'prebuffer', b'foo', b'bar']
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Stream ended')]

    @pytest.mark.parametrize(('writeerror', 'logs', 'runnerthread'), [pytest.param(OSError(errno.EPIPE, 'Broken pipe'), True, {}, id='Acceptable error: EPIPE'), pytest.param(OSError(errno.EINVAL, 'Invalid argument'), True, {}, id='Acceptable error: EINVAL'), pytest.param(OSError(errno.ECONNRESET, 'Connection reset'), True, {}, id='Acceptable error: ECONNRESET'), pytest.param(OSError('Unknown error'), False, {'exception': OSError('Error when writing to output: Unknown error, exiting')}, id='Non-acceptable error')], indirect=['runnerthread'])
    def test_writeerror(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakePlayerOutput, logs: bool, writeerror: Exception):
        if False:
            i = 10
            return i + 15
        runnerthread.start()
        with patch.object(output, '_write', side_effect=writeerror):
            assert output.handshake.step(TIMEOUT_AWAIT_HANDSHAKE)
            assert output.data == []
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        expectedlogs = ([('streamrunner', 'info', 'HTTP connection closed')] if logs else []) + [('streamrunner', 'info', 'Stream ended')]
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == expectedlogs

class TestHasProgress:

    @pytest.mark.parametrize('output', [pytest.param(FakePlayerOutput(Path('mocked')), id='Player output without record'), pytest.param(FakeFileOutput(fd=Mock()), id='FileOutput with file descriptor'), pytest.param(FakeHTTPOutput(), id='HTTPServer')])
    def test_no_progress(self, output: Union[FakePlayerOutput, FakeFileOutput, FakeHTTPOutput]):
        if False:
            return 10
        stream_runner = FakeStreamRunner(StreamIO(), output, show_progress=True)
        assert not stream_runner.progress

    @pytest.mark.parametrize(('output', 'expected'), [pytest.param(FakePlayerOutput(Path('mocked'), record=FakeFileOutput(Path('record'))), Path('record'), id='PlayerOutput with record'), pytest.param(FakeFileOutput(filename=Path('filename')), Path('filename'), id='FileOutput with file name'), pytest.param(FakeFileOutput(record=FakeFileOutput(filename=Path('record'))), Path('record'), id='FileOutput with record'), pytest.param(FakeFileOutput(filename=Path('filename'), record=FakeFileOutput(filename=Path('record'))), Path('filename'), id='FileOutput with file name and record')])
    def test_has_progress(self, output: Union[FakePlayerOutput, FakeFileOutput], expected: Path):
        if False:
            return 10
        stream_runner = FakeStreamRunner(StreamIO(), output, show_progress=True)
        assert stream_runner.progress
        assert not stream_runner.progress.is_alive()
        assert stream_runner.progress.stream is sys.stderr
        assert stream_runner.progress.path == expected

class TestProgress:

    @pytest.fixture()
    def output(self):
        if False:
            for i in range(10):
                print('nop')
        return FakeFileOutput(Path('filename'))

    @pytest.fixture()
    def stream_runner(self, stream: FakeStream, output: FakeFileOutput):
        if False:
            while True:
                i = 10
        with patch('streamlink_cli.streamrunner.Progress', FakeProgress):
            stream_runner = FakeStreamRunner(stream, output, show_progress=True)
            assert not stream_runner.playerpoller
            assert not stream_runner.is_http
            assert isinstance(stream_runner.progress, FakeProgress)
            assert stream_runner.progress.path == Path('filename')
            assert not stream_runner.progress.is_alive()
            yield stream_runner
            assert not stream_runner.progress.is_alive()

    @pytest.mark.asyncio()
    async def test_read_write(self, caplog: pytest.LogCaptureFixture, runnerthread: Thread, stream_runner: FakeStreamRunner, stream: FakeStream, output: FakeFileOutput):
        stream.data.extend((b'foo', b'bar'))
        runnerthread.start()
        assert output.data == []
        await assert_handshake_steps(output, stream_runner.progress)
        assert output.data == [b'prebuffer']
        assert stream_runner.progress.data == [b'prebuffer']
        await assert_handshake_steps(stream, output, stream_runner.progress)
        assert output.data == [b'prebuffer', b'foo']
        assert stream_runner.progress.data == [b'prebuffer', b'foo']
        await assert_handshake_steps(stream, output, stream_runner.progress)
        assert output.data == [b'prebuffer', b'foo', b'bar']
        assert stream_runner.progress.data == [b'prebuffer', b'foo', b'bar']
        assert not stream.closed, 'Stream is not closed'
        await assert_handshake_steps(stream)
        assert output.data == [b'prebuffer', b'foo', b'bar']
        assert stream_runner.progress.data == [b'prebuffer', b'foo', b'bar']
        assert_thread_termination(runnerthread, 'Runner thread has terminated')
        assert [(record.module, record.levelname, record.message) for record in caplog.records] == [('streamrunner', 'info', 'Stream ended')]