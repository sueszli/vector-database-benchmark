"""
Test for unbuffered stdio (stdout/stderr) mode.
"""
import os
import asyncio
import pytest
from PyInstaller.compat import is_win

@pytest.mark.skipif(os.environ.get('CI', 'false').lower() == 'true', reason='The test does not support CI (pytest-xdist sometimes runs it in secondary thread).')
@pytest.mark.parametrize('stream_mode', ['binary', 'text'])
@pytest.mark.parametrize('output_stream', ['stdout', 'stderr'])
def test_unbuffered_stdio(tmp_path, output_stream, stream_mode, pyi_builder_spec):
    if False:
        print('Hello World!')
    pyi_builder_spec.test_spec('pyi_unbuffered_output.spec', app_args=['--num-stars', '0'])
    executable = os.path.join(tmp_path, 'dist', 'pyi_unbuffered_output', 'pyi_unbuffered_output')
    EXPECTED_STARS = 5

    class SubprocessDotCounter(asyncio.SubprocessProtocol):

        def __init__(self, loop, output='stdout'):
            if False:
                for i in range(10):
                    print('nop')
            self.count = 0
            self.loop = loop
            assert output in {'stdout', 'stderr'}
            self.out_fd = 1 if output == 'stdout' else 2

        def pipe_data_received(self, fd, data):
            if False:
                i = 10
                return i + 15
            if fd == self.out_fd:
                if not data.endswith(b'*'):
                    return
                self.count += data.count(b'*')

        def connection_lost(self, exc):
            if False:
                print('Hello World!')
            self.loop.stop()
    if is_win:
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    counter_proto = SubprocessDotCounter(loop, output=output_stream)
    try:
        proc = loop.subprocess_exec(lambda : counter_proto, executable, '--num-stars', str(EXPECTED_STARS), '--output-stream', output_stream, '--stream-mode', stream_mode)
        (transport, _) = loop.run_until_complete(proc)
        loop.run_forever()
    finally:
        loop.close()
        transport.close()
    assert counter_proto.count == EXPECTED_STARS