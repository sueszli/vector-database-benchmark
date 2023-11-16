import platform
import pytest
is_pypy = platform.python_implementation() == 'PyPy'
if is_pypy:
    from pyboy.plugins.rewind import DeltaFixedAllocBuffers, CompressedFixedAllocBuffers, FixedAllocBuffers, FILL_VALUE, FIXED_BUFFER_SIZE

def write_bytes(buf, values):
    if False:
        i = 10
        return i + 15
    for v in values:
        buf.write(v % 256)

@pytest.mark.skipif(not is_pypy, reason="This test doesn't work in Cython")
class TestRewind:

    def test_all(self):
        if False:
            return 10
        for buf in [FixedAllocBuffers(), CompressedFixedAllocBuffers(), DeltaFixedAllocBuffers()]:
            A = [1] * 16
            B = [2] * 16
            C = [4] * 16
            D = [8] * 16
            for E in [A, B, C, D]:
                write_bytes(buf, E)
                buf.new()
            for E in [D, C, B, A]:
                assert buf.seek_frame(-1)
                tests = [(x, buf.read()) for x in E]
                assert all(list(map(lambda x: x[0] == x[1], tests)))
            order = [A, B, C, D]
            if isinstance(buf, DeltaFixedAllocBuffers):
                order.pop(0)
            for E in order:
                assert buf.seek_frame(1)
                tests = [(x, buf.read()) for x in E]
                assert all(list(map(lambda x: x[0] == x[1], tests)))

    def test_delta_seek(self):
        if False:
            return 10
        buf = DeltaFixedAllocBuffers()
        A = [1] * 16
        B = [2] * 16
        C = [3] * 16
        write_bytes(buf, A)
        buf.new()
        write_bytes(buf, B)
        buf.new()
        write_bytes(buf, C)
        buf.new()
        assert buf.seek_frame(-1)
        tests = [(x, buf.read()) for x in C]
        assert all(list(map(lambda x: x[0] == x[1], tests)))
        assert buf.seek_frame(-1)
        tests = [(x, buf.read()) for x in B]
        assert all(list(map(lambda x: x[0] == x[1], tests)))
        assert buf.seek_frame(-1)
        tests = [(x, buf.read()) for x in A]
        assert all(list(map(lambda x: x[0] == x[1], tests)))
        assert not buf.seek_frame(-1)
        assert buf.seek_frame(1)
        tests = [(x, buf.read()) for x in B]
        assert all(list(map(lambda x: x[0] == x[1], tests)))
        assert buf.seek_frame(1)
        tests = [(x, buf.read()) for x in C]
        assert all(list(map(lambda x: x[0] == x[1], tests)))
        assert not buf.seek_frame(1)

    def test_compressed_buffer(self):
        if False:
            print('Hello World!')
        buf = CompressedFixedAllocBuffers()
        write_bytes(buf, [0 for _ in range(10)])
        assert all(map(lambda x: x == FILL_VALUE, buf.buffer[:12]))
        buf.flush()
        assert all(map(lambda x: x == FILL_VALUE, buf.buffer[2:12]))
        assert buf.buffer[0] == 0
        assert buf.buffer[1] == 10
        buf.flush()
        assert all(map(lambda x: x == FILL_VALUE, buf.buffer[2:12]))
        assert buf.buffer[0] == 0
        assert buf.buffer[1] == 10
        write_bytes(buf, [0 for _ in range(256)])
        buf.flush()
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[2:8], [0, 255, 0, 1] + [FILL_VALUE] * 2)))
        write_bytes(buf, [0 for _ in range(255)])
        buf.flush()
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[6:10], [0, 255] + [FILL_VALUE] * 4)))

    def test_delta_buffer(self):
        if False:
            return 10
        buf = DeltaFixedAllocBuffers()
        assert all(map(lambda x: x == FILL_VALUE, buf.buffer[:60]))
        assert all(map(lambda x: x == 0, buf.internal_buffer[:60]))
        write_bytes(buf, range(20))
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:60], [0, 1] + list(range(1, 20)) + [FILL_VALUE] * 40)))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], list(range(20)) + [0] * 40)))
        buf.new()
        write_bytes(buf, range(128, 128 + 20))
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:60], [0, 1] + list(range(1, 20)) + [128] * 20 + [FILL_VALUE] * 20)))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], list(range(128, 128 + 20)) + [0] * 40)))
        buf.new()
        write_bytes(buf, [255] * 20)
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:61], [0, 1] + list(range(1, 20)) + [128] * 20 + [x ^ 255 for x in list(range(128, 128 + 20))])))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], [255] * 20 + [0] * 40)))
        buf.new()

    def test_delta_buffer_repeat_pattern(self):
        if False:
            i = 10
            return i + 15
        buf = DeltaFixedAllocBuffers()
        assert all(map(lambda x: x == FILL_VALUE, buf.buffer[:60]))
        assert all(map(lambda x: x == 0, buf.internal_buffer[:60]))
        write_bytes(buf, [170] * 20)
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:60], [170] * 20 + [FILL_VALUE] * 40)))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], [170] * 20 + [0] * 40)))
        buf.new()
        write_bytes(buf, [170] * 20)
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:60], [170] * 20 + [FILL_VALUE] * 40)))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], [170] * 20 + [0] * 40)))
        buf.new()
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:60], [170] * 20 + [0, 20] + [FILL_VALUE] * 38)))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], [170] * 20 + [0] * 40)))
        write_bytes(buf, [170] * 20)
        buf.new()
        assert all(map(lambda x: x[0] == x[1], zip(buf.buffer[:60], [170] * 20 + [0, 20, 0, 20] + [FILL_VALUE] * 36)))
        assert all(map(lambda x: x[0] == x[1], zip(buf.internal_buffer[:60], [170] * 20 + [0] * 40)))

    def test_buffer_overrun(self):
        if False:
            i = 10
            return i + 15
        buf = FixedAllocBuffers()
        write_bytes(buf, [170] * (FIXED_BUFFER_SIZE - 10))
        buf.new()
        assert len(buf.sections) == 2
        write_bytes(buf, [170] * 20)
        assert len(buf.sections) == 1
        buf.new()
        assert len(buf.sections) == 2