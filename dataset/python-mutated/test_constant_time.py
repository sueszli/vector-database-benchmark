import pytest
from cryptography.hazmat.primitives import constant_time

class TestConstantTimeBytesEq:

    def test_reject_unicode(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            constant_time.bytes_eq(b'foo', 'foo')
        with pytest.raises(TypeError):
            constant_time.bytes_eq('foo', b'foo')
        with pytest.raises(TypeError):
            constant_time.bytes_eq('foo', 'foo')

    def test_compares(self):
        if False:
            for i in range(10):
                print('nop')
        assert constant_time.bytes_eq(b'foo', b'foo') is True
        assert constant_time.bytes_eq(b'foo', b'bar') is False
        assert constant_time.bytes_eq(b'foobar', b'foo') is False
        assert constant_time.bytes_eq(b'foo', b'foobar') is False