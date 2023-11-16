"""Tests for acme.fields."""
import datetime
import sys
import unittest
import warnings
import josepy as jose
import pytest
import pytz

class FixedTest(unittest.TestCase):
    """Tests for acme.fields.Fixed."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        from acme.fields import fixed
        self.field = fixed('name', 'x')

    def test_decode(self):
        if False:
            print('Hello World!')
        assert 'x' == self.field.decode('x')

    def test_decode_bad(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(jose.DeserializationError):
            self.field.decode('y')

    def test_encode(self):
        if False:
            return 10
        assert 'x' == self.field.encode('x')

    def test_encode_override(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'y' == self.field.encode('y')

class RFC3339FieldTest(unittest.TestCase):
    """Tests for acme.fields.RFC3339Field."""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.decoded = datetime.datetime(2015, 3, 27, tzinfo=pytz.UTC)
        self.encoded = '2015-03-27T00:00:00Z'

    def test_default_encoder(self):
        if False:
            while True:
                i = 10
        from acme.fields import RFC3339Field
        assert self.encoded == RFC3339Field.default_encoder(self.decoded)

    def test_default_encoder_naive_fails(self):
        if False:
            return 10
        from acme.fields import RFC3339Field
        with pytest.raises(ValueError):
            RFC3339Field.default_encoder(datetime.datetime.now())

    def test_default_decoder(self):
        if False:
            while True:
                i = 10
        from acme.fields import RFC3339Field
        assert self.decoded == RFC3339Field.default_decoder(self.encoded)

    def test_default_decoder_raises_deserialization_error(self):
        if False:
            while True:
                i = 10
        from acme.fields import RFC3339Field
        with pytest.raises(jose.DeserializationError):
            RFC3339Field.default_decoder('')
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))