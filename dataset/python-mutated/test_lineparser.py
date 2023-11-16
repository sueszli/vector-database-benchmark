"""Tests for qutebrowser.misc.lineparser."""
import pathlib
from unittest import mock
import pytest
from qutebrowser.misc import lineparser as lineparsermod

class TestBaseLineParser:
    CONFDIR = "this really doesn't matter"
    FILENAME = 'and neither does this'

    @pytest.fixture
    def lineparser(self):
        if False:
            for i in range(10):
                print('nop')
        'Fixture providing a BaseLineParser.'
        return lineparsermod.BaseLineParser(self.CONFDIR, self.FILENAME)

    def test_prepare_save_missing(self, mocker, lineparser):
        if False:
            print('Hello World!')
        "Test if _prepare_save does what it's supposed to do."
        os_mock = mocker.patch('qutebrowser.misc.lineparser.os')
        lineparser._prepare_save()
        os_mock.makedirs.assert_called_with(self.CONFDIR, 493, exist_ok=True)

    def test_double_open(self, mocker, lineparser):
        if False:
            print('Hello World!')
        'Test if _open refuses reentry.'
        mocker.patch('builtins.open', mock.mock_open())
        with lineparser._open('r'):
            with pytest.raises(OSError, match='Refusing to double-open LineParser.'):
                with lineparser._open('r'):
                    pass

    def test_binary(self, mocker):
        if False:
            while True:
                i = 10
        'Test if _open and _write correctly handle binary files.'
        open_mock = mock.mock_open()
        mocker.patch('builtins.open', open_mock)
        testdata = b'\xf0\xff'
        lineparser = lineparsermod.BaseLineParser(self.CONFDIR, self.FILENAME, binary=True)
        with lineparser._open('r') as f:
            lineparser._write(f, [testdata])
        open_mock.assert_called_once_with(str(pathlib.Path(self.CONFDIR) / self.FILENAME), 'rb')
        open_mock().write.assert_has_calls([mock.call(testdata), mock.call(b'\n')])

class TestLineParser:

    @pytest.fixture
    def lineparser(self, tmp_path):
        if False:
            while True:
                i = 10
        'Fixture to get a LineParser for tests.'
        lp = lineparsermod.LineParser(str(tmp_path), 'file')
        lp.save()
        return lp

    def test_init(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        'Test if creating a line parser correctly reads its file.'
        (tmp_path / 'file').write_text('one\ntwo\n')
        lineparser = lineparsermod.LineParser(str(tmp_path), 'file')
        assert lineparser.data == ['one', 'two']
        (tmp_path / 'file').write_bytes(b'\xfe\n\xff\n')
        lineparser = lineparsermod.LineParser(str(tmp_path), 'file', binary=True)
        assert lineparser.data == [b'\xfe', b'\xff']

    def test_clear(self, tmp_path, lineparser):
        if False:
            i = 10
            return i + 15
        'Test if clear() empties its file.'
        lineparser.data = ['one', 'two']
        lineparser.save()
        assert (tmp_path / 'file').read_text() == 'one\ntwo\n'
        lineparser.clear()
        assert not lineparser.data
        assert (tmp_path / 'file').read_text() == ''

    def test_double_open(self, lineparser):
        if False:
            for i in range(10):
                print('nop')
        'Test if save() bails on an already open file.'
        with lineparser._open('r'):
            with pytest.raises(OSError, match='Refusing to double-open LineParser.'):
                lineparser.save()

    def test_prepare_save(self, tmp_path, lineparser):
        if False:
            while True:
                i = 10
        'Test if save() bails when _prepare_save() returns False.'
        (tmp_path / 'file').write_text('pristine\n')
        lineparser.data = ['changed']
        lineparser._prepare_save = lambda : False
        lineparser.save()
        assert (tmp_path / 'file').read_text() == 'pristine\n'