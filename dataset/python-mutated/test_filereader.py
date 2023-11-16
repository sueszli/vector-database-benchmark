import codecs
import os
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from robot.utils import FileReader
from robot.utils.asserts import assert_equal, assert_raises
TEMPDIR = os.getenv('TEMPDIR') or tempfile.gettempdir()
PATH = os.path.join(TEMPDIR, 'filereader.test')
STRING = 'Hyvää\ntyötä\nCпасибо\n'

def assert_reader(reader, name=PATH):
    if False:
        print('Hello World!')
    assert_equal(reader.read(), STRING, formatter=repr)
    assert_equal(reader.name, name)
    assert_open(reader.file)

def assert_open(*files):
    if False:
        i = 10
        return i + 15
    for f in files:
        assert_equal(f.closed, False)

def assert_closed(*files):
    if False:
        for i in range(10):
            print('nop')
    for f in files:
        assert_equal(f.closed, True)

class TestReadFile(unittest.TestCase):
    BOM = b''
    created_files = set()

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls._create()

    @classmethod
    def _create(cls, content=STRING, path=PATH, encoding='UTF-8'):
        if False:
            while True:
                i = 10
        with open(path, 'wb') as f:
            f.write(cls.BOM)
            f.write(content.replace('\n', os.linesep).encode(encoding))
        cls.created_files.add(path)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        for path in cls.created_files:
            os.remove(path)
        cls.created_files = set()

    def test_path_as_string(self):
        if False:
            print('Hello World!')
        with FileReader(PATH) as reader:
            assert_reader(reader)
        assert_closed(reader.file)

    def test_open_text_file(self):
        if False:
            while True:
                i = 10
        with open(PATH, encoding='UTF-8') as f:
            with FileReader(f) as reader:
                assert_reader(reader)
            assert_open(f, reader.file)
        assert_closed(f, reader.file)

    def test_path_as_pathlib_path(self):
        if False:
            return 10
        with FileReader(Path(PATH)) as reader:
            assert_reader(reader)
        assert_closed(reader.file)

    def test_codecs_open_file(self):
        if False:
            print('Hello World!')
        with codecs.open(PATH, encoding='UTF-8') as f:
            with FileReader(f) as reader:
                assert_reader(reader)
            assert_open(f, reader.file)
        assert_closed(f, reader.file)

    def test_open_binary_file(self):
        if False:
            print('Hello World!')
        with open(PATH, 'rb') as f:
            with FileReader(f) as reader:
                assert_reader(reader)
            assert_open(f, reader.file)
        assert_closed(f, reader.file)

    def test_stringio(self):
        if False:
            return 10
        f = StringIO(STRING)
        with FileReader(f) as reader:
            assert_reader(reader, '<in-memory file>')
        assert_open(f)

    def test_bytesio(self):
        if False:
            for i in range(10):
                print('nop')
        f = BytesIO(self.BOM + STRING.encode('UTF-8'))
        with FileReader(f) as reader:
            assert_reader(reader, '<in-memory file>')
        assert_open(f)

    def test_text(self):
        if False:
            i = 10
            return i + 15
        with FileReader(STRING, accept_text=True) as reader:
            assert_reader(reader, '<in-memory file>')
        assert_closed(reader.file)

    def test_text_with_special_chars(self):
        if False:
            while True:
                i = 10
        for text in ('!"#¤%&/()=?', '*** Test Cases ***', 'in:va:lid'):
            with FileReader(text, accept_text=True) as reader:
                assert_equal(reader.read(), text)

    def test_text_when_text_is_not_accepted(self):
        if False:
            return 10
        assert_raises(IOError, FileReader, STRING)

    def test_readlines(self):
        if False:
            while True:
                i = 10
        with FileReader(PATH) as reader:
            assert_equal(list(reader.readlines()), STRING.splitlines(True))

    def test_invalid_encoding(self):
        if False:
            print('Hello World!')
        russian = STRING.split()[-1]
        path = os.path.join(TEMPDIR, 'filereader.iso88595')
        self._create(russian, path, encoding='ISO-8859-5')
        with FileReader(path) as reader:
            assert_raises(UnicodeDecodeError, reader.read)

class TestReadFileWithBom(TestReadFile):
    BOM = codecs.BOM_UTF8
if __name__ == '__main__':
    unittest.main()