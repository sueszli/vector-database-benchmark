import os
from io import BytesIO
import pytest
from nltk.corpus.reader import SeekableUnicodeStreamReader

def check_reader(unicode_string, encoding):
    if False:
        print('Hello World!')
    bytestr = unicode_string.encode(encoding)
    stream = BytesIO(bytestr)
    reader = SeekableUnicodeStreamReader(stream, encoding)
    assert reader.tell() == 0
    assert unicode_string == ''.join(reader.readlines())
    stream.seek(0, os.SEEK_END)
    assert reader.tell() == stream.tell()
    reader.seek(0)
    contents = ''
    char = None
    while char != '':
        char = reader.read(1)
        contents += char
    assert unicode_string == contents
ENCODINGS = ['ascii', 'latin1', 'greek', 'hebrew', 'utf-16', 'utf-8']
STRINGS = ['\n    This is a test file.\n    It is fairly short.\n    ', 'This file can be encoded with latin1. \x83', "    This is a test file.\n    Here's a blank line:\n\n    And here's some unicode: î ģ ￣\n    ", '    This is a test file.\n    Unicode characters: ó ∢ ㌳䑄 啕\n    ', "    This is a larger file.  It has some lines that are longer     than 72 characters.  It's got lots of repetition.  Here's     some unicode chars: î ģ ￣ \ueeee ⍅\n\n    How fun!  Let's repeat it twenty times.\n    " * 20]

@pytest.mark.parametrize('string', STRINGS)
def test_reader(string):
    if False:
        for i in range(10):
            print('nop')
    for encoding in ENCODINGS:
        try:
            string.encode(encoding)
        except UnicodeEncodeError:
            continue
        check_reader(string, encoding)

def test_reader_stream_closes_when_deleted():
    if False:
        while True:
            i = 10
    reader = SeekableUnicodeStreamReader(BytesIO(b''), 'ascii')
    assert not reader.stream.closed
    reader.__del__()
    assert reader.stream.closed

def teardown_module(module=None):
    if False:
        print('Hello World!')
    import gc
    gc.collect()