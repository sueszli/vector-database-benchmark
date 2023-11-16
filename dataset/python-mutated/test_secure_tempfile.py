import os
import pytest
from secure_tempfile import SecureTemporaryFile
MESSAGE = '410,757,864,530'

def test_read_before_writing():
    if False:
        while True:
            i = 10
    f = SecureTemporaryFile('/tmp')
    with pytest.raises(AssertionError) as err:
        f.read()
    assert 'You must write before reading!' in str(err)

def test_write_then_read_once():
    if False:
        for i in range(10):
            print('nop')
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    assert f.read().decode('utf-8') == MESSAGE

def test_write_twice_then_read_once():
    if False:
        return 10
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    f.write(MESSAGE)
    assert f.read().decode('utf-8') == MESSAGE * 2

def test_write_then_read_twice():
    if False:
        for i in range(10):
            print('nop')
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    assert f.read().decode('utf-8') == MESSAGE
    assert f.read() == b''

def test_write_then_read_then_write():
    if False:
        while True:
            i = 10
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    f.read()
    with pytest.raises(AssertionError) as err:
        f.write('be gentle to each other so we can be dangerous together')
    assert 'You cannot write after reading!' in str(err)

def test_read_write_unicode():
    if False:
        return 10
    f = SecureTemporaryFile('/tmp')
    unicode_msg = '鬼神 Kill Em All 1989'
    f.write(unicode_msg)
    assert f.read().decode('utf-8') == unicode_msg

def test_file_seems_encrypted():
    if False:
        return 10
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    with open(f.filepath, 'rb') as fh:
        contents = fh.read()
    assert MESSAGE.encode('utf-8') not in contents
    assert MESSAGE not in contents.decode()

def test_file_is_removed_from_disk():
    if False:
        return 10
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    assert os.path.exists(f.filepath)
    f.close()
    assert not os.path.exists(f.filepath)
    f = SecureTemporaryFile('/tmp')
    f.write(MESSAGE)
    f.read()
    assert os.path.exists(f.filepath)
    f.close()
    assert not os.path.exists(f.filepath)

def test_buffered_read():
    if False:
        for i in range(10):
            print('nop')
    f = SecureTemporaryFile('/tmp')
    msg = MESSAGE * 1000
    f.write(msg)
    out = b''
    while True:
        chars = f.read(1024)
        if chars:
            out += chars
        else:
            break
    assert out.decode('utf-8') == msg

def test_tmp_file_id_omits_invalid_chars():
    if False:
        i = 10
        return i + 15
    "The `SecureTempFile.tmp_file_id` instance attribute is used as the filename\n    for the secure temporary file. This attribute should not contain\n    invalid characters such as '/' and '\x00' (null)."
    f = SecureTemporaryFile('/tmp')
    assert '/' not in f.tmp_file_id
    assert '\x00' not in f.tmp_file_id