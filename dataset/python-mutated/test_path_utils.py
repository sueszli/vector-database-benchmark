import pytest
from tribler.core.utilities.path_util import Path, tail

@pytest.fixture
def tribler_tmp_path(tmp_path):
    if False:
        while True:
            i = 10
    return Path(tmp_path)

def test_put_path_relative(tmpdir):
    if False:
        while True:
            i = 10
    assert Path(tmpdir).normalize_to(None) == Path(tmpdir)
    assert Path(tmpdir).normalize_to('') == Path(tmpdir)
    assert Path(tmpdir).normalize_to('1/2') == Path(tmpdir)
    assert Path(tmpdir / '1').normalize_to(tmpdir) == Path('1')

def test_normalize_to(tmpdir):
    if False:
        print('Hello World!')
    assert Path(tmpdir).normalize_to(None) == Path(tmpdir)
    assert Path(tmpdir).normalize_to('') == Path(tmpdir)
    assert Path(tmpdir / '1' / '2').normalize_to(Path(tmpdir)) == Path('1') / '2'

def test_tail_no_file():
    if False:
        i = 10
        return i + 15
    'Test that in the case of missed file, an exception raises'
    with pytest.raises(FileNotFoundError):
        tail('missed_file.txt')

def test_tail_small_file(tribler_tmp_path: Path):
    if False:
        print('Hello World!')
    'Test that tail works correct with a small file '
    log_file = tribler_tmp_path / 'log.txt'
    log_file.write_text('text', 'utf-8')
    assert tail(log_file) == 'text'

def test_tail_count(tribler_tmp_path: Path):
    if False:
        while True:
            i = 10
    'Test that tail returns desired count of lines'
    log_file = tribler_tmp_path / 'log.txt'
    content = '\n'.join((f'{i}' for i in range(100)))
    log_file.write_text(content, 'utf-8')
    assert tail(log_file, 0) == ''
    assert tail(log_file, 1) == '99'
    assert tail(log_file, 2) == '98\n99'
    assert tail(log_file, 1000) == content
    with pytest.raises(ValueError):
        tail(log_file, -1)

def test_tail_encodings(tribler_tmp_path: Path):
    if False:
        while True:
            i = 10
    'Test that the `tail` function can read logs with "utf-8", "ascii", "latin-1" encodings '
    encodings = ['utf-8', 'ascii', 'latin-1']
    log_files = []
    content = '\n'.join((f'{i}' for i in range(100)))
    for encoding in encodings:
        path = tribler_tmp_path / encoding
        path.write_text(content, encoding)
        log_files.append(path)
    for log in log_files:
        assert tail(log, 100) == content

def test_size_file(tribler_tmp_path: Path):
    if False:
        for i in range(10):
            print('nop')
    path = tribler_tmp_path / '10bytes.file'
    path.write_bytes(b'0' * 10)
    assert path.size() == 10

def test_size_missed_file(tribler_tmp_path: Path):
    if False:
        for i in range(10):
            print('nop')
    path = tribler_tmp_path / '10bytes.file'
    assert path.size() == 0

def test_size_folder(tribler_tmp_path: Path):
    if False:
        print('Hello World!')
    (tribler_tmp_path / 'file.100bytes').write_bytes(b'0' * 100)
    (tribler_tmp_path / 'folder1').mkdir()
    (tribler_tmp_path / 'folder1' / 'file.100bytes').write_bytes(b'0' * 100)
    (tribler_tmp_path / 'folder1' / 'file1.100bytes').write_bytes(b'0' * 100)
    assert tribler_tmp_path.size(include_dir_sizes=False) == 300
    assert tribler_tmp_path.size() >= 300