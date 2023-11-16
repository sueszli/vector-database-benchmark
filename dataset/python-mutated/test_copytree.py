import os
from lightning.app.source_code.copytree import _copytree, _read_lightningignore

def test_read_lightningignore(tmpdir):
    if False:
        return 10
    '_read_lightningignore() removes comments from ignore files.'
    test_path = tmpdir.join('.lightningignore')
    expected = 'test'
    not_expected = '# comment'
    with open(test_path, 'a') as f:
        f.write(not_expected)
        f.write(expected)
    result = _read_lightningignore(test_path)
    assert not_expected not in result
    assert expected not in result

def test_read_lightningignore_excludes_empty_lines(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '_read_lightningignore() excludes empty lines.'
    test_path = tmpdir.join('.lightningignore')
    gitignore = '\n\n    foo\n\n    bar\n\n\n\n    '
    test_path.write(gitignore)
    result = _read_lightningignore(test_path)
    assert len(result) == 2

def test_copytree_ignoring_files(tmp_path_factory):
    if False:
        while True:
            i = 10
    test_dir = tmp_path_factory.mktemp('lightningignore-test')
    source = test_dir / 'source'
    source.mkdir()
    source.joinpath('.lightningignore').write_text('dir1/*.txt\ndir0\n*.zip')
    dest = test_dir / 'dest'
    source.joinpath('dir3').mkdir()
    source.joinpath('dir3').joinpath('.lightningignore').write_text('*.pt')
    source.joinpath('dir3').joinpath('model.pt').write_text('')
    source.joinpath('dir3').joinpath('model.non-pt').write_text('')
    source.joinpath('dir0').mkdir()
    source.joinpath('dir0/file1').write_text('')
    source.joinpath('dir1').mkdir()
    source.joinpath('dir1/file.tar.gz').write_text('')
    source.joinpath('dir1/file.txt').write_text('')
    source.joinpath('dir2').mkdir()
    source.joinpath('dir2/file.txt').write_text('')
    source.joinpath('dir2/file.zip').write_text('')
    files_copied = _copytree(source, dest)
    relative_names = set()
    for file in files_copied:
        relative_names.add(file.split('source')[1].strip('/').strip('\\'))
    if os.name == 'nt':
        assert {'.lightningignore', 'dir2\\file.txt', 'dir3\\.lightningignore', 'dir3\\model.non-pt', 'dir1\\file.tar.gz'} == relative_names
    else:
        assert {'.lightningignore', 'dir2/file.txt', 'dir3/.lightningignore', 'dir3/model.non-pt', 'dir1/file.tar.gz'} == relative_names
    first_level_dirs = list(dest.iterdir())
    assert len(first_level_dirs) == 4
    assert {'.lightningignore', 'dir2', 'dir1', 'dir3'} == {d.name for d in first_level_dirs}
    for d in first_level_dirs:
        if d.name == 'dir1':
            assert 'file.txt' not in [file.name for file in d.iterdir()]
            assert 'file.tar.gz' in [file.name for file in d.iterdir()]
            assert len([file.name for file in d.iterdir()]) == 1
        if d.name == 'dir2':
            assert 'file.zip' not in [file.name for file in d.iterdir()]
            assert 'file.txt' in [file.name for file in d.iterdir()]
            assert len([file.name for file in d.iterdir()]) == 1
        if d.name == 'dir3':
            assert 'model.pt' not in [file.name for file in d.iterdir()]
            assert 'model.non-pt' in [file.name for file in d.iterdir()]
            assert '.lightningignore' in [file.name for file in d.iterdir()]
            assert len([file.name for file in d.iterdir()]) == 2