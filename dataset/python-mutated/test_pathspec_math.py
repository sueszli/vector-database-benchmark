import pytest
from dvc.pathspec_math import PatternInfo, _change_dirname

@pytest.mark.parametrize('patterns, dirname, changed', [('#comment', '/dir', '#comment'), ('\\#hash', '/dir', 'dir/**/#hash'), ('\\#hash', '/#dir', '#dir/**/#hash'), (' space', '/dir', 'dir/**/space'), ('\\ space', '/dir', 'dir/**/ space'), ('!include', '/dir', '!/dir/**/include'), ('\\!important!.txt', '/dir', 'dir/**/!important!.txt'), ('/separator.txt', '/dir', 'dir/separator.txt'), ('subdir/separator.txt', '/dir', 'dir/subdir/separator.txt'), ('no_sep', '/dir', 'dir/**/no_sep'), ('doc/fortz/', '/dir', 'dir/doc/fortz/'), ('fortz/', '/dir', 'dir/**/fortz/'), ('*aste*risk*', '/dir', 'dir/**/*aste*risk*'), ('?fi?le?', '/dir', 'dir/**/?fi?le?'), ('[a-zA-Z]file[a-zA-Z]', '/dir', 'dir/**/[a-zA-Z]file[a-zA-Z]'), ('**/foo', '/dir', 'dir/**/foo'), ('**/foo/bar', '/dir', 'dir/**/foo/bar'), ('abc/**', '/dir', 'dir/abc/**'), ('a/**/b', '/dir', 'dir/a/**/b'), ('/***.txt', '/dir', 'dir/***.txt'), ('data/***', '/dir', 'dir/data/***'), ('***/file.txt', '/dir', 'dir/***/file.txt'), ('***file', '/dir', 'dir/**/***file'), ('a/***/b', '/dir', 'dir/a/***/b')])
def test_dvcignore_pattern_change_dir(tmp_dir, patterns, dirname, changed):
    if False:
        return 10
    assert _change_dirname(dirname, [PatternInfo(patterns, '')], '/') == [PatternInfo(changed, '')]