import pytest
from hypothesistooling.releasemanagement import bump_version_info, parse_release_file_contents, release_date_string, replace_assignment_in_string as replace, update_markdown_changelog

def parse_release(contents):
    if False:
        return 10
    return parse_release_file_contents(contents, '<string>')

def test_update_single_line():
    if False:
        i = 10
        return i + 15
    assert replace('a = 1', 'a', '2') == 'a = 2'

def test_update_without_spaces():
    if False:
        return 10
    assert replace('a=1', 'a', '2') == 'a=2'

def test_update_in_middle():
    if False:
        return 10
    assert replace('a = 1\nb=2\nc = 3', 'b', '4') == 'a = 1\nb=4\nc = 3'

def test_quotes_string_to_assign():
    if False:
        while True:
            i = 10
    assert replace('a.c = 1', 'a.c', '2') == 'a.c = 2'
    with pytest.raises(ValueError):
        replace('abc = 1', 'a.c', '2')

def test_duplicates_are_errors():
    if False:
        return 10
    with pytest.raises(ValueError):
        replace('a = 1\na=1', 'a', '2')

def test_missing_is_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        replace('', 'a', '1')

def test_bump_minor_version():
    if False:
        while True:
            i = 10
    assert bump_version_info((1, 1, 1), 'minor')[0] == '1.2.0'

def test_parse_release_file():
    if False:
        while True:
            i = 10
    assert parse_release('RELEASE_TYPE: patch\nhi') == ('patch', 'hi')
    assert parse_release('RELEASE_TYPE: minor\n\n\n\nhi') == ('minor', 'hi')
    assert parse_release('RELEASE_TYPE: major\n \n\nhi') == ('major', 'hi')

def test_invalid_release():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        parse_release('RELEASE_TYPE: wrong\nstuff')
    with pytest.raises(ValueError):
        parse_release('')
TEST_CHANGELOG = f'\n# A test project 1.2.3 ({release_date_string()})\n\nsome stuff happened\n\n# some previous log entry\n'

def test_update_changelog(tmpdir):
    if False:
        return 10
    path = tmpdir.join('CHANGELOG.md')
    path.write('# some previous log entry\n')
    update_markdown_changelog(str(path), 'A test project', '1.2.3', 'some stuff happened')
    assert path.read().strip() == TEST_CHANGELOG.strip()

def test_changelog_parsing_strips_trailing_whitespace():
    if False:
        return 10
    header = 'RELEASE_TYPE: patch\n\n'
    contents = 'Adds a feature\n    indented.\n'
    (level, out) = parse_release(header + contents.replace('feature', 'feature    '))
    assert contents.strip() == out