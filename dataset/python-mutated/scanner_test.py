import pytest
from hscommon.jobprogress import job
from pathlib import Path
from hscommon.testutil import eq_
from core import fs
from core.engine import getwords, Match
from core.ignore import IgnoreList
from core.scanner import Scanner, ScanType
from core.me.scanner import ScannerME

class NamedObject:

    def __init__(self, name='foobar', size=1, path=None):
        if False:
            i = 10
            return i + 15
        if path is None:
            path = Path(name)
        else:
            path = Path(path, name)
        self.name = name
        self.size = size
        self.path = path
        self.words = getwords(name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<NamedObject {!r} {!r}>'.format(self.name, self.path)

    def exists(self):
        if False:
            for i in range(10):
                print('nop')
        return self.path.exists()
no = NamedObject

@pytest.fixture
def fake_fileexists(request):
    if False:
        print('Hello World!')
    monkeypatch = request.getfixturevalue('monkeypatch')
    monkeypatch.setattr(Path, 'exists', lambda _: True)

def test_empty(fake_fileexists):
    if False:
        return 10
    s = Scanner()
    r = s.get_dupe_groups([])
    eq_(r, [])

def test_default_settings(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    eq_(s.min_match_percentage, 80)
    eq_(s.scan_type, ScanType.FILENAME)
    eq_(s.mix_file_kind, True)
    eq_(s.word_weighting, False)
    eq_(s.match_similar_words, False)
    eq_(s.size_threshold, 0)
    eq_(s.large_size_threshold, 0)
    eq_(s.big_file_size_threshold, 0)

def test_simple_with_default_settings(fake_fileexists):
    if False:
        return 10
    s = Scanner()
    f = [no('foo bar', path='p1'), no('foo bar', path='p2'), no('foo bleh')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    g = r[0]
    eq_(len(g), 2)
    assert g.ref in f[:2]
    assert g.dupes[0] in f[:2]

def test_simple_with_lower_min_match(fake_fileexists):
    if False:
        for i in range(10):
            print('nop')
    s = Scanner()
    s.min_match_percentage = 50
    f = [no('foo bar', path='p1'), no('foo bar', path='p2'), no('foo bleh')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    g = r[0]
    eq_(len(g), 3)

def test_trim_all_ref_groups(fake_fileexists):
    if False:
        return 10
    s = Scanner()
    f = [no('foo', path='p1'), no('foo', path='p2'), no('bar', path='p1'), no('bar', path='p2')]
    f[2].is_ref = True
    f[3].is_ref = True
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    eq_(s.discarded_file_count, 0)

def test_prioritize(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    f = [no('foo', path='p1'), no('foo', path='p2'), no('bar', path='p1'), no('bar', path='p2')]
    f[1].size = 2
    f[2].size = 3
    f[3].is_ref = True
    r = s.get_dupe_groups(f)
    (g1, g2) = r
    assert f[1] in (g1.ref, g2.ref)
    assert f[0] in (g1.dupes[0], g2.dupes[0])
    assert f[3] in (g1.ref, g2.ref)
    assert f[2] in (g1.dupes[0], g2.dupes[0])

def test_content_scan(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    f = [no('foo'), no('bar'), no('bleh')]
    f[0].digest = f[0].digest_partial = f[0].digest_samples = 'foobar'
    f[1].digest = f[1].digest_partial = f[1].digest_samples = 'foobar'
    f[2].digest = f[2].digest_partial = f[1].digest_samples = 'bleh'
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    eq_(len(r[0]), 2)
    eq_(s.discarded_file_count, 0)

def test_content_scan_compare_sizes_first(fake_fileexists):
    if False:
        return 10

    class MyFile(no):

        @property
        def digest(self):
            if False:
                for i in range(10):
                    print('nop')
            raise AssertionError()
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    f = [MyFile('foo', 1), MyFile('bar', 2)]
    eq_(len(s.get_dupe_groups(f)), 0)

def test_ignore_file_size(fake_fileexists):
    if False:
        i = 10
        return i + 15
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    small_size = 10
    s.size_threshold = 0
    large_size = 100 * 1024 * 1024
    s.large_size_threshold = 0
    f = [no('smallignore1', small_size - 1), no('smallignore2', small_size - 1), no('small1', small_size), no('small2', small_size), no('large1', large_size), no('large2', large_size), no('largeignore1', large_size + 1), no('largeignore2', large_size + 1)]
    f[0].digest = f[0].digest_partial = f[0].digest_samples = 'smallignore'
    f[1].digest = f[1].digest_partial = f[1].digest_samples = 'smallignore'
    f[2].digest = f[2].digest_partial = f[2].digest_samples = 'small'
    f[3].digest = f[3].digest_partial = f[3].digest_samples = 'small'
    f[4].digest = f[4].digest_partial = f[4].digest_samples = 'large'
    f[5].digest = f[5].digest_partial = f[5].digest_samples = 'large'
    f[6].digest = f[6].digest_partial = f[6].digest_samples = 'largeignore'
    f[7].digest = f[7].digest_partial = f[7].digest_samples = 'largeignore'
    r = s.get_dupe_groups(f)
    eq_(len(r), 4)
    s.size_threshold = small_size
    r = s.get_dupe_groups(f)
    eq_(len(r), 3)
    s.size_threshold = 0
    s.large_size_threshold = large_size
    r = s.get_dupe_groups(f)
    eq_(len(r), 3)
    s.size_threshold = small_size
    r = s.get_dupe_groups(f)
    eq_(len(r), 2)

def test_big_file_partial_hashes(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    smallsize = 1
    bigsize = 100 * 1024 * 1024
    s.big_file_size_threshold = bigsize
    f = [no('bigfoo', bigsize), no('bigbar', bigsize), no('smallfoo', smallsize), no('smallbar', smallsize)]
    f[0].digest = f[0].digest_partial = f[0].digest_samples = 'foobar'
    f[1].digest = f[1].digest_partial = f[1].digest_samples = 'foobar'
    f[2].digest = f[2].digest_partial = 'bleh'
    f[3].digest = f[3].digest_partial = 'bleh'
    r = s.get_dupe_groups(f)
    eq_(len(r), 2)
    f[1].digest = f[1].digest_samples = 'difffoobar'
    s.big_file_size_threshold = 0
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    s.big_file_size_threshold = bigsize
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)

def test_min_match_perc_doesnt_matter_for_content_scan(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    f = [no('foo'), no('bar'), no('bleh')]
    f[0].digest = f[0].digest_partial = f[0].digest_samples = 'foobar'
    f[1].digest = f[1].digest_partial = f[1].digest_samples = 'foobar'
    f[2].digest = f[2].digest_partial = f[2].digest_samples = 'bleh'
    s.min_match_percentage = 101
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    eq_(len(r[0]), 2)
    s.min_match_percentage = 0
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    eq_(len(r[0]), 2)

def test_content_scan_doesnt_put_digest_in_words_at_the_end(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    f = [no('foo'), no('bar')]
    f[0].digest = f[0].digest_partial = f[0].digest_samples = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    f[1].digest = f[1].digest_partial = f[1].digest_samples = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
    r = s.get_dupe_groups(f)
    r[0]

def test_extension_is_not_counted_in_filename_scan(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.min_match_percentage = 100
    f = [no('foo.bar'), no('foo.bleh')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    eq_(len(r[0]), 2)

def test_job(fake_fileexists):
    if False:
        for i in range(10):
            print('nop')

    def do_progress(progress, desc=''):
        if False:
            return 10
        log.append(progress)
        return True
    s = Scanner()
    log = []
    f = [no('foo bar'), no('foo bar'), no('foo bleh')]
    s.get_dupe_groups(f, j=job.Job(1, do_progress))
    eq_(log[0], 0)
    eq_(log[-1], 100)

def test_mix_file_kind(fake_fileexists):
    if False:
        i = 10
        return i + 15
    s = Scanner()
    s.mix_file_kind = False
    f = [no('foo.1'), no('foo.2')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 0)

def test_word_weighting(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.min_match_percentage = 75
    s.word_weighting = True
    f = [no('foo bar'), no('foo bar bleh')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)
    g = r[0]
    m = g.get_match_of(g.dupes[0])
    eq_(m.percentage, 75)

def test_similar_words(fake_fileexists):
    if False:
        i = 10
        return i + 15
    s = Scanner()
    s.match_similar_words = True
    f = [no('The White Stripes'), no('The Whites Stripe'), no('Limp Bizkit'), no('Limp Bizkitt')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 2)

def test_fields(fake_fileexists):
    if False:
        for i in range(10):
            print('nop')
    s = Scanner()
    s.scan_type = ScanType.FIELDS
    f = [no('The White Stripes - Little Ghost'), no('The White Stripes - Little Acorn')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 0)

def test_fields_no_order(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    s.scan_type = ScanType.FIELDSNOORDER
    f = [no('The White Stripes - Little Ghost'), no('Little Ghost - The White Stripes')]
    r = s.get_dupe_groups(f)
    eq_(len(r), 1)

def test_tag_scan(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    s.scan_type = ScanType.TAG
    o1 = no('foo')
    o2 = no('bar')
    o1.artist = 'The White Stripes'
    o1.title = 'The Air Near My Fingers'
    o2.artist = 'The White Stripes'
    o2.title = 'The Air Near My Fingers'
    r = s.get_dupe_groups([o1, o2])
    eq_(len(r), 1)

def test_tag_with_album_scan(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    s.scan_type = ScanType.TAG
    s.scanned_tags = {'artist', 'album', 'title'}
    o1 = no('foo')
    o2 = no('bar')
    o3 = no('bleh')
    o1.artist = 'The White Stripes'
    o1.title = 'The Air Near My Fingers'
    o1.album = 'Elephant'
    o2.artist = 'The White Stripes'
    o2.title = 'The Air Near My Fingers'
    o2.album = 'Elephant'
    o3.artist = 'The White Stripes'
    o3.title = 'The Air Near My Fingers'
    o3.album = 'foobar'
    r = s.get_dupe_groups([o1, o2, o3])
    eq_(len(r), 1)

def test_that_dash_in_tags_dont_create_new_fields(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    s.scan_type = ScanType.TAG
    s.scanned_tags = {'artist', 'album', 'title'}
    s.min_match_percentage = 50
    o1 = no('foo')
    o2 = no('bar')
    o1.artist = 'The White Stripes - a'
    o1.title = 'The Air Near My Fingers - a'
    o1.album = 'Elephant - a'
    o2.artist = 'The White Stripes - b'
    o2.title = 'The Air Near My Fingers - b'
    o2.album = 'Elephant - b'
    r = s.get_dupe_groups([o1, o2])
    eq_(len(r), 1)

def test_tag_scan_with_different_scanned(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.scan_type = ScanType.TAG
    s.scanned_tags = {'track', 'year'}
    o1 = no('foo')
    o2 = no('bar')
    o1.artist = 'The White Stripes'
    o1.title = 'some title'
    o1.track = 'foo'
    o1.year = 'bar'
    o2.artist = 'The White Stripes'
    o2.title = 'another title'
    o2.track = 'foo'
    o2.year = 'bar'
    r = s.get_dupe_groups([o1, o2])
    eq_(len(r), 1)

def test_tag_scan_only_scans_existing_tags(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.scan_type = ScanType.TAG
    s.scanned_tags = {'artist', 'foo'}
    o1 = no('foo')
    o2 = no('bar')
    o1.artist = 'The White Stripes'
    o1.foo = 'foo'
    o2.artist = 'The White Stripes'
    o2.foo = 'bar'
    r = s.get_dupe_groups([o1, o2])
    eq_(len(r), 1)

def test_tag_scan_converts_to_str(fake_fileexists):
    if False:
        return 10
    s = Scanner()
    s.scan_type = ScanType.TAG
    s.scanned_tags = {'track'}
    o1 = no('foo')
    o2 = no('bar')
    o1.track = 42
    o2.track = 42
    try:
        r = s.get_dupe_groups([o1, o2])
    except TypeError:
        raise AssertionError()
    eq_(len(r), 1)

def test_tag_scan_non_ascii(fake_fileexists):
    if False:
        for i in range(10):
            print('nop')
    s = Scanner()
    s.scan_type = ScanType.TAG
    s.scanned_tags = {'title'}
    o1 = no('foo')
    o2 = no('bar')
    o1.title = 'foobaré'
    o2.title = 'foobaré'
    try:
        r = s.get_dupe_groups([o1, o2])
    except UnicodeEncodeError:
        raise AssertionError()
    eq_(len(r), 1)

def test_ignore_list(fake_fileexists):
    if False:
        return 10
    s = Scanner()
    f1 = no('foobar')
    f2 = no('foobar')
    f3 = no('foobar')
    f1.path = Path('dir1/foobar')
    f2.path = Path('dir2/foobar')
    f3.path = Path('dir3/foobar')
    ignore_list = IgnoreList()
    ignore_list.ignore(str(f1.path), str(f2.path))
    ignore_list.ignore(str(f1.path), str(f3.path))
    r = s.get_dupe_groups([f1, f2, f3], ignore_list=ignore_list)
    eq_(len(r), 1)
    g = r[0]
    eq_(len(g.dupes), 1)
    assert f1 not in g
    assert f2 in g
    assert f3 in g
    eq_(s.discarded_file_count, 0)

def test_ignore_list_checks_for_unicode(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    f1 = no('foobar')
    f2 = no('foobar')
    f3 = no('foobar')
    f1.path = Path('foo1é')
    f2.path = Path('foo2é')
    f3.path = Path('foo3é')
    ignore_list = IgnoreList()
    ignore_list.ignore(str(f1.path), str(f2.path))
    ignore_list.ignore(str(f1.path), str(f3.path))
    r = s.get_dupe_groups([f1, f2, f3], ignore_list=ignore_list)
    eq_(len(r), 1)
    g = r[0]
    eq_(len(g.dupes), 1)
    assert f1 not in g
    assert f2 in g
    assert f3 in g

def test_file_evaluates_to_false(fake_fileexists):
    if False:
        return 10

    class FalseNamedObject(NamedObject):

        def __bool__(self):
            if False:
                print('Hello World!')
            return False
    s = Scanner()
    f1 = FalseNamedObject('foobar', path='p1')
    f2 = FalseNamedObject('foobar', path='p2')
    r = s.get_dupe_groups([f1, f2])
    eq_(len(r), 1)

def test_size_threshold(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    f1 = no('foo', 1, path='p1')
    f2 = no('foo', 2, path='p2')
    f3 = no('foo', 3, path='p3')
    s.size_threshold = 2
    groups = s.get_dupe_groups([f1, f2, f3])
    eq_(len(groups), 1)
    [group] = groups
    eq_(len(group), 2)
    assert f1 not in group
    assert f2 in group
    assert f3 in group

def test_tie_breaker_path_deepness(fake_fileexists):
    if False:
        print('Hello World!')
    s = Scanner()
    (o1, o2) = (no('foo'), no('foo'))
    o1.path = Path('foo')
    o2.path = Path('foo/bar')
    [group] = s.get_dupe_groups([o1, o2])
    assert group.ref is o2

def test_tie_breaker_copy(fake_fileexists):
    if False:
        for i in range(10):
            print('nop')
    s = Scanner()
    (o1, o2) = (no('foo bar Copy'), no('foo bar'))
    o1.path = Path('deeper/path')
    o2.path = Path('foo')
    [group] = s.get_dupe_groups([o1, o2])
    assert group.ref is o2

def test_tie_breaker_same_name_plus_digit(fake_fileexists):
    if False:
        i = 10
        return i + 15
    s = Scanner()
    o1 = no('foo bar 42')
    o2 = no('foo bar [42]')
    o3 = no('foo bar (42)')
    o4 = no('foo bar {42}')
    o5 = no('foo bar')
    o1.path = Path('deeper/path')
    o2.path = Path('deeper/path')
    o3.path = Path('deeper/path')
    o4.path = Path('deeper/path')
    o5.path = Path('foo')
    [group] = s.get_dupe_groups([o1, o2, o3, o4, o5])
    assert group.ref is o5

def test_partial_group_match(fake_fileexists):
    if False:
        for i in range(10):
            print('nop')
    s = Scanner()
    (o1, o2, o3) = (no('a b'), no('a'), no('b'))
    s.min_match_percentage = 50
    [group] = s.get_dupe_groups([o1, o2, o3])
    eq_(len(group), 2)
    assert o1 in group
    if o2 in group:
        assert o3 not in group
    else:
        assert o3 in group
    eq_(s.discarded_file_count, 1)

def test_dont_group_files_that_dont_exist(tmpdir):
    if False:
        return 10
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    p = Path(str(tmpdir))
    with p.joinpath('file1').open('w') as fp:
        fp.write('foo')
    with p.joinpath('file2').open('w') as fp:
        fp.write('foo')
    (file1, file2) = fs.get_files(p)

    def getmatches(*args, **kw):
        if False:
            i = 10
            return i + 15
        file2.path.unlink()
        return [Match(file1, file2, 100)]
    s._getmatches = getmatches
    assert not s.get_dupe_groups([file1, file2])

def test_folder_scan_exclude_subfolder_matches(fake_fileexists):
    if False:
        i = 10
        return i + 15
    s = Scanner()
    s.scan_type = ScanType.FOLDERS
    topf1 = no('top folder 1', size=42)
    topf1.digest = topf1.digest_partial = topf1.digest_samples = b'some_digest__1'
    topf1.path = Path('/topf1')
    topf2 = no('top folder 2', size=42)
    topf2.digest = topf2.digest_partial = topf2.digest_samples = b'some_digest__1'
    topf2.path = Path('/topf2')
    subf1 = no('sub folder 1', size=41)
    subf1.digest = subf1.digest_partial = subf1.digest_samples = b'some_digest__2'
    subf1.path = Path('/topf1/sub')
    subf2 = no('sub folder 2', size=41)
    subf2.digest = subf2.digest_partial = subf2.digest_samples = b'some_digest__2'
    subf2.path = Path('/topf2/sub')
    eq_(len(s.get_dupe_groups([topf1, topf2, subf1, subf2])), 1)
    otherf = no('other folder', size=41)
    otherf.digest = otherf.digest_partial = otherf.digest_samples = b'some_digest__2'
    otherf.path = Path('/otherfolder')
    eq_(len(s.get_dupe_groups([topf1, topf2, subf1, subf2, otherf])), 2)

def test_ignore_files_with_same_path(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    f1 = no('foobar', path='path1/foobar')
    f2 = no('foobar', path='path1/foobar')
    eq_(s.get_dupe_groups([f1, f2]), [])

def test_dont_count_ref_files_as_discarded(fake_fileexists):
    if False:
        while True:
            i = 10
    s = Scanner()
    s.scan_type = ScanType.CONTENTS
    o1 = no('foo', path='p1')
    o2 = no('foo', path='p2')
    o3 = no('foo', path='p3')
    o1.digest = o1.digest_partial = o1.digest_samples = 'foobar'
    o2.digest = o2.digest_partial = o2.digest_samples = 'foobar'
    o3.digest = o3.digest_partial = o3.digest_samples = 'foobar'
    o1.is_ref = True
    o2.is_ref = True
    eq_(len(s.get_dupe_groups([o1, o2, o3])), 1)
    eq_(s.discarded_file_count, 0)

def test_prioritize_me(fake_fileexists):
    if False:
        i = 10
        return i + 15
    s = ScannerME()
    (o1, o2) = (no('foo', path='p1'), no('foo', path='p2'))
    o1.bitrate = 1
    o2.bitrate = 2
    [group] = s.get_dupe_groups([o1, o2])
    assert group.ref is o2