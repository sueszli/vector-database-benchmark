import pytest
from hscommon.conflict import get_conflicted_name, get_unconflicted_name, is_conflicted, smart_copy, smart_move
from pathlib import Path
from hscommon.testutil import eq_

class TestCaseGetConflictedName:

    def test_simple(self):
        if False:
            return 10
        name = get_conflicted_name(['bar'], 'bar')
        eq_('[000] bar', name)
        name = get_conflicted_name(['bar', '[000] bar'], 'bar')
        eq_('[001] bar', name)

    def test_no_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        name = get_conflicted_name(['bar'], 'foobar')
        eq_('foobar', name)

    def test_fourth_digit(self):
        if False:
            i = 10
            return i + 15
        names = ['bar'] + ['[%03d] bar' % i for i in range(1000)]
        name = get_conflicted_name(names, 'bar')
        eq_('[1000] bar', name)

    def test_auto_unconflict(self):
        if False:
            for i in range(10):
                print('nop')
        name = get_conflicted_name([], '[000] foobar')
        eq_('foobar', name)
        name = get_conflicted_name(['bar'], '[001] bar')
        eq_('[000] bar', name)

class TestCaseGetUnconflictedName:

    def test_main(self):
        if False:
            while True:
                i = 10
        eq_('foobar', get_unconflicted_name('[000] foobar'))
        eq_('foobar', get_unconflicted_name('[9999] foobar'))
        eq_('[000]foobar', get_unconflicted_name('[000]foobar'))
        eq_('[000a] foobar', get_unconflicted_name('[000a] foobar'))
        eq_('foobar', get_unconflicted_name('foobar'))
        eq_('foo [000] bar', get_unconflicted_name('foo [000] bar'))

class TestCaseIsConflicted:

    def test_main(self):
        if False:
            i = 10
            return i + 15
        assert is_conflicted('[000] foobar')
        assert is_conflicted('[9999] foobar')
        assert not is_conflicted('[000]foobar')
        assert not is_conflicted('[000a] foobar')
        assert not is_conflicted('foobar')
        assert not is_conflicted('foo [000] bar')

class TestCaseMoveCopy:

    @pytest.fixture
    def do_setup(self, request):
        if False:
            for i in range(10):
                print('nop')
        tmpdir = request.getfixturevalue('tmpdir')
        self.path = Path(str(tmpdir))
        self.path.joinpath('foo').touch()
        self.path.joinpath('bar').touch()
        self.path.joinpath('dir').mkdir()

    def test_move_no_conflict(self, do_setup):
        if False:
            print('Hello World!')
        smart_move(self.path.joinpath('foo'), self.path.joinpath('baz'))
        assert self.path.joinpath('baz').exists()
        assert not self.path.joinpath('foo').exists()

    def test_copy_no_conflict(self, do_setup):
        if False:
            for i in range(10):
                print('nop')
        smart_copy(self.path.joinpath('foo'), self.path.joinpath('baz'))
        assert self.path.joinpath('baz').exists()
        assert self.path.joinpath('foo').exists()

    def test_move_no_conflict_dest_is_dir(self, do_setup):
        if False:
            print('Hello World!')
        smart_move(self.path.joinpath('foo'), self.path.joinpath('dir'))
        assert self.path.joinpath('dir', 'foo').exists()
        assert not self.path.joinpath('foo').exists()

    def test_move_conflict(self, do_setup):
        if False:
            for i in range(10):
                print('nop')
        smart_move(self.path.joinpath('foo'), self.path.joinpath('bar'))
        assert self.path.joinpath('[000] bar').exists()
        assert not self.path.joinpath('foo').exists()

    def test_move_conflict_dest_is_dir(self, do_setup):
        if False:
            return 10
        smart_move(self.path.joinpath('foo'), self.path.joinpath('dir'))
        smart_move(self.path.joinpath('bar'), self.path.joinpath('foo'))
        smart_move(self.path.joinpath('foo'), self.path.joinpath('dir'))
        assert self.path.joinpath('dir', 'foo').exists()
        assert self.path.joinpath('dir', '[000] foo').exists()
        assert not self.path.joinpath('foo').exists()
        assert not self.path.joinpath('bar').exists()

    def test_copy_folder(self, tmpdir):
        if False:
            print('Hello World!')
        path = Path(str(tmpdir))
        path.joinpath('foo').mkdir()
        path.joinpath('bar').mkdir()
        smart_copy(path.joinpath('foo'), path.joinpath('bar'))
        assert path.joinpath('[000] bar').exists()