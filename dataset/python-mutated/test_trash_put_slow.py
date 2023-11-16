import os
from os.path import exists as file_exists
from typing import List
from typing import Optional
import pytest
from tests import run_command
from tests.run_command import temp_dir
from tests.support.files import make_empty_file, require_empty_dir, make_sticky_dir
from tests.support.my_path import MyPath
from trashcli.fs import read_file
from trashcli.lib.environ import Environ

@pytest.fixture
def runner(temp_dir):
    if False:
        print('Hello World!')
    return Runner(temp_dir)

class Runner:

    def __init__(self, cwd):
        if False:
            print('Hello World!')
        self.cwd = cwd

    def run_trashput(self, args, env=None):
        if False:
            print('Hello World!')
        env = env or {}
        env['TRASH_PUT_FAKE_UID_FOR_TESTING'] = '123'
        return run_command.run_command(self.cwd, 'trash-put', list(args), env=env)

@pytest.mark.slow
class TestDeletingExistingFile:

    @pytest.fixture
    def trash_foo(self, temp_dir, runner):
        if False:
            for i in range(10):
                print('nop')
        make_empty_file(temp_dir / 'foo')
        result = runner.run_trashput([temp_dir / 'foo'], env={'XDG_DATA_HOME': temp_dir / 'XDG_DATA_HOME'})
        yield result

    def test_it_should_remove_the_file(self, temp_dir, trash_foo):
        if False:
            for i in range(10):
                print('nop')
        assert file_exists(temp_dir / 'foo') is False

    def test_it_should_remove_it_silently(self, trash_foo):
        if False:
            return 10
        assert trash_foo.stdout == ''

    def test_a_trashinfo_file_should_have_been_created(self, temp_dir, trash_foo):
        if False:
            while True:
                i = 10
        read_file(temp_dir / 'XDG_DATA_HOME/Trash/info/foo.trashinfo')

@pytest.mark.slow
class TestWhenDeletingAnExistingFileInVerboseMode:

    @pytest.fixture
    def run_trashput(self, temp_dir, runner):
        if False:
            print('Hello World!')
        make_empty_file(temp_dir / 'foo')
        return runner.run_trashput(['-v', temp_dir / 'foo'], env={'XDG_DATA_HOME': temp_dir / 'XDG_DATA_HOME', 'HOME': temp_dir / 'home'})

    def test_should_tell_where_a_file_is_trashed(self, temp_dir, run_trashput):
        if False:
            print('Hello World!')
        output = run_trashput.clean_tmp_and_grep(temp_dir, 'trashed in')
        assert "trash-put: '/foo' trashed in /XDG_DATA_HOME/Trash" in output

    def test_should_be_successful(self, run_trashput):
        if False:
            i = 10
            return i + 15
        assert 0 == run_trashput.exit_code

@pytest.mark.slow
class TestWhenDeletingANonExistingFile:

    def test_should_be_succesfull(self, temp_dir, runner):
        if False:
            return 10
        result = runner.run_trashput(['-v', temp_dir / 'non-existent'])
        assert 0 != result.exit_code

@pytest.mark.slow
class TestWhenFedWithDotArguments:

    def test_dot_argument_is_skipped(self, temp_dir, runner):
        if False:
            print('Hello World!')
        result = runner.run_trashput(['.'])
        assert result.stderr == "trash-put: cannot trash directory '.'\n"

    def test_dot_dot_argument_is_skipped(self, temp_dir, runner):
        if False:
            for i in range(10):
                print('nop')
        result = runner.run_trashput(['..'])
        assert result.stderr == "trash-put: cannot trash directory '..'\n"

    def test_dot_argument_is_skipped_even_in_subdirs(self, temp_dir, runner):
        if False:
            return 10
        sandbox = MyPath.make_temp_dir()
        result = runner.run_trashput(['%s/.' % sandbox])
        assert "trash-put: cannot trash '.' directory '%s/.'\n" % sandbox == result.stderr
        assert file_exists(sandbox)
        sandbox.clean_up()

    def test_dot_dot_argument_is_skipped_even_in_subdirs(self, temp_dir, runner):
        if False:
            for i in range(10):
                print('nop')
        sandbox = MyPath.make_temp_dir()
        result = runner.run_trashput(['%s/..' % sandbox])
        assert result.stderr == "trash-put: cannot trash '..' directory '%s/..'\n" % sandbox
        assert file_exists(sandbox)
        sandbox.clean_up()

@pytest.mark.slow
class TestUnsecureTrashDirMessages:

    @pytest.fixture
    def fake_vol(self, temp_dir):
        if False:
            for i in range(10):
                print('nop')
        vol = temp_dir / 'fake-vol'
        require_empty_dir(vol)
        return vol

    def test_when_is_unsticky(self, temp_dir, fake_vol, runner):
        if False:
            return 10
        make_empty_file(fake_vol / 'foo')
        require_empty_dir(fake_vol / '.Trash')
        result = runner.run_trashput(['--force-volume', fake_vol, '-v', fake_vol / 'foo'])
        assert result.clean_vol_and_grep('/.Trash/123', fake_vol) == ['trash-put:  `- failed to trash /vol/foo in /vol/.Trash/123, because trash dir is insecure, its parent should be sticky, trash-dir: /vol/.Trash/123, parent: /vol/.Trash']

    def test_when_it_is_not_a_dir(self, fake_vol, runner, temp_dir):
        if False:
            print('Hello World!')
        make_empty_file(fake_vol / 'foo')
        make_empty_file(fake_vol / '.Trash')
        result = runner.run_trashput(['--force-volume', fake_vol, '-v', fake_vol / 'foo'])
        assert result.clean_vol_and_grep('/.Trash/123', fake_vol) == ['trash-put:  `- failed to trash /vol/foo in /vol/.Trash/123, because trash dir cannot be created as its parent is a file instead of being a directory, trash-dir: /vol/.Trash/123, parent: /vol/.Trash']

    def test_when_is_a_symlink(self, fake_vol, temp_dir, runner):
        if False:
            i = 10
            return i + 15
        make_empty_file(fake_vol / 'foo')
        make_sticky_dir(fake_vol / 'link-destination')
        os.symlink('link-destination', fake_vol / '.Trash')
        result = runner.run_trashput(['--force-volume', fake_vol, '-v', fake_vol / 'foo'])
        assert result.clean_vol_and_grep('insecure', fake_vol) == ['trash-put:  `- failed to trash /vol/foo in /vol/.Trash/123, because trash dir is insecure, its parent should not be a symlink, trash-dir: /vol/.Trash/123, parent: /vol/.Trash']