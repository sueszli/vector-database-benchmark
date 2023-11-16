"""Test certbot.reverter."""
import csv
import logging
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import pytest
from certbot import errors
from certbot.compat import os
from certbot.tests import util as test_util

class ReverterCheckpointLocalTest(test_util.ConfigTestCase):
    """Test the Reverter Class."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        from certbot.reverter import Reverter
        logging.disable(logging.CRITICAL)
        self.reverter = Reverter(self.config)
        tup = setup_test_files()
        (self.config1, self.config2, self.dir1, self.dir2, self.sets) = tup

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.config.work_dir)
        shutil.rmtree(self.dir1)
        shutil.rmtree(self.dir2)
        logging.disable(logging.NOTSET)

    @mock.patch('certbot.reverter.Reverter._read_and_append')
    def test_no_change(self, mock_read):
        if False:
            for i in range(10):
                print('nop')
        mock_read.side_effect = OSError('cannot even')
        try:
            self.reverter.add_to_checkpoint(self.sets[0], 'save1')
        except OSError:
            pass
        self.reverter.finalize_checkpoint('blah')
        path = os.listdir(self.reverter.config.backup_dir)[0]
        no_change = os.path.join(self.reverter.config.backup_dir, path, 'CHANGES_SINCE')
        with open(no_change, 'r') as f:
            x = f.read()
        assert 'No changes' in x

    def test_basic_add_to_temp_checkpoint(self):
        if False:
            return 10
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'save1')
        self.reverter.add_to_temp_checkpoint(self.sets[1], 'save2')
        assert os.path.isdir(self.config.temp_checkpoint_dir)
        assert get_save_notes(self.config.temp_checkpoint_dir) == 'save1save2'
        assert not os.path.isfile(os.path.join(self.config.temp_checkpoint_dir, 'NEW_FILES'))
        assert get_filepaths(self.config.temp_checkpoint_dir) == '{0}\n{1}\n'.format(self.config1, self.config2)

    def test_add_to_checkpoint_copy_failure(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('certbot.reverter.shutil.copy2') as mock_copy2:
            mock_copy2.side_effect = IOError('bad copy')
            with pytest.raises(errors.ReverterError):
                self.reverter.add_to_checkpoint(self.sets[0], 'save1')

    def test_checkpoint_conflict(self):
        if False:
            while True:
                i = 10
        'Make sure that checkpoint errors are thrown appropriately.'
        config3 = os.path.join(self.dir1, 'config3.txt')
        self.reverter.register_file_creation(True, config3)
        update_file(config3, 'This is a new file!')
        self.reverter.add_to_checkpoint(self.sets[2], 'save1')
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'save2')
        with pytest.raises(errors.ReverterError):
            self.reverter.add_to_checkpoint(self.sets[2], 'save3')
        self.reverter.add_to_checkpoint(self.sets[1], 'save4')
        with pytest.raises(errors.ReverterError):
            self.reverter.add_to_checkpoint({config3}, 'invalid save')

    def test_multiple_saves_and_temp_revert(self):
        if False:
            while True:
                i = 10
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'save1')
        update_file(self.config1, 'updated-directive')
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'save2-updated dir')
        update_file(self.config1, "new directive change that we won't keep")
        self.reverter.revert_temporary_config()
        assert read_in(self.config1) == 'directive-dir1'

    def test_multiple_registration_fail_and_revert(self):
        if False:
            return 10
        config3 = os.path.join(self.dir1, 'config3.txt')
        update_file(config3, 'Config3')
        config4 = os.path.join(self.dir2, 'config4.txt')
        update_file(config4, 'Config4')
        self.reverter.register_file_creation(True, self.config1)
        self.reverter.register_file_creation(True, self.config2)
        self.reverter.register_file_creation(True, config3, config4)
        self.reverter.recovery_routine()
        assert not os.path.isfile(self.config1)
        assert not os.path.isfile(self.config2)
        assert not os.path.isfile(config3)
        assert not os.path.isfile(config4)

    def test_multiple_registration_same_file(self):
        if False:
            while True:
                i = 10
        self.reverter.register_file_creation(True, self.config1)
        self.reverter.register_file_creation(True, self.config1)
        self.reverter.register_file_creation(True, self.config1)
        self.reverter.register_file_creation(True, self.config1)
        files = get_new_files(self.config.temp_checkpoint_dir)
        assert len(files) == 1

    def test_register_file_creation_write_error(self):
        if False:
            return 10
        m_open = mock.mock_open()
        with mock.patch('certbot.reverter.open', m_open, create=True):
            m_open.side_effect = OSError('bad open')
            with pytest.raises(errors.ReverterError):
                self.reverter.register_file_creation(True, self.config1)

    def test_bad_registration(self):
        if False:
            print('Hello World!')
        with pytest.raises(errors.ReverterError):
            self.reverter.register_file_creation('filepath')

    def test_register_undo_command(self):
        if False:
            for i in range(10):
                print('nop')
        coms = [['a2dismod', 'ssl'], ['a2dismod', 'rewrite'], ['cleanslate']]
        for com in coms:
            self.reverter.register_undo_command(True, com)
        act_coms = get_undo_commands(self.config.temp_checkpoint_dir)
        for (a_com, com) in zip(act_coms, coms):
            assert a_com == com

    def test_bad_register_undo_command(self):
        if False:
            for i in range(10):
                print('nop')
        m_open = mock.mock_open()
        with mock.patch('certbot.reverter.open', m_open, create=True):
            m_open.side_effect = OSError('bad open')
            with pytest.raises(errors.ReverterError):
                self.reverter.register_undo_command(True, ['command'])

    @mock.patch('certbot.util.run_script')
    def test_run_undo_commands(self, mock_run):
        if False:
            i = 10
            return i + 15
        mock_run.side_effect = ['', errors.SubprocessError]
        coms = [['invalid_command'], ['a2dismod', 'ssl']]
        for com in coms:
            self.reverter.register_undo_command(True, com)
        self.reverter.revert_temporary_config()
        assert mock_run.call_count == 2

    def test_recovery_routine_in_progress_failure(self):
        if False:
            print('Hello World!')
        self.reverter.add_to_checkpoint(self.sets[0], 'perm save')
        self.reverter._recover_checkpoint = mock.MagicMock(side_effect=errors.ReverterError)
        with pytest.raises(errors.ReverterError):
            self.reverter.recovery_routine()

    def test_recover_checkpoint_revert_temp_failures(self):
        if False:
            return 10
        mock_recover = mock.MagicMock(side_effect=errors.ReverterError('e'))
        self.reverter._recover_checkpoint = mock_recover
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'config1 save')
        with pytest.raises(errors.ReverterError):
            self.reverter.revert_temporary_config()

    def test_recover_checkpoint_rollback_failure(self):
        if False:
            while True:
                i = 10
        mock_recover = mock.MagicMock(side_effect=errors.ReverterError('e'))
        self.reverter._recover_checkpoint = mock_recover
        self.reverter.add_to_checkpoint(self.sets[0], 'config1 save')
        self.reverter.finalize_checkpoint('Title')
        with pytest.raises(errors.ReverterError):
            self.reverter.rollback_checkpoints(1)

    def test_recover_checkpoint_copy_failure(self):
        if False:
            print('Hello World!')
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'save1')
        with mock.patch('certbot.reverter.shutil.copy2') as mock_copy2:
            mock_copy2.side_effect = OSError('bad copy')
            with pytest.raises(errors.ReverterError):
                self.reverter.revert_temporary_config()

    def test_recover_checkpoint_rm_failure(self):
        if False:
            for i in range(10):
                print('nop')
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'temp save')
        with mock.patch('certbot.reverter.shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = OSError('Cannot remove tree')
            with pytest.raises(errors.ReverterError):
                self.reverter.revert_temporary_config()

    @mock.patch('certbot.reverter.logger.warning')
    def test_recover_checkpoint_missing_new_files(self, mock_warn):
        if False:
            while True:
                i = 10
        self.reverter.register_file_creation(True, os.path.join(self.dir1, 'missing_file.txt'))
        self.reverter.revert_temporary_config()
        assert mock_warn.call_count == 1

    @mock.patch('certbot.reverter.os.remove')
    def test_recover_checkpoint_remove_failure(self, mock_remove):
        if False:
            i = 10
            return i + 15
        self.reverter.register_file_creation(True, self.config1)
        mock_remove.side_effect = OSError("Can't remove")
        with pytest.raises(errors.ReverterError):
            self.reverter.revert_temporary_config()

    def test_recovery_routine_temp_and_perm(self):
        if False:
            while True:
                i = 10
        config3 = os.path.join(self.dir1, 'config3.txt')
        self.reverter.register_file_creation(False, config3)
        update_file(config3, 'This is a new perm file!')
        self.reverter.add_to_checkpoint(self.sets[0], 'perm save1')
        update_file(self.config1, 'updated perm config1')
        self.reverter.add_to_checkpoint(self.sets[1], 'perm save2')
        update_file(self.config2, 'updated perm config2')
        self.reverter.add_to_temp_checkpoint(self.sets[0], 'temp save1')
        update_file(self.config1, 'second update now temp config1')
        config4 = os.path.join(self.dir2, 'config4.txt')
        self.reverter.register_file_creation(True, config4)
        update_file(config4, 'New temporary file!')
        self.reverter.recovery_routine()
        assert not os.path.isfile(config3)
        assert not os.path.isfile(config4)
        assert read_in(self.config1) == 'directive-dir1'
        assert read_in(self.config2) == 'directive-dir2'

class TestFullCheckpointsReverter(test_util.ConfigTestCase):
    """Tests functions having to deal with full checkpoints."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        from certbot.reverter import Reverter
        logging.disable(logging.CRITICAL)
        self.reverter = Reverter(self.config)
        tup = setup_test_files()
        (self.config1, self.config2, self.dir1, self.dir2, self.sets) = tup

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.config.work_dir)
        shutil.rmtree(self.dir1)
        shutil.rmtree(self.dir2)
        logging.disable(logging.NOTSET)

    def test_rollback_improper_inputs(self):
        if False:
            print('Hello World!')
        with pytest.raises(errors.ReverterError):
            self.reverter.rollback_checkpoints('-1')
        with pytest.raises(errors.ReverterError):
            self.reverter.rollback_checkpoints(-1000)
        with pytest.raises(errors.ReverterError):
            self.reverter.rollback_checkpoints('one')

    def test_rollback_finalize_checkpoint_valid_inputs(self):
        if False:
            return 10
        config3 = self._setup_three_checkpoints()
        assert len(os.listdir(self.config.backup_dir)) == 3
        self.reverter.rollback_checkpoints(1)
        assert read_in(self.config1) == 'update config1'
        assert read_in(self.config2) == 'update config2'
        assert read_in(config3) == 'Final form config3'
        self.reverter.rollback_checkpoints(1)
        assert read_in(self.config1) == 'update config1'
        assert read_in(self.config2) == 'directive-dir2'
        assert not os.path.isfile(config3)
        all_dirs = os.listdir(self.config.backup_dir)
        assert len(all_dirs) == 1
        assert 'First Checkpoint' in get_save_notes(os.path.join(self.config.backup_dir, all_dirs[0]))
        self.reverter.rollback_checkpoints(1)
        assert read_in(self.config1) == 'directive-dir1'

    def test_finalize_checkpoint_no_in_progress(self):
        if False:
            print('Hello World!')
        self.reverter.finalize_checkpoint('No checkpoint...')

    @mock.patch('certbot.reverter.shutil.move')
    def test_finalize_checkpoint_cannot_title(self, mock_move):
        if False:
            for i in range(10):
                print('nop')
        self.reverter.add_to_checkpoint(self.sets[0], 'perm save')
        mock_move.side_effect = OSError('cannot move')
        with pytest.raises(errors.ReverterError):
            self.reverter.finalize_checkpoint('Title')

    @mock.patch('certbot.reverter.filesystem.replace')
    def test_finalize_checkpoint_no_rename_directory(self, mock_replace):
        if False:
            for i in range(10):
                print('nop')
        self.reverter.add_to_checkpoint(self.sets[0], 'perm save')
        mock_replace.side_effect = OSError
        with pytest.raises(errors.ReverterError):
            self.reverter.finalize_checkpoint('Title')

    @mock.patch('certbot.reverter.logger')
    def test_rollback_too_many(self, mock_logger):
        if False:
            print('Hello World!')
        self.reverter.rollback_checkpoints(1)
        assert mock_logger.warning.call_count == 1
        self._setup_three_checkpoints()
        mock_logger.warning.call_count = 0
        self.reverter.rollback_checkpoints(4)
        assert mock_logger.warning.call_count == 1

    def test_multi_rollback(self):
        if False:
            print('Hello World!')
        config3 = self._setup_three_checkpoints()
        self.reverter.rollback_checkpoints(3)
        assert read_in(self.config1) == 'directive-dir1'
        assert read_in(self.config2) == 'directive-dir2'
        assert not os.path.isfile(config3)

    def _setup_three_checkpoints(self):
        if False:
            print('Hello World!')
        'Generate some finalized checkpoints.'
        self.reverter.add_to_checkpoint(self.sets[0], 'first save')
        self.reverter.finalize_checkpoint('First Checkpoint')
        update_file(self.config1, 'update config1')
        config3 = os.path.join(self.dir1, 'config3.txt')
        self.reverter.register_file_creation(False, config3)
        update_file(config3, 'directive-config3')
        self.reverter.add_to_checkpoint(self.sets[1], 'second save')
        self.reverter.finalize_checkpoint('Second Checkpoint')
        update_file(self.config2, 'update config2')
        update_file(config3, 'update config3')
        self.reverter.add_to_checkpoint(self.sets[2], 'third save')
        self.reverter.finalize_checkpoint('Third Checkpoint - Save both')
        update_file(self.config1, 'Final form config1')
        update_file(self.config2, 'Final form config2')
        update_file(config3, 'Final form config3')
        return config3

def setup_test_files():
    if False:
        print('Hello World!')
    'Setup sample configuration files.'
    dir1 = tempfile.mkdtemp('dir1')
    dir2 = tempfile.mkdtemp('dir2')
    config1 = os.path.join(dir1, 'config.txt')
    config2 = os.path.join(dir2, 'config.txt')
    with open(config1, 'w') as file_fd:
        file_fd.write('directive-dir1')
    with open(config2, 'w') as file_fd:
        file_fd.write('directive-dir2')
    sets = [{config1}, {config2}, {config1, config2}]
    return (config1, config2, dir1, dir2, sets)

def get_save_notes(dire):
    if False:
        return 10
    'Read save notes'
    return read_in(os.path.join(dire, 'CHANGES_SINCE'))

def get_filepaths(dire):
    if False:
        while True:
            i = 10
    'Get Filepaths'
    return read_in(os.path.join(dire, 'FILEPATHS'))

def get_new_files(dire):
    if False:
        i = 10
        return i + 15
    'Get new files.'
    return read_in(os.path.join(dire, 'NEW_FILES')).splitlines()

def get_undo_commands(dire):
    if False:
        for i in range(10):
            print('nop')
    'Get new files.'
    with open(os.path.join(dire, 'COMMANDS')) as csvfile:
        return list(csv.reader(csvfile))

def read_in(path):
    if False:
        while True:
            i = 10
    'Read in a file, return the str'
    with open(path, 'r') as file_fd:
        return file_fd.read()

def update_file(filename, string):
    if False:
        return 10
    'Update a file with a new value.'
    with open(filename, 'w') as file_fd:
        file_fd.write(string)
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))