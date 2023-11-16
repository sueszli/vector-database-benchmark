"""Reverter class saves configuration checkpoints and allows for recovery."""
import csv
import glob
import logging
import shutil
import time
import traceback
from typing import Iterable
from typing import List
from typing import Set
from typing import TextIO
from typing import Tuple
from certbot import configuration
from certbot import errors
from certbot import util
from certbot._internal import constants
from certbot.compat import filesystem
from certbot.compat import os
logger = logging.getLogger(__name__)

class Reverter:
    """Reverter Class - save and revert configuration checkpoints.

    This class can be used by the plugins, especially Installers, to
    undo changes made to the user's system. Modifications to files and
    commands to do undo actions taken by the plugin should be registered
    with this class before the action is taken.

    Once a change has been registered with this class, there are three
    states the change can be in. First, the change can be a temporary
    change. This should be used for changes that will soon be reverted,
    such as config changes for the purpose of solving a challenge.
    Changes are added to this state through calls to
    :func:`~add_to_temp_checkpoint` and reverted when
    :func:`~revert_temporary_config` or :func:`~recovery_routine` is
    called.

    The second state a change can be in is in progress. These changes
    are not temporary, however, they also have not been finalized in a
    checkpoint. A change must become in progress before it can be
    finalized. Changes are added to this state through calls to
    :func:`~add_to_checkpoint` and reverted when
    :func:`~recovery_routine` is called.

    The last state a change can be in is finalized in a checkpoint. A
    change is put into this state by first becoming an in progress
    change and then calling :func:`~finalize_checkpoint`. Changes
    in this state can be reverted through calls to
    :func:`~rollback_checkpoints`.

    As a final note, creating new files and registering undo commands
    are handled specially and use the methods
    :func:`~register_file_creation` and :func:`~register_undo_command`
    respectively. Both of these methods can be used to create either
    temporary or in progress changes.

    .. note:: Consider moving everything over to CSV format.

    :param config: Configuration.
    :type config: :class:`certbot.configuration.NamespaceConfig`

    """

    def __init__(self, config: configuration.NamespaceConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        util.make_or_verify_dir(config.backup_dir, constants.CONFIG_DIRS_MODE, self.config.strict_permissions)

    def revert_temporary_config(self) -> None:
        if False:
            print('Hello World!')
        'Reload users original configuration files after a temporary save.\n\n        This function should reinstall the users original configuration files\n        for all saves with temporary=True\n\n        :raises .ReverterError: when unable to revert config\n\n        '
        if os.path.isdir(self.config.temp_checkpoint_dir):
            try:
                self._recover_checkpoint(self.config.temp_checkpoint_dir)
            except errors.ReverterError:
                logger.critical('Incomplete or failed recovery for %s', self.config.temp_checkpoint_dir)
                raise errors.ReverterError('Unable to revert temporary config')

    def rollback_checkpoints(self, rollback: int=1) -> None:
        if False:
            i = 10
            return i + 15
        'Revert \'rollback\' number of configuration checkpoints.\n\n        :param int rollback: Number of checkpoints to reverse. A str num will be\n           cast to an integer. So "2" is also acceptable.\n\n        :raises .ReverterError:\n            if there is a problem with the input or if the function is\n            unable to correctly revert the configuration checkpoints\n\n        '
        try:
            rollback = int(rollback)
        except ValueError:
            logger.error('Rollback argument must be a positive integer')
            raise errors.ReverterError('Invalid Input')
        if rollback < 0:
            logger.error('Rollback argument must be a positive integer')
            raise errors.ReverterError('Invalid Input')
        backups = os.listdir(self.config.backup_dir)
        backups.sort()
        if not backups:
            logger.warning("Certbot hasn't modified your configuration, so rollback isn't available.")
        elif len(backups) < rollback:
            logger.warning('Unable to rollback %d checkpoints, only %d exist', rollback, len(backups))
        while rollback > 0 and backups:
            cp_dir = os.path.join(self.config.backup_dir, backups.pop())
            try:
                self._recover_checkpoint(cp_dir)
            except errors.ReverterError:
                logger.critical('Failed to load checkpoint during rollback')
                raise errors.ReverterError('Unable to load checkpoint during rollback')
            rollback -= 1

    def add_to_temp_checkpoint(self, save_files: Set[str], save_notes: str) -> None:
        if False:
            while True:
                i = 10
        'Add files to temporary checkpoint.\n\n        :param set save_files: set of filepaths to save\n        :param str save_notes: notes about changes during the save\n\n        '
        self._add_to_checkpoint_dir(self.config.temp_checkpoint_dir, save_files, save_notes)

    def add_to_checkpoint(self, save_files: Set[str], save_notes: str) -> None:
        if False:
            i = 10
            return i + 15
        'Add files to a permanent checkpoint.\n\n        :param set save_files: set of filepaths to save\n        :param str save_notes: notes about changes during the save\n\n        '
        self._check_tempfile_saves(save_files)
        self._add_to_checkpoint_dir(self.config.in_progress_dir, save_files, save_notes)

    def _add_to_checkpoint_dir(self, cp_dir: str, save_files: Set[str], save_notes: str) -> None:
        if False:
            print('Hello World!')
        'Add save files to checkpoint directory.\n\n        :param str cp_dir: Checkpoint directory filepath\n        :param set save_files: set of files to save\n        :param str save_notes: notes about changes made during the save\n\n        :raises IOError: if unable to open cp_dir + FILEPATHS file\n        :raises .ReverterError: if unable to add checkpoint\n\n        '
        util.make_or_verify_dir(cp_dir, constants.CONFIG_DIRS_MODE, self.config.strict_permissions)
        (op_fd, existing_filepaths) = self._read_and_append(os.path.join(cp_dir, 'FILEPATHS'))
        idx = len(existing_filepaths)
        for filename in save_files:
            if filename not in existing_filepaths:
                logger.debug('Creating backup of %s', filename)
                try:
                    shutil.copy2(filename, os.path.join(cp_dir, os.path.basename(filename) + '_' + str(idx)))
                    op_fd.write('{0}\n'.format(filename))
                except IOError:
                    op_fd.close()
                    logger.error('Unable to add file %s to checkpoint %s', filename, cp_dir)
                    raise errors.ReverterError('Unable to add file {0} to checkpoint {1}'.format(filename, cp_dir))
                idx += 1
        op_fd.close()
        with open(os.path.join(cp_dir, 'CHANGES_SINCE'), 'a') as notes_fd:
            notes_fd.write(save_notes)

    def _read_and_append(self, filepath: str) -> Tuple[TextIO, List[str]]:
        if False:
            return 10
        'Reads the file lines and returns a file obj.\n\n        Read the file returning the lines, and a pointer to the end of the file.\n\n        '
        if os.path.isfile(filepath):
            op_fd = open(filepath, 'r+')
            lines = op_fd.read().splitlines()
        else:
            lines = []
            op_fd = open(filepath, 'w')
        return (op_fd, lines)

    def _recover_checkpoint(self, cp_dir: str) -> None:
        if False:
            i = 10
            return i + 15
        'Recover a specific checkpoint.\n\n        Recover a specific checkpoint provided by cp_dir\n        Note: this function does not reload augeas.\n\n        :param str cp_dir: checkpoint directory file path\n\n        :raises errors.ReverterError: If unable to recover checkpoint\n\n        '
        if os.path.isfile(os.path.join(cp_dir, 'COMMANDS')):
            self._run_undo_commands(os.path.join(cp_dir, 'COMMANDS'))
        if os.path.isfile(os.path.join(cp_dir, 'FILEPATHS')):
            try:
                with open(os.path.join(cp_dir, 'FILEPATHS')) as paths_fd:
                    filepaths = paths_fd.read().splitlines()
                    for (idx, path) in enumerate(filepaths):
                        shutil.copy2(os.path.join(cp_dir, os.path.basename(path) + '_' + str(idx)), path)
            except (IOError, OSError):
                logger.error('Unable to recover files from %s', cp_dir)
                raise errors.ReverterError(f'Unable to recover files from {cp_dir}')
        self._remove_contained_files(os.path.join(cp_dir, 'NEW_FILES'))
        try:
            shutil.rmtree(cp_dir)
        except OSError:
            logger.error('Unable to remove directory: %s', cp_dir)
            raise errors.ReverterError('Unable to remove directory: %s' % cp_dir)

    def _run_undo_commands(self, filepath: str) -> None:
        if False:
            i = 10
            return i + 15
        'Run all commands in a file.'
        kwargs = {'newline': ''}
        with open(filepath, 'r', **kwargs) as csvfile:
            csvreader = csv.reader(csvfile)
            for command in reversed(list(csvreader)):
                try:
                    util.run_script(command)
                except errors.SubprocessError:
                    logger.error('Unable to run undo command: %s', ' '.join(command))

    def _check_tempfile_saves(self, save_files: Set[str]) -> None:
        if False:
            return 10
        "Verify save isn't overwriting any temporary files.\n\n        :param set save_files: Set of files about to be saved.\n\n        :raises certbot.errors.ReverterError:\n            when save is attempting to overwrite a temporary file.\n\n        "
        protected_files = []
        temp_path = os.path.join(self.config.temp_checkpoint_dir, 'FILEPATHS')
        if os.path.isfile(temp_path):
            with open(temp_path, 'r') as protected_fd:
                protected_files.extend(protected_fd.read().splitlines())
        new_path = os.path.join(self.config.temp_checkpoint_dir, 'NEW_FILES')
        if os.path.isfile(new_path):
            with open(new_path, 'r') as protected_fd:
                protected_files.extend(protected_fd.read().splitlines())
        for filename in protected_files:
            if filename in save_files:
                raise errors.ReverterError(f'Attempting to overwrite challenge file - {filename}')

    def register_file_creation(self, temporary: bool, *files: str) -> None:
        if False:
            return 10
        'Register the creation of all files during certbot execution.\n\n        Call this method before writing to the file to make sure that the\n        file will be cleaned up if the program exits unexpectedly.\n        (Before a save occurs)\n\n        :param bool temporary: If the file creation registry is for\n            a temp or permanent save.\n        :param \\*files: file paths (str) to be registered\n\n        :raises certbot.errors.ReverterError: If\n            call does not contain necessary parameters or if the file creation\n            is unable to be registered.\n\n        '
        if not files:
            raise errors.ReverterError('Forgot to provide files to registration call')
        cp_dir = self._get_cp_dir(temporary)
        new_fd = None
        try:
            (new_fd, ex_files) = self._read_and_append(os.path.join(cp_dir, 'NEW_FILES'))
            for path in files:
                if path not in ex_files:
                    new_fd.write('{0}\n'.format(path))
        except (IOError, OSError):
            logger.error('Unable to register file creation(s) - %s', files)
            raise errors.ReverterError('Unable to register file creation(s) - {0}'.format(files))
        finally:
            if new_fd is not None:
                new_fd.close()

    def register_undo_command(self, temporary: bool, command: Iterable[str]) -> None:
        if False:
            while True:
                i = 10
        'Register a command to be run to undo actions taken.\n\n        .. warning:: This function does not enforce order of operations in terms\n            of file modification vs. command registration.  All undo commands\n            are run first before all normal files are reverted to their previous\n            state.  If you need to maintain strict order, you may create\n            checkpoints before and after the the command registration. This\n            function may be improved in the future based on demand.\n\n        :param bool temporary: Whether the command should be saved in the\n            IN_PROGRESS or TEMPORARY checkpoints.\n        :param command: Command to be run.\n        :type command: list of str\n\n        '
        commands_fp = os.path.join(self._get_cp_dir(temporary), 'COMMANDS')
        kwargs = {'newline': ''}
        try:
            mode = 'a' if os.path.isfile(commands_fp) else 'w'
            with open(commands_fp, mode, **kwargs) as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(command)
        except (IOError, OSError):
            logger.error('Unable to register undo command')
            raise errors.ReverterError('Unable to register undo command.')

    def _get_cp_dir(self, temporary: bool) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the proper reverter directory.'
        if temporary:
            cp_dir = self.config.temp_checkpoint_dir
        else:
            cp_dir = self.config.in_progress_dir
        util.make_or_verify_dir(cp_dir, constants.CONFIG_DIRS_MODE, self.config.strict_permissions)
        return cp_dir

    def recovery_routine(self) -> None:
        if False:
            i = 10
            return i + 15
        'Revert configuration to most recent finalized checkpoint.\n\n        Remove all changes (temporary and permanent) that have not been\n        finalized. This is useful to protect against crashes and other\n        execution interruptions.\n\n        :raises .errors.ReverterError: If unable to recover the configuration\n\n        '
        self.revert_temporary_config()
        if os.path.isdir(self.config.in_progress_dir):
            try:
                self._recover_checkpoint(self.config.in_progress_dir)
            except errors.ReverterError:
                logger.critical('Incomplete or failed recovery for IN_PROGRESS checkpoint - %s', self.config.in_progress_dir)
                raise errors.ReverterError('Incomplete or failed recovery for IN_PROGRESS checkpoint - %s' % self.config.in_progress_dir)

    def _remove_contained_files(self, file_list: str) -> bool:
        if False:
            print('Hello World!')
        'Erase all files contained within file_list.\n\n        :param str file_list: file containing list of file paths to be deleted\n\n        :returns: Success\n        :rtype: bool\n\n        :raises certbot.errors.ReverterError: If\n            all files within file_list cannot be removed\n\n        '
        if not os.path.isfile(file_list):
            return False
        try:
            with open(file_list, 'r') as list_fd:
                filepaths = list_fd.read().splitlines()
                for path in filepaths:
                    if os.path.lexists(path):
                        os.remove(path)
                    else:
                        logger.warning('File: %s - Could not be found to be deleted\n - Certbot probably shut down unexpectedly', path)
        except (IOError, OSError):
            logger.critical('Unable to remove filepaths contained within %s', file_list)
            raise errors.ReverterError('Unable to remove filepaths contained within {0}'.format(file_list))
        return True

    def finalize_checkpoint(self, title: str) -> None:
        if False:
            return 10
        'Finalize the checkpoint.\n\n        Timestamps and permanently saves all changes made through the use\n        of :func:`~add_to_checkpoint` and :func:`~register_file_creation`\n\n        :param str title: Title describing checkpoint\n\n        :raises certbot.errors.ReverterError: when the\n            checkpoint is not able to be finalized.\n\n        '
        if not os.path.isdir(self.config.in_progress_dir):
            return
        changes_since_path = os.path.join(self.config.in_progress_dir, 'CHANGES_SINCE')
        changes_since_tmp_path = os.path.join(self.config.in_progress_dir, 'CHANGES_SINCE.tmp')
        if not os.path.exists(changes_since_path):
            logger.info('Rollback checkpoint is empty (no changes made?)')
            with open(changes_since_path, 'w') as f:
                f.write('No changes\n')
        try:
            with open(changes_since_tmp_path, 'w') as changes_tmp:
                changes_tmp.write('-- %s --\n' % title)
                with open(changes_since_path, 'r') as changes_orig:
                    changes_tmp.write(changes_orig.read())
            shutil.move(changes_since_tmp_path, changes_since_path)
        except (IOError, OSError):
            logger.error('Unable to finalize checkpoint - adding title')
            logger.debug('Exception was:\n%s', traceback.format_exc())
            raise errors.ReverterError('Unable to add title')
        self._timestamp_progress_dir()

    def _checkpoint_timestamp(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Determine the timestamp of the checkpoint, enforcing monotonicity.'
        timestamp = str(time.time())
        others = glob.glob(os.path.join(self.config.backup_dir, '[0-9]*'))
        others = [os.path.basename(d) for d in others]
        others.append(timestamp)
        others.sort()
        if others[-1] != timestamp:
            timetravel = str(float(others[-1]) + 1)
            logger.warning('Current timestamp %s does not correspond to newest reverter checkpoint; your clock probably jumped. Time travelling to %s', timestamp, timetravel)
            timestamp = timetravel
        elif len(others) > 1 and others[-2] == timestamp:
            logger.debug('Race condition with timestamp %s, incrementing by 0.01', timestamp)
            timetravel = str(float(others[-1]) + 0.01)
            timestamp = timetravel
        return timestamp

    def _timestamp_progress_dir(self) -> None:
        if False:
            i = 10
            return i + 15
        'Timestamp the checkpoint.'
        for _ in range(2):
            timestamp = self._checkpoint_timestamp()
            final_dir = os.path.join(self.config.backup_dir, timestamp)
            try:
                filesystem.replace(self.config.in_progress_dir, final_dir)
                return
            except OSError:
                logger.warning('Unexpected race condition, retrying (%s)', timestamp)
        logger.error('Unable to finalize checkpoint, %s -> %s', self.config.in_progress_dir, final_dir)
        raise errors.ReverterError('Unable to finalize checkpoint renaming')