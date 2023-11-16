"""Molecule Scenario Module."""
from __future__ import annotations
import errno
import fcntl
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from time import sleep
from molecule import scenarios, util
from molecule.constants import RC_TIMEOUT
LOG = logging.getLogger(__name__)

class Scenario:
    """A Molecule scenario."""

    def __init__(self, config) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a new scenario class and returns None.\n\n        :param config: An instance of a Molecule config.\n        :return: None\n        '
        self._lock = None
        self.config = config
        self._setup()

    def _remove_scenario_state_directory(self):
        if False:
            return 10
        'Remove scenario cached disk stored state.\n\n        :return: None\n        '
        directory = str(Path(self.ephemeral_directory).parent)
        LOG.info('Removing %s', directory)
        shutil.rmtree(directory)

    def prune(self):
        if False:
            for i in range(10):
                print('nop')
        'Prune the scenario ephemeral directory files and returns None.\n\n        "safe files" will not be pruned, including the ansible configuration\n        and inventory used by this scenario, the scenario state file, and\n        files declared as "safe_files" in the ``driver`` configuration\n        declared in ``molecule.yml``.\n\n        :return: None\n        '
        LOG.info('Pruning extra files from scenario ephemeral directory')
        safe_files = [self.config.provisioner.config_file, self.config.provisioner.inventory_file, self.config.state.state_file, *self.config.driver.safe_files]
        files = util.os_walk(self.ephemeral_directory, '*')
        for f in files:
            if not any((sf for sf in safe_files if fnmatch.fnmatch(f, sf))):
                try:
                    os.remove(f)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
        for (dirpath, dirs, files) in os.walk(self.ephemeral_directory, topdown=False):
            if not dirs and (not files):
                os.removedirs(dirpath)

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.config.config['scenario']['name']

    @property
    def directory(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config.molecule_file:
            return os.path.dirname(self.config.molecule_file)
        return os.getcwd()

    @property
    def ephemeral_directory(self):
        if False:
            print('Hello World!')
        path = os.getenv('MOLECULE_EPHEMERAL_DIRECTORY', None)
        if not path:
            project_directory = os.path.basename(self.config.project_directory)
            if self.config.is_parallel:
                project_directory = f'{project_directory}-{self.config._run_uuid}'
            project_scenario_directory = os.path.join(self.config.cache_directory, project_directory, self.name)
            path = ephemeral_directory(project_scenario_directory)
        if os.environ.get('MOLECULE_PARALLEL', False) and (not self._lock):
            with open(os.path.join(path, '.lock'), 'w') as self._lock:
                for i in range(1, 5):
                    try:
                        fcntl.lockf(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        delay = 30 * i
                        LOG.warning('Retrying to acquire lock on %s, waiting for %s seconds', path, delay)
                        sleep(delay)
                else:
                    LOG.warning('Timedout trying to acquire lock on %s', path)
                    raise SystemExit(RC_TIMEOUT)
        return path

    @property
    def inventory_directory(self):
        if False:
            print('Hello World!')
        return os.path.join(self.ephemeral_directory, 'inventory')

    @property
    def check_sequence(self):
        if False:
            while True:
                i = 10
        return self.config.config['scenario']['check_sequence']

    @property
    def cleanup_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.config['scenario']['cleanup_sequence']

    @property
    def converge_sequence(self):
        if False:
            i = 10
            return i + 15
        return self.config.config['scenario']['converge_sequence']

    @property
    def create_sequence(self):
        if False:
            i = 10
            return i + 15
        return self.config.config['scenario']['create_sequence']

    @property
    def dependency_sequence(self):
        if False:
            i = 10
            return i + 15
        return ['dependency']

    @property
    def destroy_sequence(self):
        if False:
            return 10
        return self.config.config['scenario']['destroy_sequence']

    @property
    def idempotence_sequence(self):
        if False:
            return 10
        return ['idempotence']

    @property
    def prepare_sequence(self):
        if False:
            print('Hello World!')
        return ['prepare']

    @property
    def side_effect_sequence(self):
        if False:
            while True:
                i = 10
        return ['side_effect']

    @property
    def syntax_sequence(self):
        if False:
            i = 10
            return i + 15
        return ['syntax']

    @property
    def test_sequence(self):
        if False:
            return 10
        return self.config.config['scenario']['test_sequence']

    @property
    def verify_sequence(self):
        if False:
            i = 10
            return i + 15
        return ['verify']

    @property
    def sequence(self) -> list[str]:
        if False:
            return 10
        'Select the sequence based on scenario and subcommand of the provided scenario object and returns a list.'
        result = []
        our_scenarios = scenarios.Scenarios([self.config])
        matrix = our_scenarios._get_matrix()
        try:
            result = matrix[self.name][self.config.subcommand]
            if not isinstance(result, list):
                raise RuntimeError('Unexpected sequence type {result}.')
        except KeyError:
            pass
        return result

    def _setup(self):
        if False:
            return 10
        'Prepare the scenario for Molecule and returns None.\n\n        :return: None\n        '
        if not os.path.isdir(self.inventory_directory):
            os.makedirs(self.inventory_directory, exist_ok=True)

def ephemeral_directory(path: str | None=None) -> str:
    if False:
        return 10
    'Return temporary directory to be used by molecule.\n\n    Molecule users should not make any assumptions about its location,\n    permissions or its content as this may change in future release.\n    '
    d = os.getenv('MOLECULE_EPHEMERAL_DIRECTORY')
    if not d:
        d = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    if not d:
        raise RuntimeError('Unable to determine ephemeral directory to use.')
    d = os.path.abspath(os.path.join(d, path if path else 'molecule'))
    if not os.path.isdir(d):
        os.umask(63)
        Path(d).mkdir(mode=448, parents=True, exist_ok=True)
    return d