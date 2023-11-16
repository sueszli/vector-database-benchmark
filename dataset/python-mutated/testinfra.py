"""Testinfra Verifier Module."""
import glob
import logging
import os
from molecule import util
from molecule.api import Verifier
LOG = logging.getLogger(__name__)

class Testinfra(Verifier):
    """`Testinfra`_ is no longer the default test verifier since version 3.0.

    Additional options can be passed to ``testinfra`` through the options
    dict.  Any option set in this section will override the defaults.

    !!! note

        Molecule will remove any options matching '^[v]+$', and pass ``-vvv``
        to the underlying ``pytest`` command when executing ``molecule
        --debug``.

    ``` yaml
        verifier:
          name: testinfra
          options:
            n: 1
    ```

    The testing can be disabled by setting ``enabled`` to False.

    ``` yaml
        verifier:
          name: testinfra
          enabled: False
    ```

    Environment variables can be passed to the verifier.

    ``` yaml
        verifier:
          name: testinfra
          env:
            FOO: bar
    ```

    Change path to the test directory.

    ``` yaml
        verifier:
          name: testinfra
          directory: /foo/bar/
    ```

    Additional tests from another file or directory relative to the scenario's
    tests directory (supports regexp).

    ``` yaml
        verifier:
          name: testinfra
          additional_files_or_dirs:
            - ../path/to/test_1.py
            - ../path/to/test_2.py
            - ../path/to/directory/*
    ```
    .. _`Testinfra`: https://testinfra.readthedocs.io
    """

    def __init__(self, config=None) -> None:
        if False:
            print('Hello World!')
        'Set up the requirements to execute ``testinfra`` and returns None.\n\n        :param config: An instance of a Molecule config.\n        :return: None\n        '
        super().__init__(config)
        self._testinfra_command = None
        self._tests = []

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return 'testinfra'

    @property
    def default_options(self):
        if False:
            print('Hello World!')
        d = self._config.driver.testinfra_options
        d['p'] = 'no:cacheprovider'
        if self._config.debug:
            d['debug'] = True
            d['vvv'] = True
        if self._config.args.get('sudo'):
            d['sudo'] = True
        return d

    @property
    def options(self):
        if False:
            print('Hello World!')
        o = self._config.config['verifier']['options']
        if self._config.debug:
            o = util.filter_verbose_permutation(o)
        return util.merge_dicts(self.default_options, o)

    @property
    def default_env(self):
        if False:
            return 10
        env = util.merge_dicts(os.environ, self._config.env)
        env = util.merge_dicts(env, self._config.provisioner.env)
        return env

    @property
    def additional_files_or_dirs(self):
        if False:
            i = 10
            return i + 15
        files_list = []
        c = self._config.config
        for f in c['verifier']['additional_files_or_dirs']:
            glob_path = os.path.join(self._config.verifier.directory, f)
            glob_list = glob.glob(glob_path)
            if glob_list:
                files_list.extend(glob_list)
        return files_list

    def bake(self):
        if False:
            print('Hello World!')
        "Bake a ``testinfra`` command so it's ready to execute and returns None.\n\n        :return: None\n        "
        options = self.options
        verbose_flag = util.verbose_flag(options)
        args = verbose_flag
        self._testinfra_command = ['pytest', *util.dict2args(options), *self._tests, *args]

    def execute(self, action_args=None):
        if False:
            return 10
        if not self.enabled:
            msg = 'Skipping, verifier is disabled.'
            LOG.warning(msg)
            return
        if self._config:
            self._tests = self._get_tests(action_args)
        else:
            self._tests = []
        if not len(self._tests) > 0:
            msg = 'Skipping, no tests found.'
            LOG.warning(msg)
            return
        self.bake()
        msg = f'Executing Testinfra tests found in {self.directory}/...'
        LOG.info(msg)
        result = util.run_command(self._testinfra_command, env=self.env, debug=self._config.debug, cwd=self._config.scenario.directory)
        if result.returncode == 0:
            msg = 'Verifier completed successfully.'
            LOG.info(msg)
        else:
            util.sysexit(result.returncode)

    def _get_tests(self, action_args=None):
        if False:
            i = 10
            return i + 15
        "Walk the verifier's directory for tests and returns a list.\n\n        :return: list\n        "
        if action_args:
            tests = []
            for arg in action_args:
                args_tests = list(util.os_walk(os.path.join(self._config.scenario.directory, arg), 'test_*.py', followlinks=True))
                tests.extend(args_tests)
            return sorted(tests)
        return sorted(list(util.os_walk(self.directory, 'test_*.py', followlinks=True)) + self.additional_files_or_dirs)

    def schema(self):
        if False:
            return 10
        return {'verifier': {'type': 'dict', 'schema': {'name': {'type': 'string', 'allowed': ['testinfra']}}}}