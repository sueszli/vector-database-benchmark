"""Ansible-Playbook Provisioner Module."""
import logging
import shlex
import warnings
from molecule import util
from molecule.api import MoleculeRuntimeWarning
LOG = logging.getLogger(__name__)

class AnsiblePlaybook:
    """Provisioner Playbook."""

    def __init__(self, playbook, config, verify=False) -> None:
        if False:
            while True:
                i = 10
        'Set up the requirements to execute ``ansible-playbook`` and returns         None.\n\n        :param playbook: A string containing the path to the playbook.\n        :param config: An instance of a Molecule config.\n        :param verify: An optional bool to toggle the Plabook mode between\n         provision and verify. False: provision; True: verify. Default is False.\n        :returns: None\n        '
        self._ansible_command = None
        self._playbook = playbook
        self._config = config
        self._cli = {}
        if verify:
            self._env = util.merge_dicts(self._config.verifier.env, self._config.config['verifier']['env'])
        else:
            self._env = self._config.provisioner.env

    def bake(self):
        if False:
            for i in range(10):
                print('nop')
        "Bake an ``ansible-playbook`` command so it's ready to execute and         returns ``None``.\n\n        :return: None\n        "
        if not self._playbook:
            return
        self.add_cli_arg('inventory', self._config.provisioner.inventory_directory)
        options = util.merge_dicts(self._config.provisioner.options, self._cli)
        verbose_flag = util.verbose_flag(options)
        if self._playbook != self._config.provisioner.playbooks.converge:
            if options.get('become'):
                del options['become']
        if self._config.action not in ['create', 'destroy']:
            ansible_args = list(self._config.provisioner.ansible_args) + list(self._config.ansible_args)
        else:
            ansible_args = []
        self._ansible_command = ['ansible-playbook', *util.dict2args(options), *util.bool2args(verbose_flag), *ansible_args, self._playbook]

    def execute(self, action_args=None):
        if False:
            print('Hello World!')
        'Execute ``ansible-playbook`` and returns a string.\n\n        :return: str\n        '
        if self._ansible_command is None:
            self.bake()
        if not self._playbook:
            LOG.warning('Skipping, %s action has no playbook.', self._config.action)
            return None
        with warnings.catch_warnings(record=True) as warns:
            warnings.filterwarnings('default', category=MoleculeRuntimeWarning)
            self._config.driver.sanity_checks()
            cwd = self._config.scenario_path
            result = util.run_command(cmd=self._ansible_command, env=self._env, debug=self._config.debug, cwd=cwd)
        if result.returncode != 0:
            from rich.markup import escape
            util.sysexit_with_message(f'Ansible return code was {result.returncode}, command was: [dim]{escape(shlex.join(result.args))}[/dim]', result.returncode, warns=warns)
        return result.stdout

    def add_cli_arg(self, name, value):
        if False:
            print('Hello World!')
        'Add argument to CLI passed to ansible-playbook and returns None.\n\n        :param name: A string containing the name of argument to be added.\n        :param value: The value of argument to be added.\n        :return: None\n        '
        if value:
            self._cli[name] = value

    def add_env_arg(self, name, value):
        if False:
            while True:
                i = 10
        'Add argument to environment passed to ansible-playbook and returns         None.\n\n        :param name: A string containing the name of argument to be added.\n        :param value: The value of argument to be added.\n        :return: None\n        '
        self._env[name] = value