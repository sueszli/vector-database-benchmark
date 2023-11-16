"""Idempotence Command Module."""
import logging
import re
import click
from molecule import util
from molecule.command import base
from molecule.text import strip_ansi_escape
LOG = logging.getLogger(__name__)

class Idempotence(base.Base):
    """Runs the converge step a second time.

    If no tasks will be marked as changed     the scenario will be considered idempotent.
    """

    def execute(self, action_args=None):
        if False:
            return 10
        'Execute the actions necessary to perform a `molecule idempotence` and         returns None.\n\n        :return: None\n        '
        if not self._config.state.converged:
            msg = 'Instances not converged.  Please converge instances first.'
            util.sysexit_with_message(msg)
        output = self._config.provisioner.converge()
        idempotent = self._is_idempotent(output)
        if idempotent:
            msg = 'Idempotence completed successfully.'
            LOG.info(msg)
        else:
            details = '\n'.join(self._non_idempotent_tasks(output))
            msg = f'Idempotence test failed because of the following tasks:\n{details}'
            util.sysexit_with_message(msg)

    def _is_idempotent(self, output):
        if False:
            print('Hello World!')
        'Parse the output of the provisioning for changed and returns a bool.\n\n        :param output: A string containing the output of the ansible run.\n        :return: bool\n        '
        output = re.sub('\\n\\s*\\n*', '\n', output)
        changed = re.search('(changed=[1-9][0-9]*)', output)
        if changed:
            return False
        return True

    def _non_idempotent_tasks(self, output):
        if False:
            while True:
                i = 10
        'Parse the output to identify the non idempotent tasks.\n\n        :param (str) output: A string containing the output of the ansible run.\n        :return: A list containing the names of the non idempotent tasks.\n        '
        output = re.sub('\\n\\s*\\n*', '\n', output)
        output = strip_ansi_escape(output)
        output_lines = output.split('\n')
        res = []
        task_line = ''
        for (_, line) in enumerate(output_lines):
            if line.startswith('TASK'):
                task_line = line
            elif line.startswith('changed'):
                host_name = re.search('\\[(.*)\\]', line).groups()[0]
                task_name = re.search('\\[(.*)\\]', task_line).groups()[0]
                res.append(f'* [{host_name}] => {task_name}')
        return res

@base.click_command_ex()
@click.pass_context
@click.option('--scenario-name', '-s', default=base.MOLECULE_DEFAULT_SCENARIO_NAME, help=f'Name of the scenario to target. ({base.MOLECULE_DEFAULT_SCENARIO_NAME})')
@click.argument('ansible_args', nargs=-1, type=click.UNPROCESSED)
def idempotence(ctx, scenario_name, ansible_args):
    if False:
        for i in range(10):
            print('nop')
    'Use the provisioner to configure the instances and parse the output to     determine idempotence.\n    '
    args = ctx.obj.get('args')
    subcommand = base._get_subcommand(__name__)
    command_args = {'subcommand': subcommand}
    base.execute_cmdline_scenarios(scenario_name, args, command_args, ansible_args)