"""Side-effect Command Module."""
import logging
import click
from molecule.command import base
LOG = logging.getLogger(__name__)

class SideEffect(base.Base):
    """This action has side effects and not enabled by default.

    See the provisioners documentation for further details.
    """

    def execute(self, action_args=None):
        if False:
            print('Hello World!')
        'Execute the actions necessary to perform a `molecule side-effect` and         returns None.\n\n        :return: None\n        '
        if not self._config.provisioner.playbooks.side_effect:
            msg = 'Skipping, side effect playbook not configured.'
            LOG.warning(msg)
            return
        self._config.provisioner.side_effect(action_args)

@base.click_command_ex()
@click.pass_context
@click.option('--scenario-name', '-s', default=base.MOLECULE_DEFAULT_SCENARIO_NAME, help=f'Name of the scenario to target. ({base.MOLECULE_DEFAULT_SCENARIO_NAME})')
def side_effect(ctx, scenario_name):
    if False:
        for i in range(10):
            print('nop')
    'Use the provisioner to perform side-effects to the instances.'
    args = ctx.obj.get('args')
    subcommand = base._get_subcommand(__name__)
    command_args = {'subcommand': subcommand}
    base.execute_cmdline_scenarios(scenario_name, args, command_args)