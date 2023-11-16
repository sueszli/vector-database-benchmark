"""Converge Command Module."""
from __future__ import annotations
import logging
import click
from molecule.command import base
LOG = logging.getLogger(__name__)

class Converge(base.Base):
    """Converge Command Class."""

    def execute(self, action_args: list[str] | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Execute the actions necessary to perform a `molecule converge` and         returns None.\n\n        :return: None\n        '
        self._config.provisioner.converge()
        self._config.state.change_state('converged', True)

@base.click_command_ex()
@click.pass_context
@click.option('--scenario-name', '-s', default=base.MOLECULE_DEFAULT_SCENARIO_NAME, help=f'Name of the scenario to target. ({base.MOLECULE_DEFAULT_SCENARIO_NAME})')
@click.argument('ansible_args', nargs=-1, type=click.UNPROCESSED)
def converge(ctx, scenario_name, ansible_args):
    if False:
        for i in range(10):
            print('nop')
    'Use the provisioner to configure instances (dependency, create, prepare converge).'
    args = ctx.obj.get('args')
    subcommand = base._get_subcommand(__name__)
    command_args = {'subcommand': subcommand}
    base.execute_cmdline_scenarios(scenario_name, args, command_args, ansible_args)