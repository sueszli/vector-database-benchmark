"""
Generates the services and commands for selection in SAM CLI generate-event
"""
import functools
import click
from samcli.cli.cli_config_file import ConfigProvider, configuration_option
from samcli.cli.options import debug_option
from samcli.lib.generated_sample_events import events
from samcli.lib.telemetry.metric import track_command
from samcli.lib.utils.version_checker import check_newer_version

class ServiceCommand(click.MultiCommand):
    """
    Top level command that defines the service provided

    Methods
    ----------------
    get_command(self, ctx, cmd_name):
        Get the subcommand(s) under a given service name.
    list_commands(self, ctx):
        List all of the subcommands
    """

    def __init__(self, events_lib: events.Events, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Constructor for the ServiceCommand class\n\n        Parameters\n        ----------\n        events_lib: samcli.commands.local.lib.generated_sample_events.events\n            The events library that allows for CLI population and substitution\n        args: list\n            any arguments passed in before kwargs\n        kwargs: dict\n            dictionary containing the keys/values used to construct the ServiceCommand\n        '
        super().__init__(*args, **kwargs)
        if not events_lib:
            raise ValueError('Events library is necessary to run this command')
        self.events_lib = events_lib
        self.all_cmds = self.events_lib.event_mapping

    def get_command(self, ctx, cmd_name):
        if False:
            print('Hello World!')
        '\n        gets the subcommands under the service name\n\n        Parameters\n        ----------\n        ctx : Context\n            the context object passed into the method\n        cmd_name : str\n            the service name\n        Returns\n        -------\n        EventTypeSubCommand:\n            returns subcommand if successful, None if not.\n        '
        if cmd_name not in self.all_cmds:
            return None
        return EventTypeSubCommand(self.events_lib, cmd_name, self.all_cmds[cmd_name])

    def list_commands(self, ctx):
        if False:
            return 10
        '\n        lists the service commands available\n\n        Parameters\n        ----------\n        ctx: Context\n            the context object passed into the method\n        Returns\n        -------\n        list\n            returns sorted list of the service commands available\n        '
        return sorted(self.all_cmds.keys())

class EventTypeSubCommand(click.MultiCommand):
    """
    Class that describes the commands underneath a given service type

    Methods
    ----------------
    get_command(self, ctx, cmd_name):
        Get the subcommand(s) under a given service name.
    list_commands(self, ctx):
        List all of the subcommands
    """
    TAGS = 'tags'

    def __init__(self, events_lib: events.Events, top_level_cmd_name, subcmd_definition, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        constructor for the EventTypeSubCommand class\n\n        Parameters\n        ----------\n        events_lib: samcli.commands.local.lib.generated_sample_events.events\n            The events library that allows for CLI population and substitution\n        top_level_cmd_name: string\n            the service name\n        subcmd_definition: dict\n            the subcommands and their values underneath the service command\n        args: tuple\n            any arguments passed in before kwargs\n        kwargs: dict\n            key/value pairs passed into the constructor\n        '
        super().__init__(*args, **kwargs)
        self.top_level_cmd_name = top_level_cmd_name
        self.subcmd_definition = subcmd_definition
        self.events_lib = events_lib

    def get_command(self, ctx, cmd_name):
        if False:
            return 10
        '\n        gets the Click Commands underneath a service name\n\n        Parameters\n        ----------\n        ctx: Context\n            context object passed in\n        cmd_name: string\n            the service name\n        Returns\n        -------\n        cmd: Click.Command\n            the Click Commands that can be called from the CLI\n        '
        if cmd_name not in self.subcmd_definition:
            return None
        parameters = []
        for param_name in self.subcmd_definition[cmd_name][self.TAGS].keys():
            default = self.subcmd_definition[cmd_name][self.TAGS][param_name]['default']
            parameters.append(click.Option(['--{}'.format(param_name)], default=default, help="Specify the {} name you'd like, otherwise the default = {}".format(param_name, default)))
        command_callback = functools.partial(self.cmd_implementation, self.events_lib, self.top_level_cmd_name, cmd_name)
        cmd = click.Command(name=cmd_name, short_help=self.subcmd_definition[cmd_name]['help'], params=parameters, callback=command_callback)
        cmd = configuration_option(provider=ConfigProvider(section='parameters'))(debug_option(cmd))
        return cmd

    def list_commands(self, ctx):
        if False:
            i = 10
            return i + 15
        '\n        lists the commands underneath a particular event\n\n        Parameters\n        ----------\n        ctx: Context\n            the context object passed in\n        Returns\n        -------\n        the sorted list of commands under a service\n        '
        return sorted(self.subcmd_definition.keys())

    @staticmethod
    @track_command
    @check_newer_version
    def cmd_implementation(events_lib: events.Events, top_level_cmd_name: str, subcmd_name: str, *args, **kwargs) -> str:
        if False:
            return 10
        '\n        calls for value substitution in the event json and returns the\n        customized json as a string\n\n        Parameters\n        ----------\n        events_lib : events.Events\n            the Events library for generating events\n        top_level_cmd_name : string\n            the name of the service\n        subcmd_name : string\n            the name of the event under the service\n        args : tuple\n            any arguments passed in before kwargs\n        kwargs : dict\n            the keys and values for substitution in the json\n        Returns\n        -------\n        event : string\n            returns the customized event json as a string\n        '
        event = events_lib.generate_event(top_level_cmd_name, subcmd_name, kwargs)
        click.echo(event)
        return event

class GenerateEventCommand(ServiceCommand):
    """
    Class that brings ServiceCommand and EventTypeSubCommand into one for easy execution
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor for GenerateEventCommand class that brings together\n        ServiceCommand and EventTypeSubCommand into one class\n\n        Parameters\n        ----------\n        args: tuple\n            any arguments passed in before kwargs\n        kwargs: dict\n            commands, subcommands, and parameters for generate-event\n        '
        super().__init__(events.Events(), *args, **kwargs)