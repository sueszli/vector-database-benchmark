from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
import click

class DefaultGroup(click.Group):
    """
    Example Usage:

    python3 cmd.py <==> python3 cmd.py run
    python3 cmd.py --test foo bar <==> python3 cmd.py run --test foo bar

    cmd.py

    @click.group(cls=DefaultGroup, default_command="run")
    def cli():
        pass

    @cli.command()
    @click.option('--test')
    @click.option('--config')
    @click.option('--blah')
    def run(test, config, blah):
        click.echo("a")
        click.echo(test)

    @cli.command()
    def init()
        click.echo('The subcommand')

    cli()
    """
    ignore_unknown_options = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        default_command = kwargs.pop('default_command', None)
        super().__init__(*args, **kwargs)
        self.default_command_name = None
        if default_command is not None:
            self.default_command_name = default_command

    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        if False:
            print('Hello World!')
        '\n        If there are no arguments, insert default command\n        '
        if not args and self.default_command_name is not None:
            args.insert(0, self.default_command_name)
        return super().parse_args(ctx, args)

    def get_command(self, ctx: click.Context, command_name: str) -> Optional[click.Command]:
        if False:
            for i in range(10):
                print('nop')
        '\n        If COMMAND_NAME is not in self.commands then it means\n        it is an option/arg to the default command. So place\n        this in the context for retrieval in resolve_command.\n\n        Also replace COMMAND_NAME with the self.default_command_name\n\n        '
        if command_name not in self.commands and self.default_command_name:
            ctx._default_command_overwrite_args0 = command_name
            command_name = self.default_command_name
        return super().get_command(ctx, command_name)

    def resolve_command(self, ctx: click.Context, args: List[str]) -> Tuple[Optional[str], Optional[click.Command], List[str]]:
        if False:
            i = 10
            return i + 15
        '\n        MultiCommand.resolve_command assumes args[0] is the command name\n        if we are running a default command then args[0] will actually be\n        an option/arg to the default command.\n\n        In get_command we put this first arg into _default_command_overwrite_arg0\n        in the context. So here we just read it from context again\n\n        If args[0] is actually a command name then _default_command_overwrite_args0\n        will not be set so this function is equivalent to existing behavior\n        '
        (cmd_name, cmd, args) = super().resolve_command(ctx, args)
        if hasattr(ctx, '_default_command_overwrite_args0'):
            args.insert(0, ctx._default_command_overwrite_args0)
        return (cmd_name, cmd, args)