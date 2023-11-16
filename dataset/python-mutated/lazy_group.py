import importlib
from typing import Dict, List, Optional
import asyncclick as click

class LazyGroup(click.Group):
    """
    A click Group that can lazily load subcommands.
    """

    def __init__(self, *args, lazy_subcommands: Optional[Dict[str, str]]=None, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> List[str]:
        if False:
            i = 10
            return i + 15
        base = super().list_commands(ctx)
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        if False:
            for i in range(10):
                print('nop')
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name: str) -> click.BaseCommand:
        if False:
            while True:
                i = 10
        import_path = self.lazy_subcommands[cmd_name]
        (modname, cmd_object_name) = import_path.rsplit('.', 1)
        mod = importlib.import_module(modname)
        cmd_object = getattr(mod, cmd_object_name)
        if not isinstance(cmd_object, click.BaseCommand):
            print(f'{cmd_object} is of instance {type(cmd_object)}')
            raise ValueError(f'Lazy loading of {import_path} failed by returning a non-command object')
        return cmd_object