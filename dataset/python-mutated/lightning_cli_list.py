from typing import Any
import click
from lightning.app.cli.cmd_apps import _AppManager

@click.group(name='list')
def get_list() -> None:
    if False:
        print('Hello World!')
    'List Lightning AI self-managed resources (e.g. apps)'
    pass

@get_list.command('apps')
def list_apps(**kwargs: Any) -> None:
    if False:
        print('Hello World!')
    'List your Lightning AI apps.'
    app_manager = _AppManager()
    app_manager.list()