import click
import inquirer
from inquirer.themes import GreenPassion
from rich.console import Console
from lightning.app.cli.cmd_apps import _AppManager

@click.group('delete')
def delete() -> None:
    if False:
        while True:
            i = 10
    'Delete Lightning AI self-managed resources (e.g. apps)'
    pass

def _find_selected_app_instance_id(app_name: str) -> str:
    if False:
        while True:
            i = 10
    console = Console()
    app_manager = _AppManager()
    all_app_names_and_ids = {}
    selected_app_instance_id = None
    for app in app_manager.list_apps():
        all_app_names_and_ids[app.name] = app.id
        if app_name == app.name or app_name == app.id:
            selected_app_instance_id = app.id
            break
    if selected_app_instance_id is None:
        console.print(f'[b][yellow]Cannot find app named "{app_name}"[/yellow][/b]')
        try:
            ask = [inquirer.List('app_name', message='Select the app name to delete', choices=list(all_app_names_and_ids.keys()))]
            app_name = inquirer.prompt(ask, theme=GreenPassion(), raise_keyboard_interrupt=True)['app_name']
            selected_app_instance_id = all_app_names_and_ids[app_name]
        except KeyboardInterrupt:
            console.print('[b][red]Cancelled by user![/b][/red]')
            raise InterruptedError
    return selected_app_instance_id

def _delete_app_confirmation_prompt(app_name: str) -> None:
    if False:
        i = 10
        return i + 15
    console = Console()
    try:
        ask = [inquirer.Confirm('confirm', message=f'Are you sure you want to delete app "{app_name}""?', default=False)]
        if inquirer.prompt(ask, theme=GreenPassion(), raise_keyboard_interrupt=True)['confirm'] is False:
            console.print('[b][red]Aborted![/b][/red]')
            raise InterruptedError
    except KeyboardInterrupt:
        console.print('[b][red]Cancelled by user![/b][/red]')
        raise InterruptedError

@delete.command('app')
@click.argument('app-name', type=str)
@click.option('skip_user_confirm_prompt', '--yes', '-y', is_flag=True, default=False, help='Do not prompt for confirmation.')
def delete_app(app_name: str, skip_user_confirm_prompt: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Delete a Lightning app.\n\n    Deleting an app also deletes all app websites, works, artifacts, and logs. This permanently removes any record of\n    the app as well as all any of its associated resources and data. This does not affect any resources and data\n    associated with other Lightning apps on your account.\n\n    '
    console = Console()
    try:
        selected_app_instance_id = _find_selected_app_instance_id(app_name=app_name)
        if not skip_user_confirm_prompt:
            _delete_app_confirmation_prompt(app_name=app_name)
    except InterruptedError:
        return
    try:
        app_manager = _AppManager()
        app_manager.delete(app_id=selected_app_instance_id)
    except Exception as ex:
        console.print(f'[b][red]An issue occurred while deleting app "{app_name}. If the issue persists, please reach out to us at [link=mailto:support@lightning.ai]support@lightning.ai[/link][/b][/red].')
        raise click.ClickException(str(ex))
    console.print(f'[b][green]App "{app_name}" has been successfully deleted"![/green][/b]')
    return