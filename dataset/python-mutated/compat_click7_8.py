from typing import List
import click
import typer
app = typer.Typer()

def shell_complete(ctx: click.Context, param: click.Parameter, incomplete: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    return ['Jonny']

@app.command(context_settings={'auto_envvar_prefix': 'TEST'})
def main(name: str=typer.Option('John', hidden=True), lastname: str=typer.Option('Doe', '/lastname', show_default='Mr. Doe'), age: int=typer.Option(lambda : 42, show_default=True), nickname: str=typer.Option('', shell_complete=shell_complete)):
    if False:
        while True:
            i = 10
    '\n    Say hello.\n    '
    print(f'Hello {name} {lastname}, it seems you have {age}, {nickname}')
if __name__ == '__main__':
    app()