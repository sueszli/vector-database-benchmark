from typing import List
import typer
from rich.console import Console
from typing_extensions import Annotated
valid_completion_items = [('Camila', 'The reader of books.'), ('Carlos', 'The writer of scripts.'), ('Sebastian', 'The type hints guy.')]
err_console = Console(stderr=True)

def complete_name(ctx: typer.Context, args: List[str], incomplete: str):
    if False:
        return 10
    err_console.print(f'{args}')
    names = ctx.params.get('name') or []
    for (name, help_text) in valid_completion_items:
        if name.startswith(incomplete) and name not in names:
            yield (name, help_text)
app = typer.Typer()

@app.command()
def main(name: Annotated[List[str], typer.Option(help='The name to say hi to.', autocompletion=complete_name)]=['World']):
    if False:
        for i in range(10):
            print('nop')
    for n in name:
        print(f'Hello {n}')
if __name__ == '__main__':
    app()