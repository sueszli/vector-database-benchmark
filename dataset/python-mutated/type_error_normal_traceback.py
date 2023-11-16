import typer
app = typer.Typer()

@app.command()
def main(name: str='morty'):
    if False:
        for i in range(10):
            print('nop')
    print(name)
broken_app = typer.Typer()

@broken_app.command()
def broken(name: str='morty'):
    if False:
        for i in range(10):
            print('nop')
    print(name + 3)
if __name__ == '__main__':
    app(standalone_mode=False)
    typer.main.get_command(broken_app)()