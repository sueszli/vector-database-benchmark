import typer
app = typer.Typer()

@app.command()
def found(name: str):
    if False:
        return 10
    print(f'Founding town: {name}')

@app.command()
def burn(name: str):
    if False:
        for i in range(10):
            print('nop')
    print(f'Burning town: {name}')
if __name__ == '__main__':
    app()