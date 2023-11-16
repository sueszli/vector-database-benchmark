import typer
app = typer.Typer()

@app.command()
def conquer(name: str):
    if False:
        while True:
            i = 10
    print(f'Conquering reign: {name}')

@app.command()
def destroy(name: str):
    if False:
        for i in range(10):
            print('nop')
    print(f'Destroying reign: {name}')
if __name__ == '__main__':
    app()