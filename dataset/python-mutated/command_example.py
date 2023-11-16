import typer
app = typer.Typer()

@app.command()
def greeting(name: str):
    if False:
        for i in range(10):
            print('nop')
    'Say hello to users'
    print(f'Hello {name}!')

@app.command()
def say_bye(name: str):
    if False:
        i = 10
        return i + 15
    'Say bye to users'
    print(f'Good bye {name}')
if __name__ == '__main__':
    app()