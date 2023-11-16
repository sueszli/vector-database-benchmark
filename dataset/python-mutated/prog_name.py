import typer
app = typer.Typer()

@app.command()
def main(i: int):
    if False:
        return 10
    pass
if __name__ == '__main__':
    app(prog_name='custom-name')