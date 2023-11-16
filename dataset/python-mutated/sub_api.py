import hug

@hug.cli()
def hello():
    if False:
        return 10
    return 'Hello world'