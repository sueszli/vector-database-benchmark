import hug

@hug.get()
@hug.cli()
def made_up_go():
    if False:
        return 10
    return 'Going!'