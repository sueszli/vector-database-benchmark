import hug

@hug.get('/birthday')
def home(name: str):
    if False:
        for i in range(10):
            print('nop')
    return 'Happy Birthday, {name}'.format(name=name)