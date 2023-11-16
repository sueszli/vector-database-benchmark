import os

def Main():
    if False:
        i = 10
        return i + 15
    print(os.environ.get('Something', 'ERROR'))
    print(os.environ.get('SomethingElse', 'ERROR'))
    print(os.environ.get('PATH', 'ERROR'))
    for (k, v) in os.environ.items():
        print(f'{k} = "{v}"')
Main()