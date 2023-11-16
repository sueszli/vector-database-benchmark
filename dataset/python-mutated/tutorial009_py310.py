def say_hi(name: str | None=None):
    if False:
        print('Hello World!')
    if name is not None:
        print(f'Hey {name}!')
    else:
        print('Hello World')