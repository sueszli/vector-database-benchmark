from typing import Union

def say_hi(name: Union[str, None]=None):
    if False:
        return 10
    if name is not None:
        print(f'Hey {name}!')
    else:
        print('Hello World')