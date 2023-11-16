from typing import Annotated

def say_hello(name: Annotated[str, 'this is just metadata']) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'Hello {name}'