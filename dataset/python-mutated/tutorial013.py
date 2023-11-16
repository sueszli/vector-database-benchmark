from typing_extensions import Annotated

def say_hello(name: Annotated[str, 'this is just metadata']) -> str:
    if False:
        return 10
    return f'Hello {name}'