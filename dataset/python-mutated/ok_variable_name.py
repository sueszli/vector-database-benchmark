import typing
IRRELEVANT = typing.TypeVar

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    List: list[str] = []
    List.append('hello')