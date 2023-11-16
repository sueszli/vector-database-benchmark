from typing import NamedTuple

class Stuff(NamedTuple):
    x: int

def main() -> None:
    if False:
        print('Hello World!')
    a_list = Stuff(5)
    print(a_list)