from __future__ import annotations

class Person:

    def __init__(self, name: str) -> None:
        if False:
            print('Hello World!')
        self.name = name

    def do_action(self, action: Action) -> Action:
        if False:
            i = 10
            return i + 15
        print(self.name, action.name, end=' ')
        return action

class Action:

    def __init__(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def amount(self, val: str) -> Action:
        if False:
            return 10
        print(val, end=' ')
        return self

    def stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        print('then stop')

def main():
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> move = Action('move')\n    >>> person = Person('Jack')\n    >>> person.do_action(move).amount('5m').stop()\n    Jack move 5m then stop\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()