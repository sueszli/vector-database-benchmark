"""
https://www.djangospin.com/design-patterns-python/mediator/

Objects in a system communicate through a Mediator instead of directly with each other.
This reduces the dependencies between communicating objects, thereby reducing coupling.

*TL;DR
Encapsulates how a set of objects interact.
"""
from __future__ import annotations

class ChatRoom:
    """Mediator class"""

    def display_message(self, user: User, message: str) -> None:
        if False:
            print('Hello World!')
        print(f'[{user} says]: {message}')

class User:
    """A class whose instances want to interact with each other"""

    def __init__(self, name: str) -> None:
        if False:
            return 10
        self.name = name
        self.chat_room = ChatRoom()

    def say(self, message: str) -> None:
        if False:
            i = 10
            return i + 15
        self.chat_room.display_message(self, message)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.name

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> molly = User(\'Molly\')\n    >>> mark = User(\'Mark\')\n    >>> ethan = User(\'Ethan\')\n\n    >>> molly.say("Hi Team! Meeting at 3 PM today.")\n    [Molly says]: Hi Team! Meeting at 3 PM today.\n    >>> mark.say("Roger that!")\n    [Mark says]: Roger that!\n    >>> ethan.say("Alright.")\n    [Ethan says]: Alright.\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()