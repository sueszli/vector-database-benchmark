"""
Example from https://en.wikipedia.org/wiki/Facade_pattern#Python


*What is this pattern about?
The Facade pattern is a way to provide a simpler unified interface to
a more complex system. It provides an easier way to access functions
of the underlying system by providing a single entry point.
This kind of abstraction is seen in many real life situations. For
example, we can turn on a computer by just pressing a button, but in
fact there are many procedures and operations done when that happens
(e.g., loading programs from disk to memory). In this case, the button
serves as an unified interface to all the underlying procedures to
turn on a computer.

*Where is the pattern used practically?
This pattern can be seen in the Python standard library when we use
the isdir function. Although a user simply uses this function to know
whether a path refers to a directory, the system makes a few
operations and calls other modules (e.g., os.stat) to give the result.

*References:
https://sourcemaking.com/design_patterns/facade
https://fkromer.github.io/python-pattern-references/design/#facade
http://python-3-patterns-idioms-test.readthedocs.io/en/latest/ChangeInterface.html#facade

*TL;DR
Provides a simpler unified interface to a complex system.
"""

class CPU:
    """
    Simple CPU representation.
    """

    def freeze(self) -> None:
        if False:
            while True:
                i = 10
        print('Freezing processor.')

    def jump(self, position: str) -> None:
        if False:
            while True:
                i = 10
        print('Jumping to:', position)

    def execute(self) -> None:
        if False:
            return 10
        print('Executing.')

class Memory:
    """
    Simple memory representation.
    """

    def load(self, position: str, data: str) -> None:
        if False:
            print('Hello World!')
        print(f"Loading from {position} data: '{data}'.")

class SolidStateDrive:
    """
    Simple solid state drive representation.
    """

    def read(self, lba: str, size: str) -> str:
        if False:
            return 10
        return f'Some data from sector {lba} with size {size}'

class ComputerFacade:
    """
    Represents a facade for various computer parts.
    """

    def __init__(self):
        if False:
            return 10
        self.cpu = CPU()
        self.memory = Memory()
        self.ssd = SolidStateDrive()

    def start(self):
        if False:
            print('Hello World!')
        self.cpu.freeze()
        self.memory.load('0x00', self.ssd.read('100', '1024'))
        self.cpu.jump('0x00')
        self.cpu.execute()

def main():
    if False:
        i = 10
        return i + 15
    "\n    >>> computer_facade = ComputerFacade()\n    >>> computer_facade.start()\n    Freezing processor.\n    Loading from 0x00 data: 'Some data from sector 100 with size 1024'.\n    Jumping to: 0x00\n    Executing.\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)