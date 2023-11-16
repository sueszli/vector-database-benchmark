"""
Implementation of the state pattern

http://ginstrom.com/scribbles/2007/10/08/design-patterns-python-style/

*TL;DR
Implements state as a derived class of the state pattern interface.
Implements state transitions by invoking methods from the pattern's superclass.
"""
from __future__ import annotations

class State:
    """Base state. This is to share functionality"""

    def scan(self) -> None:
        if False:
            i = 10
            return i + 15
        'Scan the dial to the next station'
        self.pos += 1
        if self.pos == len(self.stations):
            self.pos = 0
        print(f'Scanning... Station is {self.stations[self.pos]} {self.name}')

class AmState(State):

    def __init__(self, radio: Radio) -> None:
        if False:
            while True:
                i = 10
        self.radio = radio
        self.stations = ['1250', '1380', '1510']
        self.pos = 0
        self.name = 'AM'

    def toggle_amfm(self) -> None:
        if False:
            return 10
        print('Switching to FM')
        self.radio.state = self.radio.fmstate

class FmState(State):

    def __init__(self, radio: Radio) -> None:
        if False:
            while True:
                i = 10
        self.radio = radio
        self.stations = ['81.3', '89.1', '103.9']
        self.pos = 0
        self.name = 'FM'

    def toggle_amfm(self) -> None:
        if False:
            print('Hello World!')
        print('Switching to AM')
        self.radio.state = self.radio.amstate

class Radio:
    """A radio.     It has a scan button, and an AM/FM toggle switch."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        'We have an AM state and an FM state'
        self.amstate = AmState(self)
        self.fmstate = FmState(self)
        self.state = self.amstate

    def toggle_amfm(self) -> None:
        if False:
            return 10
        self.state.toggle_amfm()

    def scan(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.state.scan()

def main():
    if False:
        print('Hello World!')
    '\n    >>> radio = Radio()\n    >>> actions = [radio.scan] * 2 + [radio.toggle_amfm] + [radio.scan] * 2\n    >>> actions *= 2\n\n    >>> for action in actions:\n    ...    action()\n    Scanning... Station is 1380 AM\n    Scanning... Station is 1510 AM\n    Switching to FM\n    Scanning... Station is 89.1 FM\n    Scanning... Station is 103.9 FM\n    Scanning... Station is 81.3 FM\n    Scanning... Station is 89.1 FM\n    Switching to AM\n    Scanning... Station is 1250 AM\n    Scanning... Station is 1380 AM\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()