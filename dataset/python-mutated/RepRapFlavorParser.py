from . import FlavorParser

class RepRapFlavorParser(FlavorParser.FlavorParser):
    """This parser is intended to interpret the RepRap Firmware g-code flavor."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def processMCode(self, M, line, position, path):
        if False:
            for i in range(10):
                print('nop')
        if M == 82:
            self._is_absolute_extrusion = True
        elif M == 83:
            self._is_absolute_extrusion = False

    def _gCode90(self, position, params, path):
        if False:
            print('Hello World!')
        'Set the absolute positioning\n\n        RepRapFlavor code G90 sets position of X, Y, Z to absolute\n        For absolute E, M82 is used\n        '
        self._is_absolute_positioning = True
        return position

    def _gCode91(self, position, params, path):
        if False:
            while True:
                i = 10
        'Set the relative positioning\n\n        RepRapFlavor code G91 sets position of X, Y, Z to relative\n        For relative E, M83 is used\n        '
        self._is_absolute_positioning = False
        return position