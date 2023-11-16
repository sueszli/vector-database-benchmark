"""
Palette definition file.
"""
from ..data_definition import DataDefinition
FORMAT_VERSION = '1'

class PaletteMetadata(DataDefinition):
    """
    Collects palette metadata and can format it
    as a .opal custom format
    """

    def __init__(self, targetdir: str, filename: str):
        if False:
            while True:
                i = 10
        super().__init__(targetdir, filename)
        self.colours: list[tuple] = []

    def add_colour(self, colour: tuple) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a RGBA colour to the end of the palette.\n\n        :param colour: RGBA colour tuple.\n        :type colour: tuple\n        '
        self.colours.append(colour)

    def add_colours(self, colours: list[tuple]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a collection of RGBA colours to the end of the palette.\n\n        :param colours: Collection of RGBA coulour tuples.\n        :type colours: tuple, list\n        '
        self.colours.extend(colours)

    def dump(self) -> str:
        if False:
            print('Hello World!')
        output_str = ''
        output_str += '# openage palette definition file\n\n'
        output_str += f'version {FORMAT_VERSION}\n\n'
        output_str += f'entries {len(self.colours)}\n\n'
        output_str += 'colours [\n'
        for colour in self.colours:
            output_str += f"{' '.join((str(param) for param in colour))}\n"
        output_str += ']\n'
        return output_str

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'PaletteMetadata<{self.filename}>'