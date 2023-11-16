"""
Texture definition file.
"""
from enum import Enum
from ..data_definition import DataDefinition
FORMAT_VERSION = '1'

class LayerMode(Enum):
    """
    Possible values for the mode of a layer.
    """
    OFF = 'off'
    ONCE = 'once'
    LOOP = 'loop'

class TextureMetadata(DataDefinition):
    """
    Collects texture metadata and can format it
    as a .texture custom format
    """

    def __init__(self, targetdir, filename):
        if False:
            while True:
                i = 10
        super().__init__(targetdir, filename)
        self.image_file = None
        self.size = {}
        self.pxformat = {}
        self.subtexs = []

    def add_subtex(self, xpos, ypos, xsize, ysize, xhotspot, yhotspot):
        if False:
            return 10
        '\n        Add a subtex with all its spacial information.\n\n        :param xpos: X position of the subtex on the image canvas.\n        :type xpos: int\n        :param ypos: Y position of the subtex on the image canvas.\n        :type ypos: int\n        :param xsize: Width of the subtex.\n        :type xsize: int\n        :param ysize: Height of the subtex.\n        :type ysize: int\n        :param xhotspot: X position of the hotspot of the subtex.\n        :type xhotspot: int\n        :param yhotspot: Y position of the hotspot of the subtex.\n        :type yhotspot: int\n        '
        self.subtexs.append({'xpos': xpos, 'ypos': ypos, 'xsize': xsize, 'ysize': ysize, 'xhotspot': xhotspot, 'yhotspot': yhotspot})

    def set_imagefile(self, filename):
        if False:
            return 10
        '\n        Set the relative filename of the texture.\n\n        :param filename: Path to the image file.\n        :type filename: str\n        '
        self.image_file = filename

    def set_size(self, width, height):
        if False:
            while True:
                i = 10
        '\n        Define the size of the PNG file.\n\n        :param width: Width of the exported PNG in pixels.\n        :type width: int\n        :param height: Height of the exported PNG in pixels.\n        :type height: int\n        '
        self.size = {'width': width, 'height': height}

    def set_pxformat(self, pxformat='rgba8', cbits=True):
        if False:
            i = 10
            return i + 15
        '\n        Specify the pixel format of the texture.\n\n        :param pxformat: Identifier for the pixel format of each pixel.\n        :type pxformat: str\n        :param cbits: True if the pixels use a command bit.\n        :type cbits: bool\n        '
        self.pxformat = {'format': pxformat, 'cbits': cbits}

    def dump(self):
        if False:
            i = 10
            return i + 15
        output_str = ''
        output_str += '# openage texture definition file\n\n'
        output_str += f'version {FORMAT_VERSION}\n\n'
        output_str += f'imagefile "{self.image_file}"\n'
        output_str += '\n'
        output_str += f"size {self.size['width']} {self.size['height']}\n"
        output_str += '\n'
        output_str += f"pxformat {self.pxformat['format']}"
        if self.pxformat['cbits']:
            output_str += f" cbits={self.pxformat['cbits']}"
        output_str += '\n\n'
        for subtex in self.subtexs:
            output_str += f"subtex {' '.join((str(param) for param in subtex.values()))}\n"
        return output_str

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'TextureMetadata<{self.filename}>'