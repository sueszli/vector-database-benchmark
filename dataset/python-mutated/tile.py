"""
The Game Boy uses tiles as the building block for all graphics on the screen. This base-class is used both for
`pyboy.botsupport.sprite.Sprite` and `pyboy.botsupport.tilemap.TileMap`, when refering to graphics.
"""
import logging
import numpy as np
from pyboy import utils
from .constants import LOW_TILEDATA, VRAM_OFFSET
logger = logging.getLogger(__name__)
try:
    from PIL import Image
except ImportError:
    Image = None

class Tile:

    def __init__(self, mb, identifier):
        if False:
            print('Hello World!')
        "\n        The Game Boy uses tiles as the building block for all graphics on the screen. This base-class is used for\n        `pyboy.botsupport.BotSupportManager.tile`, `pyboy.botsupport.sprite.Sprite` and `pyboy.botsupport.tilemap.TileMap`, when\n        refering to graphics.\n\n        This class is not meant to be instantiated by developers reading this documentation, but it will be created\n        internally and returned by `pyboy.botsupport.sprite.Sprite.tiles` and\n        `pyboy.botsupport.tilemap.TileMap.tile`.\n\n        The data of this class is static, apart from the image data, which is loaded from the Game Boy's memory when\n        needed. Beware that the graphics for the tile can change between each call to `pyboy.PyBoy.tick`.\n        "
        self.mb = mb
        assert 0 <= identifier < 384, 'Identifier out of range'
        self.data_address = LOW_TILEDATA + 16 * identifier
        '\n        The tile data is defined in a specific area of the Game Boy. This function returns the address of the tile data\n        corresponding to the tile identifier. It is advised to use `pyboy.botsupport.tile.Tile.image` or one of the\n        other `image`-functions if you want to view the tile.\n\n        You can read how the data is read in the\n        [Pan Docs: VRAM Tile Data](http://bgb.bircd.org/pandocs.htm#vramtiledata).\n\n        Returns\n        -------\n        int:\n            address in VRAM where tile data starts\n        '
        self.tile_identifier = (self.data_address - LOW_TILEDATA) // 16
        '\n        The Game Boy has a slightly complicated indexing system for tiles. This identifier unifies the otherwise\n        complicated indexing system on the Game Boy into a single range of 0-383 (both included).\n\n        Returns\n        -------\n        int:\n            Unique identifier for the tile\n        '
        self.shape = (8, 8)
        '\n        Tiles are always 8x8 pixels.\n\n        Returns\n        -------\n        (int, int):\n            The width and height of the tile.\n        '

    def image(self):
        if False:
            i = 10
            return i + 15
        '\n        Use this function to get an easy-to-use `PIL.Image` object of the tile. The image is 8x8 pixels in RGBA colors.\n\n        Be aware, that the graphics for this tile can change between each call to `pyboy.PyBoy.tick`.\n\n        Returns\n        -------\n        PIL.Image :\n            Image of tile in 8x8 pixels and RGBA colors.\n        '
        if Image is None:
            logger.error(f'{__name__}: Missing dependency "Pillow".')
            return None
        return Image.frombytes('RGBA', (8, 8), bytes(self.image_data()))

    def image_ndarray(self):
        if False:
            print('Hello World!')
        '\n        Use this function to get an easy-to-use `numpy.ndarray` object of the tile. The array has a shape of (8, 8, 4)\n        and each value is of `numpy.uint8`. The values corresponds to and RGBA image of 8x8 pixels with each sub-color\n        in a separate cell.\n\n        Be aware, that the graphics for this tile can change between each call to `pyboy.PyBoy.tick`.\n\n        Returns\n        -------\n        numpy.ndarray :\n            Array of shape (8, 8, 4) with data type of `numpy.uint8`.\n        '
        return np.asarray(self.image_data()).view(dtype=np.uint8).reshape(8, 8, 4)

    def image_data(self):
        if False:
            return 10
        '\n        Use this function to get the raw tile data. The data is a `memoryview` corresponding to 8x8 pixels in RGBA\n        colors.\n\n        Be aware, that the graphics for this tile can change between each call to `pyboy.PyBoy.tick`.\n\n        Returns\n        -------\n        memoryview :\n            Image data of tile in 8x8 pixels and RGBA colors.\n        '
        self.data = np.zeros((8, 8), dtype=np.uint32)
        for k in range(0, 16, 2):
            byte1 = self.mb.lcd.VRAM0[self.data_address + k - VRAM_OFFSET]
            byte2 = self.mb.lcd.VRAM0[self.data_address + k + 1 - VRAM_OFFSET]
            for x in range(8):
                colorcode = utils.color_code(byte1, byte2, 7 - x)
                old_A_format = 4278190080
                self.data[k // 2][x] = self.mb.lcd.BGP.getcolor(colorcode) >> 8 | old_A_format
        return self.data

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.data_address == other.data_address

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Tile: {self.tile_identifier}'