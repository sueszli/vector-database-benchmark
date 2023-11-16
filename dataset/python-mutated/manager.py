from . import constants as _constants
from . import screen as _screen
from . import sprite as _sprite
from . import tile as _tile
from . import tilemap as _tilemap
try:
    from cython import compiled
    cythonmode = compiled
except ImportError:
    cythonmode = False

class BotSupportManager:

    def __init__(self, pyboy, mb):
        if False:
            i = 10
            return i + 15
        if not cythonmode:
            self.pyboy = pyboy
            self.mb = mb

    def __cinit__(self, pyboy, mb):
        if False:
            while True:
                i = 10
        self.pyboy = pyboy
        self.mb = mb

    def screen(self):
        if False:
            while True:
                i = 10
        "\n        Use this method to get a `pyboy.botsupport.screen.Screen` object. This can be used to get the screen buffer in\n        a variety of formats.\n\n        It's also here you can find the screen position (SCX, SCY, WX, WY) for each scan line in the screen buffer. See\n        `pyboy.botsupport.screen.Screen.tilemap_position` for more information.\n\n        Returns\n        -------\n        `pyboy.botsupport.screen.Screen`:\n            A Screen object with helper functions for reading the screen buffer.\n        "
        return _screen.Screen(self.mb)

    def sprite(self, sprite_index):
        if False:
            print('Hello World!')
        '\n        Provides a `pyboy.botsupport.sprite.Sprite` object, which makes the OAM data more presentable. The given index\n        corresponds to index of the sprite in the "Object Attribute Memory" (OAM).\n\n        The Game Boy supports 40 sprites in total. Read more details about it, in the [Pan\n        Docs](http://bgb.bircd.org/pandocs.htm).\n\n        Args:\n            index (int): Sprite index from 0 to 39.\n        Returns\n        -------\n        `pyboy.botsupport.sprite.Sprite`:\n            Sprite corresponding to the given index.\n        '
        return _sprite.Sprite(self.mb, sprite_index)

    def sprite_by_tile_identifier(self, tile_identifiers, on_screen=True):
        if False:
            print('Hello World!')
        '\n        Provided a list of tile identifiers, this function will find all occurrences of sprites using the tile\n        identifiers and return the sprite indexes where each identifier is found. Use the sprite indexes in the\n        `pyboy.botsupport.BotSupportManager.sprite` function to get a `pyboy.botsupport.sprite.Sprite` object.\n\n        Example:\n        ```\n        >>> print(pyboy.botsupport_manager().sprite_by_tile_identifier([43, 123]))\n        [[0, 2, 4], []]\n        ```\n\n        Meaning, that tile identifier `43` is found at the sprite indexes: 0, 2, and 4, while tile identifier\n        `123` was not found anywhere.\n\n        Args:\n            identifiers (list): List of tile identifiers (int)\n            on_screen (bool): Require that the matched sprite is on screen\n\n        Returns\n        -------\n        list:\n            list of sprite matches for every tile identifier in the input\n        '
        matches = []
        for i in tile_identifiers:
            match = []
            for s in range(_constants.SPRITES):
                sprite = _sprite.Sprite(self.mb, s)
                for t in sprite.tiles:
                    if t.tile_identifier == i and (not on_screen or (on_screen and sprite.on_screen)):
                        match.append(s)
            matches.append(match)
        return matches

    def tile(self, identifier):
        if False:
            while True:
                i = 10
        '\n        The Game Boy can have 384 tiles loaded in memory at once. Use this method to get a\n        `pyboy.botsupport.tile.Tile`-object for given identifier.\n\n        The identifier is a PyBoy construct, which unifies two different scopes of indexes in the Game Boy hardware. See\n        the `pyboy.botsupport.tile.Tile` object for more information.\n\n        Returns\n        -------\n        `pyboy.botsupport.tile.Tile`:\n            A Tile object for the given identifier.\n        '
        return _tile.Tile(self.mb, identifier=identifier)

    def tilemap_background(self):
        if False:
            while True:
                i = 10
        '\n        The Game Boy uses two tile maps at the same time to draw graphics on the screen. This method will provide one\n        for the _background_ tiles. The game chooses whether it wants to use the low or the high tilemap.\n\n        Read more details about it, in the [Pan Docs](http://bgb.bircd.org/pandocs.htm#vrambackgroundmaps).\n\n        Returns\n        -------\n        `pyboy.botsupport.tilemap.TileMap`:\n            A TileMap object for the tile map.\n        '
        return _tilemap.TileMap(self.mb, 'BACKGROUND')

    def tilemap_window(self):
        if False:
            i = 10
            return i + 15
        '\n        The Game Boy uses two tile maps at the same time to draw graphics on the screen. This method will provide one\n        for the _window_ tiles. The game chooses whether it wants to use the low or the high tilemap.\n\n        Read more details about it, in the [Pan Docs](http://bgb.bircd.org/pandocs.htm#vrambackgroundmaps).\n\n        Returns\n        -------\n        `pyboy.botsupport.tilemap.TileMap`:\n            A TileMap object for the tile map.\n        '
        return _tilemap.TileMap(self.mb, 'WINDOW')