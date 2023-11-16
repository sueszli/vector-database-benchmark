"""
The Game Boy has two tile maps, which defines what is rendered on the screen.
"""
import numpy as np
from pyboy.core.lcd import LCDCRegister
from .constants import HIGH_TILEMAP, LCDC_OFFSET, LOW_TILEDATA_NTILES, LOW_TILEMAP
from .tile import Tile

class TileMap:

    def __init__(self, mb, select):
        if False:
            i = 10
            return i + 15
        '\n        The Game Boy has two tile maps, which defines what is rendered on the screen. These are also referred to as\n        "background" and "window".\n\n        Use `pyboy.botsupport.BotSupportManager.tilemap_background` and\n        `pyboy.botsupport.BotSupportManager.tilemap_window` to instantiate this object.\n\n        This object defines `__getitem__`, which means it can be accessed with the square brackets to get a tile\n        identifier at a given coordinate.\n\n        Example:\n        ```\n        >>> tilemap = pyboy.tilemap_window\n        >>> tile = tilemap[10,10]\n        >>> print(tile)\n        34\n        >>> print(tilemap[0:10,10])\n        [43, 54, 23, 23, 23, 54, 12, 54, 54, 23]\n        >>> print(tilemap[0:10,0:4])\n        [[43, 54, 23, 23, 23, 54, 12, 54, 54, 23],\n         [43, 54, 43, 23, 23, 43, 12, 39, 54, 23],\n         [43, 54, 23, 12, 87, 54, 12, 54, 21, 23],\n         [43, 54, 23, 43, 23, 87, 12, 50, 54, 72]]\n        ```\n\n        Each element in the matrix, is the tile identifier of the tile to be shown on screen for each position. If you\n        need the entire 32x32 tile map, you can use the shortcut: `tilemap[:,:]`.\n        '
        self.mb = mb
        self._select = select
        self._use_tile_objects = False
        self.refresh_lcdc()
        self.shape = (32, 32)
        '\n        Tile maps are always 32x32 tiles.\n\n        Returns\n        -------\n        (int, int):\n            The width and height of the tile map.\n        '

    def refresh_lcdc(self):
        if False:
            return 10
        '\n        The tile data and view that is showed on the background and window respectively can change dynamically. If you\n        believe it has changed, you can use this method to update the tilemap from the LCDC register.\n        '
        LCDC = LCDCRegister(self.mb.getitem(LCDC_OFFSET))
        if self._select == 'WINDOW':
            self.map_offset = HIGH_TILEMAP if LCDC.windowmap_select else LOW_TILEMAP
            self.signed_tile_data = not bool(LCDC.tiledata_select)
        elif self._select == 'BACKGROUND':
            self.map_offset = HIGH_TILEMAP if LCDC.backgroundmap_select else LOW_TILEMAP
            self.signed_tile_data = not bool(LCDC.tiledata_select)
        else:
            raise KeyError(f'Invalid tilemap selected: {self._select}')

    def search_for_identifiers(self, identifiers):
        if False:
            return 10
        '\n        Provided a list of tile identifiers, this function will find all occurrences of these in the tilemap and return\n        the coordinates where each identifier is found.\n\n        Example:\n        ```\n        >>> tilemap = pyboy.tilemap_window\n        >>> print(tilemap.search_for_identifiers([43, 123]))\n        [[[0,0], [2,4], [8,7]], []]\n        ```\n\n        Meaning, that tile identifier `43` is found at the positions: (0,0), (2,4), and (8,7), while tile identifier\n        `123`was not found anywhere.\n\n        Args:\n            identifiers (list): List of tile identifiers (int)\n\n        Returns\n        -------\n        list:\n            list of matches for every tile identifier in the input\n        '
        tilemap_identifiers = np.asarray(self[:, :], dtype=np.uint32)
        matches = []
        for i in identifiers:
            matches.append([[int(y) for y in x] for x in np.argwhere(tilemap_identifiers == i)])
        return matches

    def _tile_address(self, column, row):
        if False:
            while True:
                i = 10
        '\n        Returns the memory address in the tilemap for the tile at the given coordinate. The address contains the index\n        of tile which will be shown at this position. This should not be confused with the actual tile data of\n        `pyboy.botsupport.tile.Tile.data_address`.\n\n        This can be used as an global identifier for the specific location in a tile map.\n\n        Be aware, that the tile index referenced at the memory address might change between calls to\n        `pyboy.PyBoy.tick`. And the tile data for the same tile index might also change to display something else\n        on the screen.\n\n        The index might also be a signed number. Depending on if it is signed or not, will change where the tile data\n        is read from. Use `pyboy.botsupport.tilemap.TileMap.signed_tile_index` to test if the indexes are signed for\n        this tile view. You can read how the indexes work in the\n        [Pan Docs: VRAM Tile Data](http://bgb.bircd.org/pandocs.htm#vramtiledata).\n\n        Args:\n            column (int): Column in this tile map.\n            row (int): Row in this tile map.\n\n        Returns\n        -------\n        int:\n            Address in the tile map to read a tile index.\n        '
        if not 0 <= column < 32:
            raise IndexError('column is out of bounds. Value of 0 to 31 is allowed')
        if not 0 <= row < 32:
            raise IndexError('row is out of bounds. Value of 0 to 31 is allowed')
        return self.map_offset + 32 * row + column

    def tile(self, column, row):
        if False:
            while True:
                i = 10
        '\n        Provides a `pyboy.botsupport.tile.Tile`-object which allows for easy interpretation of the tile data. The\n        object is agnostic to where it was found in the tilemap. I.e. equal `pyboy.botsupport.tile.Tile`-objects might\n        be returned from two different coordinates in the tile map if they are shown different places on the screen.\n\n        Args:\n            column (int): Column in this tile map.\n            row (int): Row in this tile map.\n\n        Returns\n        -------\n        `pyboy.botsupport.tile.Tile`:\n            Tile object corresponding to the tile index at the given coordinate in the\n            tile map.\n        '
        return Tile(self.mb, self.tile_identifier(column, row))

    def tile_identifier(self, column, row):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an identifier (integer) of the tile at the given coordinate in the tile map. The identifier can be used\n        to quickly recognize what is on the screen through this tile view.\n\n        This identifier unifies the otherwise complicated indexing system on the Game Boy into a single range of\n        0-383 (both included).\n\n        You can read how the indexes work in the\n        [Pan Docs: VRAM Tile Data](http://bgb.bircd.org/pandocs.htm#vramtiledata).\n\n        Args:\n            column (int): Column in this tile map.\n            row (int): Row in this tile map.\n\n        Returns\n        -------\n        int:\n            Tile identifier.\n        '
        tile = self.mb.getitem(self._tile_address(column, row))
        if self.signed_tile_data:
            return (tile ^ 128) - 128 + LOW_TILEDATA_NTILES
        else:
            return tile

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        adjust = 4
        _use_tile_objects = self._use_tile_objects
        self.use_tile_objects(False)
        return_data = f'Tile Map Address: {self.map_offset:#0{6}x}, ' + f"Signed Tile Data: {('Yes' if self.signed_tile_data else 'No')}\n" + ' ' * 5 + ''.join([f'{i: <4}' for i in range(32)]) + '\n' + '_' * (adjust * 32 + 2) + '\n' + '\n'.join([f'{i: <3}| ' + ''.join([str(tile).ljust(adjust) for tile in line]) for (i, line) in enumerate(self[:, :])])
        self.use_tile_objects(_use_tile_objects)
        return return_data

    def use_tile_objects(self, switch):
        if False:
            return 10
        '\n        Used to change which object is returned when using the ``__getitem__`` method (i.e. `tilemap[0,0]`).\n\n        Args:\n            switch (bool): If True, accesses will return `pyboy.botsupport.tile.Tile`-object. If False, accesses will\n                return an `int`.\n        '
        self._use_tile_objects = switch

    def __getitem__(self, xy):
        if False:
            while True:
                i = 10
        (x, y) = xy
        if x == slice(None):
            x = slice(0, 32, 1)
        if y == slice(None):
            y = slice(0, 32, 1)
        x_slice = isinstance(x, slice)
        y_slice = isinstance(y, slice)
        assert x_slice or isinstance(x, int)
        assert y_slice or isinstance(y, int)
        if self._use_tile_objects:
            tile_fun = self.tile
        else:
            tile_fun = lambda x, y: self.tile_identifier(x, y)
        if x_slice and y_slice:
            return [[tile_fun(_x, _y) for _x in range(x.stop)[x]] for _y in range(y.stop)[y]]
        elif x_slice:
            return [tile_fun(_x, y) for _x in range(x.stop)[x]]
        elif y_slice:
            return [tile_fun(x, _y) for _y in range(y.stop)[y]]
        else:
            return tile_fun(x, y)