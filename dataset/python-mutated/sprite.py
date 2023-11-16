"""
This class presents an interface to the sprites held in the OAM data on the Game Boy.
"""
from pyboy.core.lcd import LCDCRegister
from .constants import LCDC_OFFSET, OAM_OFFSET, SPRITES
from .tile import Tile

class Sprite:

    def __init__(self, mb, sprite_index):
        if False:
            return 10
        '\n        This class presents an interface to the sprites held in the OAM data on the Game Boy.\n\n        The purpose is to make it easier to interpret events on the screen, in order to program a bot, or train an AI.\n\n        Sprites are used on the Game Boy for enemy and player characters, as only sprites can have transparency, and can\n        move at pixel-precision on the screen. The other method of graphics -- tile maps -- can only be placed in a\n        grid-size of 8x8 pixels precision, and can have no transparency.\n\n        Sprites on the Game Boy are tightly associated with tiles. The sprites can be seen as "upgraded" tiles, as the\n        image data still refers back to one (or two) tiles. The tile that a sprite will show, can change between each\n        call to `pyboy.PyBoy.tick`, so make sure to verify the `Sprite.tile_identifier` hasn\'t changed.\n\n        By knowing the tile identifiers of players, enemies, power-ups and so on, you\'ll be able to search for them\n        using `pyboy.botsupport.BotSupportManager.sprite_by_tile_identifier` and feed it to your bot or AI.\n        '
        assert 0 <= sprite_index < SPRITES, f'Sprite index of {sprite_index} is out of range (0-{SPRITES})'
        self.mb = mb
        self._offset = sprite_index * 4
        self._sprite_index = sprite_index
        '\n        The index of the sprite itself. Beware, that this only represents the index or a "slot" in OAM memory.\n        Many games will change the image data of the sprite in the "slot" several times per second.\n\n        Returns\n        -------\n        int:\n            unsigned tile index\n        '
        self.y = self.mb.getitem(OAM_OFFSET + self._offset + 0) - 16
        '\n        The Y-coordinate on the screen to show the Sprite. The (x,y) coordinate points to the top-left corner of the sprite.\n\n        Returns\n        -------\n        int:\n            Y-coordinate\n        '
        self.x = self.mb.getitem(OAM_OFFSET + self._offset + 1) - 8
        '\n        The X-coordinate on the screen to show the Sprite. The (x,y) coordinate points to the top-left corner of the sprite.\n\n        Returns\n        -------\n        int:\n            X-coordinate\n        '
        self.tile_identifier = self.mb.getitem(OAM_OFFSET + self._offset + 2)
        '\n        The identifier of the tile the sprite uses. To get a better representation, see the method\n        `pyboy.botsupport.sprite.Sprite.tiles`.\n\n        For double-height sprites, this will only give the identifier of the first tile. The second tile will\n        always be the one immediately following the first (`tile_identifier + 1`).\n\n        Returns\n        -------\n        int:\n            unsigned tile index\n        '
        attr = self.mb.getitem(OAM_OFFSET + self._offset + 3)
        self.attr_obj_bg_priority = _bit(attr, 7)
        '\n        To better understand this values, look in the [Pan Docs: VRAM Sprite Attribute Table\n        (OAM)](http://bgb.bircd.org/pandocs.htm#vramspriteattributetableoam).\n\n        Returns\n        -------\n        bool:\n            The state of the bit in the attributes lookup.\n        '
        self.attr_y_flip = _bit(attr, 6)
        '\n        To better understand this values, look in the [Pan Docs: VRAM Sprite Attribute Table\n        (OAM)](http://bgb.bircd.org/pandocs.htm#vramspriteattributetableoam).\n\n        Returns\n        -------\n        bool:\n            The state of the bit in the attributes lookup.\n        '
        self.attr_x_flip = _bit(attr, 5)
        '\n        To better understand this values, look in the [Pan Docs: VRAM Sprite Attribute Table\n        (OAM)](http://bgb.bircd.org/pandocs.htm#vramspriteattributetableoam).\n\n        Returns\n        -------\n        bool:\n            The state of the bit in the attributes lookup.\n        '
        self.attr_palette_number = _bit(attr, 4)
        '\n        To better understand this values, look in the [Pan Docs: VRAM Sprite Attribute Table\n        (OAM)](http://bgb.bircd.org/pandocs.htm#vramspriteattributetableoam).\n\n        Returns\n        -------\n        bool:\n            The state of the bit in the attributes lookup.\n        '
        LCDC = LCDCRegister(self.mb.getitem(LCDC_OFFSET))
        sprite_height = 16 if LCDC.sprite_height else 8
        self.shape = (8, sprite_height)
        "\n        Sprites can be set to be 8x8 or 8x16 pixels (16 pixels tall). This is defined globally for the rendering\n        hardware, so it's either all sprites using 8x16 pixels, or all sprites using 8x8 pixels.\n\n        Returns\n        -------\n        (int, int):\n            The width and height of the sprite.\n        "
        self.tiles = [Tile(self.mb, self.tile_identifier)]
        '\n        The Game Boy support sprites of single-height (8x8 pixels) and double-height (8x16 pixels).\n\n        In the single-height format, one tile is used. For double-height sprites, the Game Boy will also use the tile\n        immediately following the identifier given, and render it below the first.\n\n        More information can be found in the [Pan Docs: VRAM Sprite Attribute Table\n        (OAM)](http://bgb.bircd.org/pandocs.htm#vramspriteattributetableoam)\n\n        Returns\n        -------\n        list:\n            A list of `pyboy.botsupport.tile.Tile` object(s) representing the graphics data for the sprite\n        '
        if sprite_height == 16:
            self.tiles += [Tile(self.mb, self.tile_identifier + 1)]
        self.on_screen = -sprite_height < self.y < 144 and -8 < self.x < 160
        "\n        To disable sprites from being rendered on screen, developers will place the sprite outside the area of the\n        screen. This is often a good way to determine if the sprite is inactive.\n\n        This check doesn't take transparency into account, and will only check the sprite's bounding-box of 8x8 or 8x16\n        pixels.\n\n        Returns\n        -------\n        bool:\n            True if the sprite has at least one pixel on screen.\n        "

    def __eq__(self, other):
        if False:
            return 10
        return self._offset == other._offset

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        tiles = ', '.join([str(t) for t in self.tiles])
        return f'Sprite [{self._sprite_index}]: Position: ({self.x}, {self.y}), Shape: {self.shape}, Tiles: ({tiles}), On screen: {self.on_screen}'

def _bit(val, bit):
    if False:
        for i in range(10):
            print('nop')
    return val >> bit & 1