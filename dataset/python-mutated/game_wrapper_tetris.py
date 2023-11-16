__pdoc__ = {'GameWrapperTetris.cartridge_title': False, 'GameWrapperTetris.post_tick': False}
import logging
from array import array
import numpy as np
from pyboy.utils import WindowEvent
from .base_plugin import PyBoyGameWrapper
logger = logging.getLogger(__name__)
tetromino_table = {'L': 0, 'J': 4, 'I': 8, 'O': 12, 'Z': 16, 'S': 20, 'T': 24}
inverse_tetromino_table = {v: k for (k, v) in tetromino_table.items()}
NEXT_TETROMINO_ADDR = 49683
TILES = 384
tiles_compressed = np.zeros(TILES, dtype=np.uint8)
tiles_types = [[47], [129], [130], [131], [132], [133], [134], [128, 136, 137, 138, 139, 143], [135]]
for (tiles_type_ID, tiles_type) in enumerate(tiles_types):
    for tile_ID in tiles_type:
        tiles_compressed[tile_ID] = tiles_type_ID
tiles_minimal = np.ones(TILES, dtype=np.uint8)
tiles_minimal[47] = 0
tiles_minimal[135] = 2

class GameWrapperTetris(PyBoyGameWrapper):
    """
    This class wraps Tetris, and provides easy access to score, lines, level and a "fitness" score for AIs.

    If you call `print` on an instance of this object, it will show an overview of everything this object provides.
    """
    cartridge_title = 'TETRIS'
    tiles_compressed = tiles_compressed
    tiles_minimal = tiles_minimal

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.shape = (10, 18)
        'The shape of the game area'
        self.score = 0
        'The score provided by the game'
        self.level = 0
        'The current level'
        self.lines = 0
        'The number of cleared lines'
        self.fitness = 0
        '\n        A built-in fitness scoring. The scoring is equals to `score`.\n\n        .. math::\n            fitness = score\n        '
        super().__init__(*args, **kwargs)
        (ROWS, COLS) = self.shape
        self._cached_game_area_tiles_raw = array('B', [255] * (ROWS * COLS * 4))
        self._cached_game_area_tiles = memoryview(self._cached_game_area_tiles_raw).cast('I', shape=(ROWS, COLS))
        super().__init__(*args, game_area_section=(2, 0) + self.shape, game_area_wrap_around=True, **kwargs)

    def _game_area_tiles(self):
        if False:
            while True:
                i = 10
        if self._tile_cache_invalid:
            self._cached_game_area_tiles = np.asarray(self.tilemap_background[2:12, :18], dtype=np.uint32)
            self._tile_cache_invalid = False
        return self._cached_game_area_tiles

    def post_tick(self):
        if False:
            while True:
                i = 10
        self._tile_cache_invalid = True
        self._sprite_cache_invalid = True
        blank = 47
        self.tilemap_background.refresh_lcdc()
        self.score = self._sum_number_on_screen(13, 3, 6, blank, 0)
        self.level = self._sum_number_on_screen(14, 7, 4, blank, 0)
        self.lines = self._sum_number_on_screen(14, 10, 4, blank, 0)
        if self.game_has_started:
            self.fitness = self.score

    def start_game(self, timer_div=None):
        if False:
            i = 10
            return i + 15
        "\n        Call this function right after initializing PyBoy. This will navigate through menus to start the game at the\n        first playable state.\n\n        The state of the emulator is saved, and using `reset_game`, you can get back to this point of the game\n        instantly.\n\n        Kwargs:\n            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.\n        "
        PyBoyGameWrapper.start_game(self)
        while True:
            self.pyboy.tick()
            self.tilemap_background.refresh_lcdc()
            if self.tilemap_background[2:9, 14] == [89, 25, 21, 10, 34, 14, 27]:
                break
        for i in range(2):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            for _ in range(6):
                self.pyboy.tick()
        self.saved_state.seek(0)
        self.pyboy.save_state(self.saved_state)
        self.game_has_started = True
        self.reset_game(timer_div=timer_div)

    def reset_game(self, timer_div=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        After calling `start_game`, you can call this method at any time to reset the game.\n\n        Kwargs:\n            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.\n        "
        PyBoyGameWrapper.reset_game(self, timer_div=timer_div)
        self._set_timer_div(timer_div)
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for _ in range(6):
            self.pyboy.tick()

    def game_area(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Use this method to get a matrix of the "game area" of the screen. This view is simplified to be perfect for\n        machine learning applications.\n\n        In Tetris, this is only the part of the screen where the "tetrominoes" are placed.\n        The score, lines cleared, and level can be found in the variables of this class.\n\n        ```text\n             0   1   2   3   4   5   6   7   8   9\n        ____________________________________________\n        0  | 47  47  47  47  47  47  47  47  47  47\n        1  | 47  47  47  47  47  47  47  47  47  47\n        2  | 47  47  47  47  47  47  47  132 132 132\n        3  | 47  47  47  47  47  47  47  132 47  47\n        4  | 47  47  47  47  47  47  47  47  47  47\n        5  | 47  47  47  47  47  47  47  47  47  47\n        6  | 47  47  47  47  47  47  47  47  47  47\n        7  | 47  47  47  47  47  47  47  47  47  47\n        8  | 47  47  47  47  47  47  47  47  47  47\n        9  | 47  47  47  47  47  47  47  47  47  47\n        10 | 47  47  47  47  47  47  47  47  47  47\n        11 | 47  47  47  47  47  47  47  47  47  47\n        12 | 47  47  47  47  47  47  47  47  47  47\n        13 | 47  47  47  47  47  47  47  47  47  47\n        14 | 47  47  47  47  47  47  47  47  47  47\n        15 | 47  47  47  47  47  47  47  47  47  47\n        16 | 47  47  47  47  47  47  47  47  47  47\n        17 | 47  47  47  47  47  47  138 139 139 143\n        ```\n\n        Returns\n        -------\n        memoryview:\n            Simplified 2-dimensional memoryview of the screen\n        '
        return PyBoyGameWrapper.game_area(self)

    def next_tetromino(self):
        if False:
            print('Hello World!')
        '\n        Returns the next Tetromino to drop.\n\n        __NOTE:__ Don\'t use this function together with\n        `pyboy.plugins.game_wrapper_tetris.GameWrapperTetris.set_tetromino`.\n\n        Returns\n        -------\n        shape:\n            `str` of which Tetromino will drop:\n            * `"L"`: L-shape\n            * `"J"`: reverse L-shape\n            * `"I"`: I-shape\n            * `"O"`: square-shape\n            * `"Z"`: zig-zag left to right\n            * `"S"`: zig-zag right to left\n            * `"T"`: T-shape\n        '
        return inverse_tetromino_table[self.pyboy.get_memory_value(NEXT_TETROMINO_ADDR) & 252]

    def set_tetromino(self, shape):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function patches the random Tetromino routine in the ROM to output any given Tetromino instead.\n\n        __NOTE__: Any changes here are not saved or loaded to game states! Use this function with caution and reapply\n        any overrides when reloading the ROM. This also applies to\n        `pyboy.plugins.game_wrapper_tetris.GameWrapperTetris.start_game` and\n        `pyboy.plugins.game_wrapper_tetris.GameWrapperTetris.reset_game`.\n\n        Args:\n            shape (str): Define which Tetromino to use:\n            * `"L"`: L-shape\n            * `"J"`: reverse L-shape\n            * `"I"`: I-shape\n            * `"O"`: square-shape\n            * `"Z"`: zig-zag left to right\n            * `"S"`: zig-zag right to left\n            * `"T"`: T-shape\n        '
        if shape not in tetromino_table:
            raise KeyError('Invalid Tetromino shape!')
        shape_number = tetromino_table[shape]
        patch1 = [62, shape_number, 0]
        for (i, byte) in enumerate(patch1):
            self.pyboy.override_memory_value(0, 8302 + i, byte)
        patch2 = [62, shape_number]
        for (i, byte) in enumerate(patch2):
            self.pyboy.override_memory_value(0, 8368 + i, byte)

    def game_over(self):
        if False:
            while True:
                i = 10
        '\n        After calling `start_game`, you can call this method at any time to know if the game is over.\n\n        Game over happens, when the game area is filled with Tetrominos without clearing any rows.\n        '
        return self.tilemap_background[2, 0] == 135

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        adjust = 4
        return f'Tetris:\n' + f'Score: {self.score}\n' + f'Level: {self.level}\n' + f'Lines: {self.lines}\n' + f'Fitness: {self.fitness}\n' + 'Sprites on screen:\n' + '\n'.join([str(s) for s in self._sprites_on_screen()]) + '\n' + 'Tiles on screen:\n' + ' ' * 5 + ''.join([f'{i: <4}' for i in range(10)]) + '\n' + '_' * (adjust * 10 + 4) + '\n' + '\n'.join([f'{i: <3}| ' + ''.join([str(tile).ljust(adjust) for tile in line]) for (i, line) in enumerate(self._game_area_np())])