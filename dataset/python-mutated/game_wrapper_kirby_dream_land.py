__pdoc__ = {'GameWrapperKirbyDreamLand.cartridge_title': False, 'GameWrapperKirbyDreamLand.post_tick': False}
import logging
from pyboy.utils import WindowEvent
from .base_plugin import PyBoyGameWrapper
logger = logging.getLogger(__name__)

class GameWrapperKirbyDreamLand(PyBoyGameWrapper):
    """
    This class wraps Kirby Dream Land, and provides easy access to score and a "fitness" score for AIs.

    If you call `print` on an instance of this object, it will show an overview of everything this object provides.
    """
    cartridge_title = 'KIRBY DREAM LA'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.shape = (20, 16)
        'The shape of the game area'
        self.score = 0
        'The score provided by the game'
        self.health = 0
        'The health provided by the game'
        self.lives_left = 0
        'The lives remaining provided by the game'
        self._game_over = False
        'The game over state'
        self.fitness = 0
        '\n        A built-in fitness scoring. Taking score, health, and lives left into account.\n\n        .. math::\n            fitness = score \\cdot health \\cdot lives\\_left\n        '
        super().__init__(*args, game_area_section=(0, 0) + self.shape, game_area_wrap_around=True, **kwargs)

    def post_tick(self):
        if False:
            print('Hello World!')
        self._tile_cache_invalid = True
        self._sprite_cache_invalid = True
        self.score = 0
        score_digits = 5
        for n in range(score_digits):
            self.score += self.pyboy.get_memory_value(53359 + n) * 10 ** (score_digits - n)
        prev_health = self.health
        self.health = self.pyboy.get_memory_value(53382)
        if self.lives_left == 0:
            if prev_health > 0 and self.health == 0:
                self._game_over = True
        self.lives_left = self.pyboy.get_memory_value(53385) - 1
        if self.game_has_started:
            self.fitness = self.score * self.health * self.lives_left

    def start_game(self, timer_div=None):
        if False:
            while True:
                i = 10
        "\n        Call this function right after initializing PyBoy. This will navigate through menus to start the game at the\n        first playable state.\n\n        The state of the emulator is saved, and using `reset_game`, you can get back to this point of the game\n        instantly.\n\n        Kwargs:\n            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.\n        "
        PyBoyGameWrapper.start_game(self, timer_div=timer_div)
        while True:
            self.pyboy.tick()
            self.tilemap_background.refresh_lcdc()
            if self.tilemap_background[0:3, 16] == [231, 224, 235]:
                break
        for _ in range(25):
            self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for _ in range(60):
            self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for _ in range(60):
            self.pyboy.tick()
        self.game_has_started = True
        self.saved_state.seek(0)
        self.pyboy.save_state(self.saved_state)
        self._set_timer_div(timer_div)

    def reset_game(self, timer_div=None):
        if False:
            i = 10
            return i + 15
        "\n        After calling `start_game`, you can call this method at any time to reset the game.\n\n        Kwargs:\n            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.\n        "
        PyBoyGameWrapper.reset_game(self, timer_div=timer_div)
        self._set_timer_div(timer_div)

    def game_area(self):
        if False:
            return 10
        '\n        Use this method to get a matrix of the "game area" of the screen.\n\n        ```text\n              0   1   2   3   4   5   6   7   8   9\n          ____________________________________________________________________________________\n          0  | 383 383 383 301 383 383 383 297 383 383 383 301 383 383 383 297 383 383 383 293\n          1  | 383 383 383 383 300 294 295 296 383 383 383 383 300 294 295 296 383 383 299 383\n          2  | 311 318 319 320 383 383 383 383 383 383 383 383 383 383 383 383 383 301 383 383\n          3  | 383 383 383 321 322 383 383 383 383 383 383 383 383 383 383 383 383 383 300 294\n          4  | 383 383 383 383 323 290 291 383 383 383 313 312 311 318 319 320 383 290 291 383\n          5  | 383 383 383 383 324 383 383 383 383 315 314 383 383 383 383 321 322 383 383 383\n          6  | 383 383 383 383 324 293 292 383 383 316 383 383 383 383 383 383 323 383 383 383\n          7  | 383 383 383 383 324 383 383 298 383 317 383 383 383 383 383 383 324 383 383 383\n          8  | 319 320 383 383 324 383 383 297 383 317 383 383 383 152 140 383 324 383 383 307\n          9  | 383 321 322 383 324 294 295 296 383 325 383 383 383 383 383 383 326 272 274 309\n          10 | 383 383 323 383 326 383 383 383 2   18  383 330 331 331 331 331 331 331 331 331\n          11 | 274 383 324 272 274 272 274 272 274 272 274 334 328 328 328 328 328 328 328 328\n          12 | 331 331 331 331 331 331 331 331 331 331 331 328 328 328 328 328 328 328 328 328\n          13 | 328 328 328 277 278 328 328 328 328 328 328 328 328 277 278 328 328 277 278 277\n          14 | 328 277 278 279 281 277 278 328 328 277 278 277 278 279 281 277 278 279 281 279\n          15 | 278 279 281 280 282 279 281 277 278 279 281 279 281 280 282 279 281 280 282 280\n        ```\n\n        Returns\n        -------\n        memoryview:\n            Simplified 2-dimensional memoryview of the screen\n        '
        return PyBoyGameWrapper.game_area(self)

    def game_over(self):
        if False:
            i = 10
            return i + 15
        return self._game_over

    def __repr__(self):
        if False:
            while True:
                i = 10
        adjust = 4
        return f'Kirby Dream Land:\n' + f'Score: {self.score}\n' + f'Health: {self.health}\n' + f'Lives left: {self.lives_left}\n' + f'Fitness: {self.fitness}\n' + 'Sprites on screen:\n' + '\n'.join([str(s) for s in self._sprites_on_screen()]) + '\n' + 'Tiles on screen:\n' + ' ' * 5 + ''.join([f'{i: <4}' for i in range(10)]) + '\n' + '_' * (adjust * 20 + 4) + '\n' + '\n'.join([f'{i: <3}| ' + ''.join([str(tile).ljust(adjust) for tile in line]) for (i, line) in enumerate(self.game_area())])