"""
The core module of the emulator
"""
import logging
import os
import time
from pyboy.openai_gym import PyBoyGymEnv
from pyboy.openai_gym import enabled as gym_enabled
from pyboy.plugins.manager import PluginManager
from pyboy.utils import IntIOWrapper, WindowEvent
from . import botsupport
from .core.mb import Motherboard
logger = logging.getLogger(__name__)
SPF = 1 / 60.0
defaults = {'color_palette': (16777215, 10066329, 5592405, 0), 'cgb_color_palette': ((16777215, 8126257, 25541, 0), (16777215, 16745604, 9714234, 0), (16777215, 16745604, 9714234, 0)), 'scale': 3, 'window_type': 'SDL2'}

class PyBoy:

    def __init__(self, gamerom_file, *, bootrom_file=None, disable_renderer=False, sound=False, sound_emulated=False, cgb=None, randomize=False, **kwargs):
        if False:
            return 10
        '\n        PyBoy is loadable as an object in Python. This means, it can be initialized from another script, and be\n        controlled and probed by the script. It is supported to spawn multiple emulators, just instantiate the class\n        multiple times.\n\n        This object, `pyboy.WindowEvent`, and the `pyboy.botsupport` module, are the only official user-facing\n        interfaces. All other parts of the emulator, are subject to change.\n\n        A range of methods are exposed, which should allow for complete control of the emulator. Please open an issue on\n        GitHub, if other methods are needed for your projects. Take a look at the files in `examples/` for a crude\n        "bots", which interact with the game.\n\n        Only the `gamerom_file` argument is required.\n\n        Args:\n            gamerom_file (str): Filepath to a game-ROM for Game Boy or Game Boy Color.\n\n        Kwargs:\n            bootrom_file (str): Filepath to a boot-ROM to use. If unsure, specify `None`.\n            disable_renderer (bool): Can be used to optimize performance, by internally disable rendering of the screen.\n            color_palette (tuple): Specify the color palette to use for rendering.\n            cgb_color_palette (list of tuple): Specify the color palette to use for rendering in CGB-mode for non-color games.\n\n        Other keyword arguments may exist for plugins that are not listed here. They can be viewed with the\n        `parser_arguments()` method in the pyboy.plugins.manager module, or by running pyboy --help in the terminal.\n        '
        self.initialized = False
        for (k, v) in defaults.items():
            if k not in kwargs:
                kwargs[k] = kwargs.get(k, defaults[k])
        if not os.path.isfile(gamerom_file):
            raise FileNotFoundError(f'ROM file {gamerom_file} was not found!')
        self.gamerom_file = gamerom_file
        self.mb = Motherboard(gamerom_file, bootrom_file or kwargs.get('bootrom'), kwargs['color_palette'], kwargs['cgb_color_palette'], disable_renderer, sound, sound_emulated, cgb, randomize=randomize)
        self.avg_pre = 0
        self.avg_tick = 0
        self.avg_post = 0
        self.frame_count = 0
        self.set_emulation_speed(1)
        self.paused = False
        self.events = []
        self.old_events = []
        self.quitting = False
        self.stopped = False
        self.window_title = 'PyBoy'
        self.plugin_manager = PluginManager(self, self.mb, kwargs)
        self.initialized = True

    def tick(self):
        if False:
            print('Hello World!')
        '\n        Progresses the emulator ahead by one frame.\n\n        To run the emulator in real-time, this will need to be called 60 times a second (for example in a while-loop).\n        This function will block for roughly 16,67ms at a time, to not run faster than real-time, unless you specify\n        otherwise with the `PyBoy.set_emulation_speed` method.\n\n        _Open an issue on GitHub if you need finer control, and we will take a look at it._\n        '
        if self.stopped:
            return True
        t_start = time.perf_counter_ns()
        self._handle_events(self.events)
        t_pre = time.perf_counter_ns()
        if not self.paused:
            if self.mb.tick():
                self.plugin_manager.handle_breakpoint()
            else:
                self.frame_count += 1
        t_tick = time.perf_counter_ns()
        self._post_tick()
        t_post = time.perf_counter_ns()
        nsecs = t_pre - t_start
        self.avg_pre = 0.9 * self.avg_pre + 0.1 * nsecs / 1000000000
        nsecs = t_tick - t_pre
        self.avg_tick = 0.9 * self.avg_tick + 0.1 * nsecs / 1000000000
        nsecs = t_post - t_tick
        self.avg_post = 0.9 * self.avg_post + 0.1 * nsecs / 1000000000
        return self.quitting

    def _handle_events(self, events):
        if False:
            for i in range(10):
                print('nop')
        events = self.plugin_manager.handle_events(events)
        for event in events:
            if event == WindowEvent.QUIT:
                self.quitting = True
            elif event == WindowEvent.RELEASE_SPEED_UP:
                self.target_emulationspeed = int(bool(self.target_emulationspeed) ^ True)
                logger.debug('Speed limit: %s' % self.target_emulationspeed)
            elif event == WindowEvent.STATE_SAVE:
                with open(self.gamerom_file + '.state', 'wb') as f:
                    self.mb.save_state(IntIOWrapper(f))
            elif event == WindowEvent.STATE_LOAD:
                state_path = self.gamerom_file + '.state'
                if not os.path.isfile(state_path):
                    logger.error(f'State file not found: {state_path}')
                    continue
                with open(state_path, 'rb') as f:
                    self.mb.load_state(IntIOWrapper(f))
            elif event == WindowEvent.PASS:
                pass
            elif event == WindowEvent.PAUSE_TOGGLE:
                if self.paused:
                    self._unpause()
                else:
                    self._pause()
            elif event == WindowEvent.PAUSE:
                self._pause()
            elif event == WindowEvent.UNPAUSE:
                self._unpause()
            elif event == WindowEvent._INTERNAL_RENDERER_FLUSH:
                self.plugin_manager._post_tick_windows()
            else:
                self.mb.buttonevent(event)

    def _pause(self):
        if False:
            i = 10
            return i + 15
        if self.paused:
            return
        self.paused = True
        self.save_target_emulationspeed = self.target_emulationspeed
        self.target_emulationspeed = 1
        logger.info('Emulation paused!')
        self._update_window_title()

    def _unpause(self):
        if False:
            while True:
                i = 10
        if not self.paused:
            return
        self.paused = False
        self.target_emulationspeed = self.save_target_emulationspeed
        logger.info('Emulation unpaused!')
        self._update_window_title()

    def _post_tick(self):
        if False:
            print('Hello World!')
        if self.frame_count % 60 == 0:
            self._update_window_title()
        self.plugin_manager.post_tick()
        self.plugin_manager.frame_limiter(self.target_emulationspeed)
        self.old_events = self.events
        self.events = []

    def _update_window_title(self):
        if False:
            i = 10
            return i + 15
        avg_emu = self.avg_pre + self.avg_tick + self.avg_post
        self.window_title = f'CPU/frame: {(self.avg_pre + self.avg_tick) / SPF * 100:0.2f}%'
        self.window_title += f" Emulation: x{(round(SPF / avg_emu) if avg_emu > 0 else 'INF')}"
        if self.paused:
            self.window_title += '[PAUSED]'
        self.window_title += self.plugin_manager.window_title()
        self.plugin_manager._set_title()

    def __del__(self):
        if False:
            print('Hello World!')
        self.stop(save=False)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self.stop()

    def stop(self, save=True):
        if False:
            print('Hello World!')
        '\n        Gently stops the emulator and all sub-modules.\n\n        Args:\n            save (bool): Specify whether to save the game upon stopping. It will always be saved in a file next to the\n                provided game-ROM.\n        '
        if self.initialized and (not self.stopped):
            logger.info('###########################')
            logger.info('# Emulator is turning off #')
            logger.info('###########################')
            self.plugin_manager.stop()
            self.mb.stop(save)
            self.stopped = True

    def botsupport_manager(self):
        if False:
            print('Hello World!')
        '\n\n        Returns\n        -------\n        `pyboy.botsupport.BotSupportManager`:\n            The manager, which gives easier access to the emulated game through the classes in `pyboy.botsupport`.\n        '
        return botsupport.BotSupportManager(self, self.mb)

    def openai_gym(self, observation_type='tiles', action_type='press', simultaneous_actions=False, **kwargs):
        if False:
            return 10
        '\n        For Reinforcement learning, it is often easier to use the standard gym environment. This method will provide one.\n        This function requires PyBoy to implement a Game Wrapper for the loaded ROM. You can find the supported games in pyboy.plugins.\n        Additional kwargs are passed to the start_game method of the game_wrapper.\n\n        Args:\n            observation_type (str): Define what the agent will be able to see:\n            * `"raw"`: Gives the raw pixels color\n            * `"tiles"`:  Gives the id of the sprites in 8x8 pixel zones of the game_area defined by the game_wrapper.\n            * `"compressed"`: Gives a more detailled but heavier representation than `"minimal"`.\n            * `"minimal"`: Gives a minimal representation defined by the game_wrapper (recommended).\n\n            action_type (str): Define how the agent will interact with button inputs\n            * `"press"`: The agent will only press inputs for 1 frame an then release it.\n            * `"toggle"`: The agent will toggle inputs, first time it press and second time it release.\n            * `"all"`: The agent have access to all inputs, press and release are separated.\n\n            simultaneous_actions (bool): Allow to inject multiple input at once. This dramatically increases the action_space: \\(n \\rightarrow 2^n\\)\n\n        Returns\n        -------\n        `pyboy.openai_gym.PyBoyGymEnv`:\n            A Gym environment based on the `Pyboy` object.\n        '
        if gym_enabled:
            return PyBoyGymEnv(self, observation_type, action_type, simultaneous_actions, **kwargs)
        else:
            logger.error(f'{__name__}: Missing dependency "gym". ')
            return None

    def game_wrapper(self):
        if False:
            i = 10
            return i + 15
        "\n        Provides an instance of a game-specific wrapper. The game is detected by the cartridge's hard-coded game title\n        (see `pyboy.PyBoy.cartridge_title`).\n\n        If the game isn't supported, None will be returned.\n\n        To get more information, find the wrapper for your game in `pyboy.plugins`.\n\n        Returns\n        -------\n        `pyboy.plugins.base_plugin.PyBoyGameWrapper`:\n            A game-specific wrapper object.\n        "
        return self.plugin_manager.gamewrapper()

    def get_memory_value(self, addr):
        if False:
            return 10
        "\n        Reads a given memory address of the Game Boy's current memory state. This will not directly give you access to\n        all switchable memory banks. Open an issue on GitHub if that is needed, or use `PyBoy.set_memory_value` to send\n        MBC commands to the virtual cartridge.\n\n        Returns\n        -------\n        int:\n            An integer with the value of the memory address\n        "
        return self.mb.getitem(addr)

    def set_memory_value(self, addr, value):
        if False:
            i = 10
            return i + 15
        '\n        Write one byte to a given memory address of the Game Boy\'s current memory state.\n\n        This will not directly give you access to all switchable memory banks.\n\n        __NOTE:__ This function will not let you change ROM addresses (0x0000 to 0x8000). If you write to these\n        addresses, it will send commands to the "Memory Bank Controller" (MBC) of the virtual cartridge. You can read\n        about the MBC at [Pan Docs](http://bgb.bircd.org/pandocs.htm).\n\n        If you need to change ROM values, see `pyboy.PyBoy.override_memory_value`.\n\n        Args:\n            addr (int): Address to write the byte\n            value (int): A byte of data\n        '
        self.mb.setitem(addr, value)

    def override_memory_value(self, rom_bank, addr, value):
        if False:
            print('Hello World!')
        "\n        Override one byte at a given memory address of the Game Boy's ROM.\n\n        This will let you override data in the ROM at any given bank. This is the memory allocated at 0x0000 to 0x8000, where 0x4000 to 0x8000 can be changed from the MBC.\n\n        __NOTE__: Any changes here are not saved or loaded to game states! Use this function with caution and reapply\n        any overrides when reloading the ROM.\n\n        If you need to change a RAM address, see `pyboy.PyBoy.set_memory_value`.\n\n        Args:\n            rom_bank (int): ROM bank to do the overwrite in\n            addr (int): Address to write the byte inside the ROM bank\n            value (int): A byte of data\n        "
        self.mb.cartridge.overrideitem(rom_bank, addr, value)

    def send_input(self, event):
        if False:
            while True:
                i = 10
        '\n        Send a single input to control the emulator. This is both Game Boy buttons and emulator controls.\n\n        See `pyboy.WindowEvent` for which events to send.\n\n        Args:\n            event (pyboy.WindowEvent): The event to send\n        '
        self.events.append(WindowEvent(event))

    def get_input(self, ignore=(WindowEvent.PASS, WindowEvent._INTERNAL_TOGGLE_DEBUG, WindowEvent._INTERNAL_RENDERER_FLUSH, WindowEvent._INTERNAL_MOUSE, WindowEvent._INTERNAL_MARK_TILE)):
        if False:
            print('Hello World!')
        '\n        Get current inputs except the events specified in "ignore" tuple.\n        This is both Game Boy buttons and emulator controls.\n\n        See `pyboy.WindowEvent` for which events to get.\n\n        Args:\n            ignore (tuple): Events this function should ignore\n\n        Returns\n        -------\n        list:\n            List of the `pyboy.utils.WindowEvent`s processed for the last call to `pyboy.PyBoy.tick`\n        '
        return [x for x in self.old_events if x not in ignore]

    def save_state(self, file_like_object):
        if False:
            while True:
                i = 10
        '\n        Saves the complete state of the emulator. It can be called at any time, and enable you to revert any progress in\n        a game.\n\n        You can either save it to a file, or in-memory. The following two examples will provide the file handle in each\n        case. Remember to `seek` the in-memory buffer to the beginning before calling `PyBoy.load_state`:\n\n            # Save to file\n            file_like_object = open("state_file.state", "wb")\n\n            # Save to memory\n            import io\n            file_like_object = io.BytesIO()\n            file_like_object.seek(0)\n\n        Args:\n            file_like_object (io.BufferedIOBase): A file-like object for which to write the emulator state.\n        '
        if isinstance(file_like_object, str):
            raise Exception('String not allowed. Did you specify a filepath instead of a file-like object?')
        self.mb.save_state(IntIOWrapper(file_like_object))

    def load_state(self, file_like_object):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restores the complete state of the emulator. It can be called at any time, and enable you to revert any progress\n        in a game.\n\n        You can either load it from a file, or from memory. See `PyBoy.save_state` for how to save the state, before you\n        can load it here.\n\n        To load a file, remember to load it as bytes:\n\n            # Load file\n            file_like_object = open("state_file.state", "rb")\n\n\n        Args:\n            file_like_object (io.BufferedIOBase): A file-like object for which to read the emulator state.\n        '
        if isinstance(file_like_object, str):
            raise Exception('String not allowed. Did you specify a filepath instead of a file-like object?')
        self.mb.load_state(IntIOWrapper(file_like_object))

    def screen_image(self):
        if False:
            print('Hello World!')
        '\n        Shortcut for `pyboy.botsupport_manager.screen.screen_image`.\n\n        Generates a PIL Image from the screen buffer.\n\n        Convenient for screen captures, but might be a bottleneck, if you use it to train a neural network. In which\n        case, read up on the `pyboy.botsupport` features, [Pan Docs](http://bgb.bircd.org/pandocs.htm) on tiles/sprites,\n        and join our Discord channel for more help.\n\n        Returns\n        -------\n        PIL.Image:\n            RGB image of (160, 144) pixels\n        '
        return self.botsupport_manager().screen().screen_image()

    def _serial(self):
        if False:
            while True:
                i = 10
        '\n        Provides all data that has been sent over the serial port since last call to this function.\n\n        Returns\n        -------\n        str :\n            Buffer data\n        '
        return self.mb.getserial()

    def set_emulation_speed(self, target_speed):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the target emulation speed. It might loose accuracy of keeping the exact speed, when using a high\n        `target_speed`.\n\n        The speed is defined as a multiple of real-time. I.e `target_speed=2` is double speed.\n\n        A `target_speed` of `0` means unlimited. I.e. fastest possible execution.\n\n        Some window types do not implement a frame-limiter, and will always run at full speed.\n\n        Args:\n            target_speed (int): Target emulation speed as multiplier of real-time.\n        '
        if target_speed > 5:
            logger.warning('The emulation speed might not be accurate when speed-target is higher than 5')
        self.target_emulationspeed = target_speed

    def cartridge_title(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the title stored on the currently loaded cartridge ROM. The title is all upper-case ASCII and may\n        have been truncated to 11 characters.\n\n        Returns\n        -------\n        str :\n            Game title\n        '
        return self.mb.cartridge.gamename

    def _rendering(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disable or enable rendering\n        '
        self.mb.lcd.disable_renderer = not value

    def _is_cpu_stuck(self):
        if False:
            for i in range(10):
                print('nop')
        return self.mb.cpu.is_stuck