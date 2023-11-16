"""doitlive IPython support."""
from warnings import warn
from click import Abort
from IPython.terminal.interactiveshell import DISPLAY_BANNER_DEPRECATED, TerminalInteractiveShell
from IPython.terminal.ipapp import TerminalIPythonApp
from prompt_toolkit.interface import CommandLineInterface, _InterfaceEventLoopCallbacks
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import create_output
from doitlive import RETURNS, wait_for, echo

class _PlayerInterfaceEventLoopCallbacks(_InterfaceEventLoopCallbacks):

    def __init__(self, cli, on_feed_key):
        if False:
            return 10
        super().__init__(cli)
        self.on_feed_key = on_feed_key

    def feed_key(self, key_press, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        key_press = self.on_feed_key(key_press)
        if key_press is not None:
            return super().feed_key(key_press, *args, **kwargs)

class _PlayerCommandLineInterface(CommandLineInterface):

    def __init__(self, application, eventloop=None, input=None, output=None, on_feed_key=None):
        if False:
            while True:
                i = 10
        super().__init__(application, eventloop, input, output)
        self.on_feed_key = on_feed_key

    def create_eventloop_callbacks(self):
        if False:
            return 10
        return _PlayerInterfaceEventLoopCallbacks(self, on_feed_key=self.on_feed_key)

class PlayerTerminalInteractiveShell(TerminalInteractiveShell):
    """A magic IPython terminal shell."""

    def __init__(self, commands, speed=1, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.commands = commands or []
        self.speed = speed
        self.current_command_index = 0
        self.current_command_pos = 0
        super().__init__(*args, **kwargs)

    def on_feed_key(self, key_press):
        if False:
            for i in range(10):
                print('nop')
        'Handles the magictyping when a key is pressed'
        if key_press.key in {Keys.Escape, Keys.ControlC}:
            echo(carriage_return=True)
            raise Abort()
        if key_press.key == Keys.Backspace:
            if self.current_command_pos > 0:
                self.current_command_pos -= 1
            return key_press
        ret = None
        if key_press.key != Keys.CPRResponse:
            if self.current_command_pos < len(self.current_command):
                current_key = self.current_command_key
                ret = KeyPress(current_key)
                increment = min([self.speed, len(self.current_command) - self.current_command_pos])
                self.current_command_pos += increment
            else:
                if key_press.key != Keys.Enter:
                    return None
                self.current_command_index += 1
                self.current_command_pos = 0
                ret = key_press
        return ret

    @property
    def current_command(self):
        if False:
            i = 10
            return i + 15
        return self.commands[self.current_command_index]

    @property
    def current_command_key(self):
        if False:
            return 10
        pos = self.current_command_pos
        end = min(pos + self.speed, len(self.current_command))
        return self.current_command[pos:end]

    def interact(self, display_banner=DISPLAY_BANNER_DEPRECATED):
        if False:
            i = 10
            return i + 15
        if display_banner is not DISPLAY_BANNER_DEPRECATED:
            warn('interact `display_banner` argument is deprecated since IPython 5.0. Call `show_banner()` if needed.', DeprecationWarning, stacklevel=2)
        self.keep_running = True
        while self.keep_running:
            print(self.separate_in, end='')
            if self.current_command_index > len(self.commands) - 1:
                echo('Do you really want to exit ([y]/n)? ', nl=False)
                wait_for(RETURNS)
                self.ask_exit()
                return None
            try:
                code = self.prompt_for_code()
            except EOFError:
                if not self.confirm_exit or self.ask_yes_no('Do you really want to exit ([y]/n)?', 'y', 'n'):
                    self.ask_exit()
            else:
                if code:
                    self.run_cell(code, store_history=True)

    def init_prompt_toolkit_cli(self):
        if False:
            while True:
                i = 10
        super().init_prompt_toolkit_cli()
        self.pt_cli = _PlayerCommandLineInterface(self._pt_app, eventloop=self._eventloop, output=create_output(true_color=self.true_color), on_feed_key=self.on_feed_key)

class PlayerTerminalIPythonApp(TerminalIPythonApp):
    """IPython app that runs the PlayerTerminalInteractiveShell."""
    commands = tuple()
    speed = 1

    def parse_command_line(self, argv=None):
        if False:
            i = 10
            return i + 15
        return None

    def init_shell(self):
        if False:
            while True:
                i = 10
        'initialize the InteractiveShell instance'
        self.shell = PlayerTerminalInteractiveShell.instance(commands=self.commands, speed=self.speed, parent=self, display_banner=False, profile_dir=self.profile_dir, ipython_dir=self.ipython_dir, user_ns=self.user_ns)
        self.shell.configurables.append(self)