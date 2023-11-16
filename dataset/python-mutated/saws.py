from __future__ import unicode_literals
from __future__ import print_function
import click
import os
import platform
import subprocess
import traceback
import webbrowser
from prompt_toolkit import AbortAction, Application, CommandLineInterface
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Always, HasFocus, IsDone
from prompt_toolkit.interface import AcceptAction
from prompt_toolkit.layout.processors import HighlightMatchingBracketProcessor, ConditionalProcessor
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.shortcuts import create_default_layout, create_eventloop
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from awscli import completer as awscli_completer
from .completer import AwsCompleter
from .lexer import CommandLexer
from .config import Config
from .style import StyleFactory
from .keys import KeyManager
from .toolbar import Toolbar
from .commands import AwsCommands
from .logger import SawsLogger
from .__init__ import __version__

class Saws(object):
    """Encapsulates the Saws CLI.

    Attributes:
        * aws_cli: An instance of prompt_toolkit's CommandLineInterface.
        * key_manager: An instance of KeyManager.
        * config: An instance of Config.
        * config_obj: An instance of ConfigObj, reads from ~/.sawsrc.
        * theme: A string representing the lexer theme.
        * logger: An instance of SawsLogger.
        * all_commands: A list of all commands, sub_commands, options, etc
            from data/SOURCES.txt.
        * commands: A list of commands from data/SOURCES.txt.
        * sub_commands: A list of sub_commands from data/SOURCES.txt.
        * completer: An instance of AwsCompleter.
    """
    PYGMENTS_CMD = ' | pygmentize -l json'

    def __init__(self, refresh_resources=True):
        if False:
            while True:
                i = 10
        'Inits Saws.\n\n        Args:\n            * refresh_resources: A boolean that determines whether to\n                refresh resources.\n\n        Returns:\n            None.\n        '
        self.aws_cli = None
        self.key_manager = None
        self.config = Config()
        self.config_obj = self.config.read_configuration()
        self.theme = self.config_obj[self.config.MAIN][self.config.THEME]
        self.logger = SawsLogger(__name__, self.config_obj[self.config.MAIN][self.config.LOG_FILE], self.config_obj[self.config.MAIN][self.config.LOG_LEVEL]).logger
        self.all_commands = AwsCommands().all_commands
        self.commands = self.all_commands[AwsCommands.CommandType.COMMANDS.value]
        self.sub_commands = self.all_commands[AwsCommands.CommandType.SUB_COMMANDS.value]
        self.completer = AwsCompleter(awscli_completer, self.all_commands, self.config, self.config_obj, self.log_exception, fuzzy_match=self.get_fuzzy_match(), shortcut_match=self.get_shortcut_match())
        if refresh_resources:
            self.completer.refresh_resources_and_options()
        self._create_cli()

    def log_exception(self, e, traceback, echo=False):
        if False:
            while True:
                i = 10
        'Logs the exception and traceback to the log file ~/.saws.log.\n\n        Args:\n            * e: A Exception that specifies the exception.\n            * traceback: A Traceback that specifies the traceback.\n            * echo: A boolean that specifies whether to echo the exception\n                to the console using click.\n\n        Returns:\n            None.\n        '
        self.logger.debug('exception: %r.', str(e))
        self.logger.error('traceback: %r', traceback.format_exc())
        if echo:
            click.secho(str(e), fg='red')

    def set_color(self, color):
        if False:
            for i in range(10):
                print('nop')
        "Setter for color output mode.\n\n        Used by prompt_toolkit's KeyBindingManager.\n        KeyBindingManager expects this function to be callable so we can't use\n        @property and @attrib.setter.\n\n        Args:\n            * color: A boolean that represents the color flag.\n\n        Returns:\n            None.\n        "
        self.config_obj[self.config.MAIN][self.config.COLOR] = color

    def get_color(self):
        if False:
            for i in range(10):
                print('nop')
        "Getter for color output mode.\n\n        Used by prompt_toolkit's KeyBindingManager.\n        KeyBindingManager expects this function to be callable so we can't use\n        @property and @attrib.setter.\n\n        Args:\n            * None.\n\n        Returns:\n            A boolean that represents the color flag.\n        "
        return self.config_obj[self.config.MAIN].as_bool(self.config.COLOR)

    def set_fuzzy_match(self, fuzzy):
        if False:
            i = 10
            return i + 15
        "Setter for fuzzy matching mode\n\n        Used by prompt_toolkit's KeyBindingManager.\n        KeyBindingManager expects this function to be callable so we can't use\n        @property and @attrib.setter.\n\n        Args:\n            * color: A boolean that represents the fuzzy flag.\n\n        Returns:\n            None.\n        "
        self.config_obj[self.config.MAIN][self.config.FUZZY] = fuzzy
        self.completer.fuzzy_match = fuzzy

    def get_fuzzy_match(self):
        if False:
            i = 10
            return i + 15
        "Getter for fuzzy matching mode\n\n        Used by prompt_toolkit's KeyBindingManager.\n        KeyBindingManager expects this function to be callable so we can't use\n        @property and @attrib.setter.\n\n        Args:\n            * None.\n\n        Returns:\n            A boolean that represents the fuzzy flag.\n        "
        return self.config_obj[self.config.MAIN].as_bool(self.config.FUZZY)

    def set_shortcut_match(self, shortcut):
        if False:
            for i in range(10):
                print('nop')
        "Setter for shortcut matching mode\n\n        Used by prompt_toolkit's KeyBindingManager.\n        KeyBindingManager expects this function to be callable so we can't use\n        @property and @attrib.setter.\n\n        Args:\n            * color: A boolean that represents the shortcut flag.\n\n        Returns:\n            None.\n        "
        self.config_obj[self.config.MAIN][self.config.SHORTCUT] = shortcut
        self.completer.shortcut_match = shortcut

    def get_shortcut_match(self):
        if False:
            i = 10
            return i + 15
        "Getter for shortcut matching mode\n\n        Used by prompt_toolkit's KeyBindingManager.\n        KeyBindingManager expects this function to be callable so we can't use\n        @property and @attrib.setter.\n\n        Args:\n            * None.\n\n        Returns:\n            A boolean that represents the shortcut flag.\n        "
        return self.config_obj[self.config.MAIN].as_bool(self.config.SHORTCUT)

    def refresh_resources_and_options(self):
        if False:
            while True:
                i = 10
        "Convenience function to refresh resources and options for completion.\n\n        Used by prompt_toolkit's KeyBindingManager.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        "
        self.completer.refresh_resources_and_options(force_refresh=True)

    def handle_docs(self, text=None, from_fkey=False):
        if False:
            i = 10
            return i + 15
        'Displays contextual web docs for `F9` or the `docs` command.\n\n        Displays the web docs specific to the currently entered:\n\n        * (optional) command\n        * (optional) subcommand\n\n        If no command or subcommand is present, the docs index page is shown.\n\n        Docs are only displayed if:\n\n        * from_fkey is True\n        * from_fkey is False and `docs` is found in text\n\n        Args:\n            * text: A string representing the input command text.\n            * from_fkey: A boolean representing whether this function is\n                being executed from an `F9` key press.\n\n        Returns:\n            A boolean representing whether the web docs were shown.\n        '
        base_url = 'http://docs.aws.amazon.com/cli/latest/reference/'
        index_html = 'index.html'
        if text is None:
            text = self.aws_cli.current_buffer.document.text
        if from_fkey:
            text = text.strip() + ' ' + AwsCommands.AWS_DOCS
        tokens = text.split()
        if len(tokens) > 2 and tokens[-1] == AwsCommands.AWS_DOCS:
            prev_word = tokens[-2]
            if prev_word in self.commands:
                prev_word = prev_word + '/'
                url = base_url + prev_word + index_html
                webbrowser.open(url)
                return True
            elif prev_word in self.sub_commands:
                command_url = tokens[-3] + '/'
                sub_command_url = tokens[-2] + '.html'
                url = base_url + command_url + sub_command_url
                webbrowser.open(url)
                return True
            webbrowser.open(base_url + index_html)
        if from_fkey or AwsCommands.AWS_DOCS in tokens:
            webbrowser.open(base_url + index_html)
            return True
        return False

    def _handle_cd(self, text):
        if False:
            while True:
                i = 10
        "Handles a `cd` shell command by calling python's os.chdir.\n\n        Simply passing in the `cd` command to subprocess.call doesn't work.\n        Note: Changing the directory within Saws will only be in effect while\n        running Saws.  Exiting the program will return you to the directory\n        you were in prior to running Saws.\n\n        Attributes:\n            * text: A string representing the input command text.\n\n        Returns:\n            A boolean representing a `cd` command was found and handled.\n        "
        CD_CMD = 'cd'
        stripped_text = text.strip()
        if stripped_text.startswith(CD_CMD):
            directory = ''
            if stripped_text == CD_CMD:
                directory = os.path.expanduser('~')
            else:
                tokens = text.split(CD_CMD + ' ')
                directory = tokens[-1]
            try:
                os.chdir(directory)
            except OSError as e:
                self.log_exception(e, traceback, echo=True)
            return True
        return False

    def _colorize_output(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Highlights output with pygments.\n\n        Only highlights the output if all of the following conditions are True:\n\n        * The color option is enabled\n        * The command will be handled by the `aws-cli`\n        * The text does not contain the `configure` command\n        * The text does not contain the `help` command, which already does\n            output highlighting\n\n        Args:\n            * text: A string that represents the input command text.\n\n        Returns:\n            A string that represents:\n                * The original command text if no highlighting was performed.\n                * The pygments highlighted command text otherwise.\n        '
        stripped_text = text.strip()
        if not self.get_color() or stripped_text == '':
            return text
        if AwsCommands.AWS_COMMAND not in stripped_text.split():
            return text
        excludes = [AwsCommands.AWS_CONFIGURE, AwsCommands.AWS_HELP, '|']
        if not any((substring in stripped_text for substring in excludes)):
            return text.strip() + self.PYGMENTS_CMD
        else:
            return text

    def _handle_keyboard_interrupt(self, e, platform):
        if False:
            while True:
                i = 10
        'Handles keyboard interrupts more gracefully on Mac/Unix/Linux.\n\n        Allows Mac/Unix/Linux to continue running on keyboard interrupt,\n        as the user might interrupt a long-running AWS command with Control-C\n        while continuing to work with Saws.\n\n        On Windows, the "Terminate batch job (Y/N)" confirmation makes it\n        tricky to handle this gracefully.  Thus, we re-raise KeyboardInterrupt.\n\n        Args:\n            * e: A KeyboardInterrupt.\n            * platform: A string that denotes platform such as\n                \'Windows\', \'Darwin\', etc.\n\n        Returns:\n            None\n\n        Raises:\n            Exception: A KeyboardInterrupt if running on Windows.\n        '
        if platform == 'Windows':
            raise e
        else:
            self.aws_cli.renderer.clear()
            self.aws_cli.input_processor.feed(KeyPress(Keys.ControlM, u''))
            self.aws_cli.input_processor.process_keys()

    def _process_command(self, text):
        if False:
            return 10
        'Processes the input command, called by the cli event loop\n\n        Args:\n            * text: A string that represents the input command text.\n\n        Returns:\n            None.\n        '
        if AwsCommands.AWS_COMMAND in text:
            text = self.completer.replace_shortcut(text)
            if self.handle_docs(text):
                return
        try:
            if not self._handle_cd(text):
                text = self._colorize_output(text)
                subprocess.call(text, shell=True)
            print('')
        except KeyboardInterrupt as e:
            self._handle_keyboard_interrupt(e, platform.system())
        except Exception as e:
            self.log_exception(e, traceback, echo=True)

    def _create_cli(self):
        if False:
            for i in range(10):
                print('nop')
        "Creates the prompt_toolkit's CommandLineInterface.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        "
        history = FileHistory(os.path.expanduser('~/.saws-history'))
        toolbar = Toolbar(self.get_color, self.get_fuzzy_match, self.get_shortcut_match)
        layout = create_default_layout(message='saws> ', reserve_space_for_menu=8, lexer=CommandLexer, get_bottom_toolbar_tokens=toolbar.handler, extra_input_processors=[ConditionalProcessor(processor=HighlightMatchingBracketProcessor(chars='[](){}'), filter=HasFocus(DEFAULT_BUFFER) & ~IsDone())])
        cli_buffer = Buffer(history=history, auto_suggest=AutoSuggestFromHistory(), enable_history_search=True, completer=self.completer, complete_while_typing=Always(), accept_action=AcceptAction.RETURN_DOCUMENT)
        self.key_manager = KeyManager(self.set_color, self.get_color, self.set_fuzzy_match, self.get_fuzzy_match, self.set_shortcut_match, self.get_shortcut_match, self.refresh_resources_and_options, self.handle_docs)
        style_factory = StyleFactory(self.theme)
        application = Application(mouse_support=False, style=style_factory.style, layout=layout, buffer=cli_buffer, key_bindings_registry=self.key_manager.manager.registry, on_exit=AbortAction.RAISE_EXCEPTION, on_abort=AbortAction.RETRY, ignore_case=True)
        eventloop = create_eventloop()
        self.aws_cli = CommandLineInterface(application=application, eventloop=eventloop)

    def run_cli(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs the main loop.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        '
        print('Version:', __version__)
        print('Theme:', self.theme)
        while True:
            document = self.aws_cli.run(reset_current_buffer=True)
            self._process_command(document.text)