"""Base Class of TensorFlow Debugger (tfdbg) Command-Line Interface."""
import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common

class BaseUI(object):
    """Base class of tfdbg user interface."""
    CLI_PROMPT = 'tfdbg> '
    CLI_EXIT_COMMANDS = ['exit', 'quit']
    ERROR_MESSAGE_PREFIX = 'ERROR: '
    INFO_MESSAGE_PREFIX = 'INFO: '

    def __init__(self, on_ui_exit=None, config=None):
        if False:
            return 10
        'Constructor of the base class.\n\n    Args:\n      on_ui_exit: (`Callable`) the callback to be called when the UI exits.\n      config: An instance of `cli_config.CLIConfig()` carrying user-facing\n        configurations.\n    '
        self._on_ui_exit = on_ui_exit
        self._command_handler_registry = debugger_cli_common.CommandHandlerRegistry()
        self._tab_completion_registry = debugger_cli_common.TabCompletionRegistry()
        self._tab_completion_registry.register_tab_comp_context([''], self.CLI_EXIT_COMMANDS + [debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND] + debugger_cli_common.CommandHandlerRegistry.HELP_COMMAND_ALIASES)
        self._config = config or cli_config.CLIConfig()
        self._config_argparser = argparse.ArgumentParser(description='config command', usage=argparse.SUPPRESS)
        subparsers = self._config_argparser.add_subparsers()
        set_parser = subparsers.add_parser('set')
        set_parser.add_argument('property_name', type=str)
        set_parser.add_argument('property_value', type=str)
        set_parser = subparsers.add_parser('show')
        self.register_command_handler('config', self._config_command_handler, self._config_argparser.format_help(), prefix_aliases=['cfg'])

    def set_help_intro(self, help_intro):
        if False:
            for i in range(10):
                print('nop')
        'Set an introductory message to the help output of the command registry.\n\n    Args:\n      help_intro: (RichTextLines) Rich text lines appended to the beginning of\n        the output of the command "help", as introductory information.\n    '
        self._command_handler_registry.set_help_intro(help_intro=help_intro)

    def register_command_handler(self, prefix, handler, help_info, prefix_aliases=None):
        if False:
            while True:
                i = 10
        'A wrapper around CommandHandlerRegistry.register_command_handler().\n\n    In addition to calling the wrapped register_command_handler() method, this\n    method also registers the top-level tab-completion context based on the\n    command prefixes and their aliases.\n\n    See the doc string of the wrapped method for more details on the args.\n\n    Args:\n      prefix: (str) command prefix.\n      handler: (callable) command handler.\n      help_info: (str) help information.\n      prefix_aliases: (list of str) aliases of the command prefix.\n    '
        self._command_handler_registry.register_command_handler(prefix, handler, help_info, prefix_aliases=prefix_aliases)
        self._tab_completion_registry.extend_comp_items('', [prefix])
        if prefix_aliases:
            self._tab_completion_registry.extend_comp_items('', prefix_aliases)

    def register_tab_comp_context(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Wrapper around TabCompletionRegistry.register_tab_comp_context().'
        self._tab_completion_registry.register_tab_comp_context(*args, **kwargs)

    def run_ui(self, init_command=None, title=None, title_color=None, enable_mouse_on_start=True):
        if False:
            print('Hello World!')
        'Run the UI until user- or command- triggered exit.\n\n    Args:\n      init_command: (str) Optional command to run on CLI start up.\n      title: (str) Optional title to display in the CLI.\n      title_color: (str) Optional color of the title, e.g., "yellow".\n      enable_mouse_on_start: (bool) Whether the mouse mode is to be enabled on\n        start-up.\n\n    Returns:\n      An exit token of arbitrary type. Can be None.\n    '
        raise NotImplementedError('run_ui() is not implemented in BaseUI')

    def _parse_command(self, command):
        if False:
            i = 10
            return i + 15
        'Parse a command string into prefix and arguments.\n\n    Args:\n      command: (str) Command string to be parsed.\n\n    Returns:\n      prefix: (str) The command prefix.\n      args: (list of str) The command arguments (i.e., not including the\n        prefix).\n      output_file_path: (str or None) The path to save the screen output\n        to (if any).\n    '
        command = command.strip()
        if not command:
            return ('', [], None)
        command_items = command_parser.parse_command(command)
        (command_items, output_file_path) = command_parser.extract_output_file_path(command_items)
        return (command_items[0], command_items[1:], output_file_path)

    def _analyze_tab_complete_input(self, text):
        if False:
            while True:
                i = 10
        'Analyze raw input to tab-completer.\n\n    Args:\n      text: (str) the full, raw input text to be tab-completed.\n\n    Returns:\n      context: (str) the context str. For example,\n        If text == "print_tensor softmax", returns "print_tensor".\n        If text == "print", returns "".\n        If text == "", returns "".\n      prefix: (str) the prefix to be tab-completed, from the last word.\n        For example, if text == "print_tensor softmax", returns "softmax".\n        If text == "print", returns "print".\n        If text == "", returns "".\n      except_last_word: (str) the input text, except the last word.\n        For example, if text == "print_tensor softmax", returns "print_tensor".\n        If text == "print_tensor -a softmax", returns "print_tensor -a".\n        If text == "print", returns "".\n        If text == "", returns "".\n    '
        text = text.lstrip()
        if not text:
            context = ''
            prefix = ''
            except_last_word = ''
        else:
            items = text.split(' ')
            if len(items) == 1:
                context = ''
                prefix = items[0]
                except_last_word = ''
            else:
                context = items[0]
                prefix = items[-1]
                except_last_word = ' '.join(items[:-1]) + ' '
        return (context, prefix, except_last_word)

    @property
    def config(self):
        if False:
            return 10
        'Obtain the CLIConfig of this `BaseUI` instance.'
        return self._config

    def _config_command_handler(self, args, screen_info=None):
        if False:
            for i in range(10):
                print('nop')
        'Command handler for the "config" command.'
        del screen_info
        parsed = self._config_argparser.parse_args(args)
        if hasattr(parsed, 'property_name') and hasattr(parsed, 'property_value'):
            self._config.set(parsed.property_name, parsed.property_value)
            return self._config.summarize(highlight=parsed.property_name)
        else:
            return self._config.summarize()