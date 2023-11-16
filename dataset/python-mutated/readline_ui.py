"""Readline-Based Command-Line Interface of TensorFlow Debugger (tfdbg)."""
import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common

class ReadlineUI(base_ui.BaseUI):
    """Readline-based Command-line UI."""

    def __init__(self, on_ui_exit=None, config=None):
        if False:
            for i in range(10):
                print('nop')
        base_ui.BaseUI.__init__(self, on_ui_exit=on_ui_exit, config=config)
        self._init_input()

    def _init_input(self):
        if False:
            print('Hello World!')
        readline.parse_and_bind('set editing-mode emacs')
        readline.set_completer_delims('\n')
        readline.set_completer(self._readline_complete)
        readline.parse_and_bind('tab: complete')
        self._input = input

    def _readline_complete(self, text, state):
        if False:
            for i in range(10):
                print('nop')
        (context, prefix, except_last_word) = self._analyze_tab_complete_input(text)
        (candidates, _) = self._tab_completion_registry.get_completions(context, prefix)
        candidates = [except_last_word + candidate for candidate in candidates]
        return candidates[state]

    def run_ui(self, init_command=None, title=None, title_color=None, enable_mouse_on_start=True):
        if False:
            while True:
                i = 10
        'Run the CLI: See the doc of base_ui.BaseUI.run_ui for more details.'
        print(title)
        if init_command is not None:
            self._dispatch_command(init_command)
        exit_token = self._ui_loop()
        if self._on_ui_exit:
            self._on_ui_exit()
        return exit_token

    def _ui_loop(self):
        if False:
            while True:
                i = 10
        while True:
            command = self._get_user_command()
            exit_token = self._dispatch_command(command)
            if exit_token is not None:
                return exit_token

    def _get_user_command(self):
        if False:
            print('Hello World!')
        print('')
        return self._input(self.CLI_PROMPT).strip()

    def _dispatch_command(self, command):
        if False:
            while True:
                i = 10
        'Dispatch user command.\n\n    Args:\n      command: (str) Command to dispatch.\n\n    Returns:\n      An exit token object. None value means that the UI loop should not exit.\n      A non-None value means the UI loop should exit.\n    '
        if command in self.CLI_EXIT_COMMANDS:
            return debugger_cli_common.EXPLICIT_USER_EXIT
        try:
            (prefix, args, output_file_path) = self._parse_command(command)
        except SyntaxError as e:
            print(str(e))
            return
        if self._command_handler_registry.is_registered(prefix):
            try:
                screen_output = self._command_handler_registry.dispatch_command(prefix, args, screen_info=None)
            except debugger_cli_common.CommandLineExit as e:
                return e.exit_token
        else:
            screen_output = debugger_cli_common.RichTextLines([self.ERROR_MESSAGE_PREFIX + 'Invalid command prefix "%s"' % prefix])
        self._display_output(screen_output)
        if output_file_path:
            try:
                screen_output.write_to_file(output_file_path)
                print('Wrote output to %s' % output_file_path)
            except Exception:
                print('Failed to write output to %s' % output_file_path)

    def _display_output(self, screen_output):
        if False:
            for i in range(10):
                print('nop')
        for line in screen_output.lines:
            print(line)