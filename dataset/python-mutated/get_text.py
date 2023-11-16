from typing import TYPE_CHECKING, Optional
from .base import MATCH_WINDOW_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import GetTextRCOptions as CLIOptions

class GetText(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: The window to get text from\n    extent/choices.screen.first_cmd_output_on_screen.last_cmd_output.last_visited_cmd_output.all.selection:         One of :code:`screen`, :code:`first_cmd_output_on_screen`, :code:`last_cmd_output`,         :code:`last_visited_cmd_output`, :code:`all`, or :code:`selection`\n    ansi/bool: Boolean, if True send ANSI formatting codes\n    cursor/bool: Boolean, if True send cursor position/style as ANSI codes\n    wrap_markers/bool: Boolean, if True add wrap markers to output\n    clear_selection/bool: Boolean, if True clear the selection in the matched window\n    self/bool: Boolean, if True use window the command was run in\n    '
    short_desc = 'Get text from the specified window'
    options_spec = MATCH_WINDOW_OPTION + '\n\n--extent\ndefault=screen\nchoices=screen, all, selection, first_cmd_output_on_screen, last_cmd_output, last_visited_cmd_output, last_non_empty_output\nWhat text to get. The default of :code:`screen` means all text currently on the screen.\n:code:`all` means all the screen+scrollback and :code:`selection` means the\ncurrently selected text. :code:`first_cmd_output_on_screen` means the output of the first\ncommand that was run in the window on screen. :code:`last_cmd_output` means\nthe output of the last command that was run in the window. :code:`last_visited_cmd_output` means\nthe first command output below the last scrolled position via scroll_to_prompt.\n:code:`last_non_empty_output` is the output from the last command run in the window that had\nsome non empty output. The last four require :ref:`shell_integration` to be enabled.\n\n\n--ansi\ntype=bool-set\nBy default, only plain text is returned. With this flag, the text will\ninclude the ANSI formatting escape codes for colors, bold, italic, etc.\n\n\n--add-cursor\ntype=bool-set\nAdd ANSI escape codes specifying the cursor position and style to the end of the text.\n\n\n--add-wrap-markers\ntype=bool-set\nAdd carriage returns at every line wrap location (where long lines are wrapped at\nscreen edges).\n\n\n--clear-selection\ntype=bool-set\nClear the selection in the matched window, if any.\n\n\n--self\ntype=bool-set\nGet text from the window this command is run in, rather than the active window.\n'
    field_to_option_map = {'wrap_markers': 'add_wrap_markers', 'cursor': 'add_cursor'}

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        return {'match': opts.match, 'extent': opts.extent, 'ansi': opts.ansi, 'cursor': opts.add_cursor, 'wrap_markers': opts.add_wrap_markers, 'clear_selection': opts.clear_selection, 'self': opts.self}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            i = 10
            return i + 15
        from kitty.window import CommandOutput
        windows = self.windows_for_match_payload(boss, window, payload_get)
        if windows and windows[0]:
            window = windows[0]
        else:
            return None
        if payload_get('extent') == 'selection':
            ans = window.text_for_selection(as_ansi=payload_get('ansi'))
        elif payload_get('extent') == 'first_cmd_output_on_screen':
            ans = window.cmd_output(CommandOutput.first_on_screen, as_ansi=bool(payload_get('ansi')), add_wrap_markers=bool(payload_get('wrap_markers')))
        elif payload_get('extent') == 'last_cmd_output':
            ans = window.cmd_output(CommandOutput.last_run, as_ansi=bool(payload_get('ansi')), add_wrap_markers=bool(payload_get('wrap_markers')))
        elif payload_get('extent') == 'last_non_empty_output':
            ans = window.cmd_output(CommandOutput.last_non_empty, as_ansi=bool(payload_get('ansi')), add_wrap_markers=bool(payload_get('wrap_markers')))
        elif payload_get('extent') == 'last_visited_cmd_output':
            ans = window.cmd_output(CommandOutput.last_visited, as_ansi=bool(payload_get('ansi')), add_wrap_markers=bool(payload_get('wrap_markers')))
        else:
            ans = window.as_text(as_ansi=bool(payload_get('ansi')), add_history=payload_get('extent') == 'all', add_cursor=bool(payload_get('cursor')), add_wrap_markers=bool(payload_get('wrap_markers')))
        if payload_get('clear_selection'):
            window.clear_selection()
        return ans
get_text = GetText()