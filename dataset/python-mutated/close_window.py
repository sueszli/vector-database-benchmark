from typing import TYPE_CHECKING, Optional
from .base import MATCH_WINDOW_OPTION, ArgsType, Boss, MatchError, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import CloseWindowRCOptions as CLIOptions

class CloseWindow(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which window to close\n    self/bool: Boolean indicating whether to close the window the command is run in\n    ignore_no_match/bool: Boolean indicating whether no matches should be ignored or return an error\n    '
    short_desc = 'Close the specified windows'
    options_spec = MATCH_WINDOW_OPTION + "\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response indicating the success of the action. Note that\nusing this option means that you will not be notified of failures.\n\n\n--self\ntype=bool-set\nClose the window this command is run in, rather than the active window.\n\n\n--ignore-no-match\ntype=bool-set\nDo not return an error if no windows are matched to be closed.\n"

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            return 10
        return {'match': opts.match, 'self': opts.self, 'ignore_no_match': opts.ignore_no_match}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            i = 10
            return i + 15
        try:
            windows = self.windows_for_match_payload(boss, window, payload_get)
        except MatchError:
            if payload_get('ignore_no_match'):
                return None
            raise
        for window in tuple(windows):
            if window:
                boss.mark_window_for_close(window)
        return None
close_window = CloseWindow()