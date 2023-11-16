from typing import TYPE_CHECKING, Optional
from .base import MATCH_WINDOW_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import RemoveMarkerRCOptions as CLIOptions

class RemoveMarker(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which window to remove the marker from\n    self/bool: Boolean indicating whether to detach the window the command is run in\n    '
    short_desc = 'Remove the currently set marker, if any.'
    options_spec = MATCH_WINDOW_OPTION + '\n\n--self\ntype=bool-set\nApply marker to the window this command is run in, rather than the active window.\n'

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        return {'match': opts.match, 'self': opts.self}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            while True:
                i = 10
        for window in self.windows_for_match_payload(boss, window, payload_get):
            if window:
                window.remove_marker()
        return None
remove_marker = RemoveMarker()