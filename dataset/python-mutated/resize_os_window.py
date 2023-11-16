from typing import TYPE_CHECKING, Optional
from .base import MATCH_WINDOW_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import ResizeOSWindowRCOptions as CLIOptions

class ResizeOSWindow(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which window to resize\n    self/bool: Boolean indicating whether to close the window the command is run in\n    incremental/bool: Boolean indicating whether to adjust the size incrementally\n    action/choices.resize.toggle-fullscreen.toggle-maximized: One of :code:`resize, toggle-fullscreen` or :code:`toggle-maximized`\n    unit/choices.cells.pixels: One of :code:`cells` or :code:`pixels`\n    width/int: Integer indicating desired window width\n    height/int: Integer indicating desired window height\n    '
    short_desc = 'Resize the specified OS Windows'
    desc = 'Resize the specified OS Windows. Note that some window managers/environments do not allow applications to resize their windows, for example, tiling window managers.'
    options_spec = MATCH_WINDOW_OPTION + "\n\n--action\ndefault=resize\nchoices=resize,toggle-fullscreen,toggle-maximized\nThe action to perform.\n\n\n--unit\ndefault=cells\nchoices=cells,pixels\nThe unit in which to interpret specified sizes.\n\n\n--width\ndefault=0\ntype=int\nChange the width of the window. Zero leaves the width unchanged.\n\n\n--height\ndefault=0\ntype=int\nChange the height of the window. Zero leaves the height unchanged.\n\n\n--incremental\ntype=bool-set\nTreat the specified sizes as increments on the existing window size\ninstead of absolute sizes.\n\n\n--self\ntype=bool-set\nResize the window this command is run in, rather than the active window.\n\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response indicating the success of the action. Note that\nusing this option means that you will not be notified of failures.\n"

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        return {'match': opts.match, 'action': opts.action, 'unit': opts.unit, 'width': opts.width, 'height': opts.height, 'self': opts.self, 'incremental': opts.incremental}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            print('Hello World!')
        windows = self.windows_for_match_payload(boss, window, payload_get)
        if windows:
            ac = payload_get('action')
            for os_window_id in {w.os_window_id for w in windows if w}:
                if ac == 'resize':
                    boss.resize_os_window(os_window_id, width=payload_get('width'), height=payload_get('height'), unit=payload_get('unit'), incremental=payload_get('incremental'))
                elif ac == 'toggle-fullscreen':
                    boss.toggle_fullscreen(os_window_id)
                elif ac == 'toggle-maximized':
                    boss.toggle_maximized(os_window_id)
        return None
resize_os_window = ResizeOSWindow()