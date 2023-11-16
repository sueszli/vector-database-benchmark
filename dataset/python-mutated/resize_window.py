from typing import TYPE_CHECKING, Optional, Union
from .base import MATCH_WINDOW_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import ResizeWindowRCOptions as CLIOptions

class ResizeWindow(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which window to resize\n    self/bool: Boolean indicating whether to resize the window the command is run in\n    increment/int: Integer specifying the resize increment\n    axis/choices.horizontal.vertical.reset: One of :code:`horizontal, vertical` or :code:`reset`\n    '
    short_desc = 'Resize the specified windows'
    desc = 'Resize the specified windows in the current layout. Note that not all layouts can resize all windows in all directions.'
    options_spec = MATCH_WINDOW_OPTION + '\n\n--increment -i\ntype=int\ndefault=2\nThe number of cells to change the size by, can be negative to decrease the size.\n\n\n--axis -a\ntype=choices\nchoices=horizontal,vertical,reset\ndefault=horizontal\nThe axis along which to resize. If :code:`horizontal`,\nit will make the window wider or narrower by the specified increment.\nIf :code:`vertical`, it will make the window taller or shorter by the specified increment.\nThe special value :code:`reset` will reset the layout to its default configuration.\n\n\n--self\ntype=bool-set\nResize the window this command is run in, rather than the active window.\n'
    string_return_is_error = True

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        return {'match': opts.match, 'increment': opts.increment, 'axis': opts.axis, 'self': opts.self}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            print('Hello World!')
        windows = self.windows_for_match_payload(boss, window, payload_get)
        resized: Union[bool, None, str] = False
        if windows and windows[0]:
            resized = boss.resize_layout_window(windows[0], increment=payload_get('increment'), is_horizontal=payload_get('axis') == 'horizontal', reset=payload_get('axis') == 'reset')
        return resized
resize_window = ResizeWindow()