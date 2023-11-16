from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, MATCH_WINDOW_OPTION, ArgsType, Boss, OpacityError, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import SetBackgroundOpacityRCOptions as CLIOptions

class SetBackgroundOpacity(RemoteCommand):
    protocol_spec = __doc__ = '\n    opacity+/float: A number between 0.1 and 1\n    match_window/str: Window to change opacity in\n    match_tab/str: Tab to change opacity in\n    all/bool: Boolean indicating operate on all windows\n    toggle/bool: Boolean indicating if opacity should be toggled between the default and the specified value\n    '
    short_desc = 'Set the background opacity'
    desc = 'Set the background opacity for the specified windows. This will only work if you have turned on :opt:`dynamic_background_opacity` in :file:`kitty.conf`. The background opacity affects all kitty windows in a single OS window. For example::\n\n    kitten @ set-background-opacity 0.5'
    options_spec = '--all -a\ntype=bool-set\nBy default, background opacity are only changed for the currently active OS window. This option will\ncause background opacity to be changed in all windows.\n\n\n--toggle\ntype=bool-set\nWhen specified, the background opacity for the matching OS windows will be reset to default if it is currently\nequal to the specified value, otherwise it will be set to the specified value.\n' + '\n\n' + MATCH_WINDOW_OPTION + '\n\n' + MATCH_TAB_OPTION.replace('--match -m', '--match-tab -t')
    args = RemoteCommand.Args(spec='OPACITY', count=1, json_field='opacity', special_parse='parse_opacity(args[0])')

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            return 10
        opacity = max(0.1, min(float(args[0]), 1.0))
        return {'opacity': opacity, 'match_window': opts.match, 'all': opts.all, 'match_tab': opts.match_tab, 'toggle': opts.toggle}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            i = 10
            return i + 15
        from kitty.fast_data_types import background_opacity_of, get_options
        opts = get_options()
        if not opts.dynamic_background_opacity:
            raise OpacityError('You must turn on the dynamic_background_opacity option in kitty.conf to be able to set background opacity')
        windows = self.windows_for_payload(boss, window, payload_get)
        for os_window_id in {w.os_window_id for w in windows if w}:
            val: float = payload_get('opacity')
            if payload_get('toggle'):
                current = background_opacity_of(os_window_id)
                if current == val:
                    val = opts.background_opacity
            boss._set_os_window_background_opacity(os_window_id, val)
        return None
set_background_opacity = SetBackgroundOpacity()