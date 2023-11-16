from typing import TYPE_CHECKING, Optional, Union
from .base import MATCH_TAB_OPTION, MATCH_WINDOW_OPTION, ArgsType, Boss, MatchError, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import DetachWindowRCOptions as CLIOptions

class DetachWindow(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which window to detach\n    target_tab/str: Which tab to move the detached window to\n    self/bool: Boolean indicating whether to detach the window the command is run in\n    '
    short_desc = 'Detach the specified windows and place them in a different/new tab'
    desc = 'Detach the specified windows and either move them into a new tab, a new OS window or add them to the specified tab. Use the special value :code:`new` for :option:`kitten @ detach-window --target-tab` to move to a new tab. If no target tab is specified the windows are moved to a new OS window.'
    options_spec = MATCH_WINDOW_OPTION + '\n\n' + MATCH_TAB_OPTION.replace('--match -m', '--target-tab -t') + 'Use the special value :code:`new` to move to a new tab.\n\n\n--self\ntype=bool-set\nDetach the window this command is run in, rather than the active window.\n'

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            return 10
        return {'match': opts.match, 'target_tab': opts.target_tab, 'self': opts.self}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            i = 10
            return i + 15
        windows = self.windows_for_match_payload(boss, window, payload_get)
        match = payload_get('target_tab')
        target_tab_id: Optional[Union[str, int]] = None
        newval: Union[str, int] = 'new'
        if match:
            if match == 'new':
                target_tab_id = newval
            else:
                tabs = tuple(boss.match_tabs(match))
                if not tabs:
                    raise MatchError(match, 'tabs')
                target_tab_id = tabs[0].id
        kwargs = {'target_os_window_id': newval} if target_tab_id is None else {'target_tab_id': target_tab_id}
        for window in windows:
            if window:
                boss._move_window_to(window=window, **kwargs)
        return None
detach_window = DetachWindow()