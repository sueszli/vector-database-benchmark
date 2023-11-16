from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, ArgsType, Boss, MatchError, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import CloseTabRCOptions as CLIOptions

class CloseTab(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which tab to close\n    self/bool: Boolean indicating whether to close the tab of the window the command is run in\n    ignore_no_match/bool: Boolean indicating whether no matches should be ignored or return an error\n    '
    short_desc = 'Close the specified tabs'
    desc = 'Close an arbitrary set of tabs. The :code:`--match` option can be used to\nspecify complex sets of tabs to close. For example, to close all non-focused\ntabs in the currently focused OS window, use::\n\n    kitten @ close-tab --match "not state:focused and state:parent_focused"\n'
    options_spec = MATCH_TAB_OPTION + "\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response indicating the success of the action. Note that\nusing this option means that you will not be notified of failures.\n\n\n--self\ntype=bool-set\nClose the tab of the window this command is run in, rather than the active tab.\n\n\n--ignore-no-match\ntype=bool-set\nDo not return an error if no tabs are matched to be closed.\n"

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        return {'match': opts.match, 'self': opts.self, 'ignore_no_match': opts.ignore_no_match}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            while True:
                i = 10
        try:
            tabs = self.tabs_for_match_payload(boss, window, payload_get)
        except MatchError:
            if payload_get('ignore_no_match'):
                return None
            raise
        for tab in tuple(tabs):
            if tab:
                boss.close_tab_no_confirm(tab)
        return None
close_tab = CloseTab()