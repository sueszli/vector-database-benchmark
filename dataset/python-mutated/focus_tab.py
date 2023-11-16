from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import FocusTabRCOptions as CLIOptions

class FocusTab(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: The tab to focus\n    '
    short_desc = 'Focus the specified tab'
    desc = 'The active window in the specified tab will be focused.'
    options_spec = MATCH_TAB_OPTION + "\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response indicating the success of the action. Note that\nusing this option means that you will not be notified of failures.\n"

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            i = 10
            return i + 15
        return {'match': opts.match, 'no_response': opts.no_response}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            for i in range(10):
                print('nop')
        for tab in self.tabs_for_match_payload(boss, window, payload_get):
            if tab:
                boss.set_active_tab(tab)
                break
        return None
focus_tab = FocusTab()