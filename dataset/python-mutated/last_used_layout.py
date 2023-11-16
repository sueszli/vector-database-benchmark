from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import LastUsedLayoutRCOptions as CLIOptions

class LastUsedLayout(RemoteCommand):
    protocol_spec = __doc__ = '\n    match/str: Which tab to change the layout of\n    all/bool: Boolean to match all tabs\n    '
    short_desc = 'Switch to the last used layout'
    desc = 'Switch to the last used window layout in the specified tabs (or the active tab if not specified).'
    options_spec = "--all -a\ntype=bool-set\nChange the layout in all tabs.\n\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response from kitty. This means that even if no matching tab is found,\nthe command will exit with a success code.\n" + '\n\n\n' + MATCH_TAB_OPTION

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            while True:
                i = 10
        return {'match': opts.match, 'all': opts.all, 'no_response': opts.no_response}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            for i in range(10):
                print('nop')
        for tab in self.tabs_for_match_payload(boss, window, payload_get):
            if tab:
                tab.last_used_layout()
        return None
last_used_layout = LastUsedLayout()