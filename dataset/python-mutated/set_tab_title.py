from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import SetTabTitleRCOptions as CLIOptions

class SetTabTitle(RemoteCommand):
    protocol_spec = __doc__ = '\n    title+/str: The new title\n    match/str: Which tab to change the title of\n    '
    short_desc = 'Set the tab title'
    desc = 'Set the title for the specified tabs. If you use the :option:`kitten @ set-tab-title --match` option the title will be set for all matched tabs. By default, only the tab in which the command is run is affected. If you do not specify a title, the title of the currently active window in the tab is used.'
    options_spec = MATCH_TAB_OPTION
    args = RemoteCommand.Args(spec='TITLE ...', json_field='title', special_parse='expand_ansi_c_escapes_in_args(args...)')

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            return 10
        return {'title': ' '.join(args), 'match': opts.match}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            for i in range(10):
                print('nop')
        for tab in self.tabs_for_match_payload(boss, window, payload_get):
            if tab:
                tab.set_title(payload_get('title'))
        return None
set_tab_title = SetTabTitle()