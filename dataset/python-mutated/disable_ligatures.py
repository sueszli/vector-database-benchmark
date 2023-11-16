from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, MATCH_WINDOW_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import DisableLigaturesRCOptions as CLIOptions

class DisableLigatures(RemoteCommand):
    protocol_spec = __doc__ = '\n    strategy+/choices.never.always.cursor: One of :code:`never`, :code:`always` or :code:`cursor`\n    match_window/str: Window to change opacity in\n    match_tab/str: Tab to change opacity in\n    all/bool: Boolean indicating operate on all windows\n    '
    short_desc = 'Control ligature rendering'
    desc = 'Control ligature rendering for the specified windows/tabs (defaults to active window). The :italic:`STRATEGY` can be one of: :code:`never`, :code:`always`, :code:`cursor`.'
    options_spec = '--all -a\ntype=bool-set\nBy default, ligatures are only affected in the active window. This option will\ncause ligatures to be changed in all windows.\n\n' + '\n\n' + MATCH_WINDOW_OPTION + '\n\n' + MATCH_TAB_OPTION.replace('--match -m', '--match-tab -t')
    args = RemoteCommand.Args(spec='STRATEGY', count=1, json_field='strategy')

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            print('Hello World!')
        if not args:
            self.fatal('You must specify the STRATEGY for disabling ligatures, must be one of never, always or cursor')
        strategy = args[0]
        if strategy not in ('never', 'always', 'cursor'):
            self.fatal(f'{strategy} is not a valid disable_ligatures strategy')
        return {'strategy': strategy, 'match_window': opts.match, 'match_tab': opts.match_tab, 'all': opts.all}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            print('Hello World!')
        windows = self.windows_for_payload(boss, window, payload_get)
        boss.disable_ligatures_in(windows, payload_get('strategy'))
        return None
disable_ligatures = DisableLigatures()