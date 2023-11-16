from typing import TYPE_CHECKING, Dict, Optional
from kitty.rgb import to_color
from .base import MATCH_TAB_OPTION, ArgsType, Boss, ParsingOfArgsFailed, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import SetTabColorRCOptions as CLIOptions
valid_color_names = frozenset('active_fg active_bg inactive_fg inactive_bg'.split())

def parse_colors(args: ArgsType) -> Dict[str, Optional[int]]:
    if False:
        i = 10
        return i + 15
    ans: Dict[str, Optional[int]] = {}
    for spec in args:
        (key, val) = spec.split('=', 1)
        key = key.lower()
        if key.lower() not in valid_color_names:
            raise KeyError(f'{key} is not a valid color name')
        if val.lower() == 'none':
            col: Optional[int] = None
        else:
            q = to_color(val, validate=True)
            if q is not None:
                col = int(q)
        ans[key.lower()] = col
    return ans

class SetTabColor(RemoteCommand):
    protocol_spec = __doc__ = '\n    colors+/dict.colors: An object mapping names to colors as 24-bit RGB integers. A color value of null indicates it should be unset.\n    match/str: Which tab to change the color of\n    self/bool: Boolean indicating whether to use the tab of the window the command is run in\n    '
    short_desc = 'Change the color of the specified tabs in the tab bar'
    desc = f'\n{short_desc}\n\nThe foreground and background colors when active and inactive can be overridden using this command. The syntax for specifying colors is: active_fg=color active_bg=color inactive_fg=color inactive_bg=color. Where color can be either a color name or a value of the form #rrggbb or the keyword NONE to revert to using the default colors.\n'
    options_spec = MATCH_TAB_OPTION + '\n\n--self\ntype=bool-set\nClose the tab this command is run in, rather than the active tab.\n'
    args = RemoteCommand.Args(spec='COLORS', json_field='colors', minimum_count=1, special_parse='parse_tab_colors(args)')

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            for i in range(10):
                print('nop')
        try:
            colors = parse_colors(args)
        except Exception as err:
            raise ParsingOfArgsFailed(str(err)) from err
        if not colors:
            raise ParsingOfArgsFailed('No colors specified')
        return {'match': opts.match, 'self': opts.self, 'colors': colors}

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            while True:
                i = 10
        colors = payload_get('colors')
        s = {k: None if colors[k] is None else int(colors[k]) for k in valid_color_names if k in colors}
        for tab in self.tabs_for_match_payload(boss, window, payload_get):
            if tab:
                for (k, v) in s.items():
                    setattr(tab, k, v)
                tab.mark_tab_bar_dirty()
        return None
set_tab_color = SetTabColor()