from typing import TYPE_CHECKING, Optional
from .base import MATCH_TAB_OPTION, ArgsType, Boss, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import NewWindowRCOptions as CLIOptions

class NewWindow(RemoteCommand):
    protocol_spec = __doc__ = '\n    args+/list.str: The command line to run in the new window, as a list, use an empty list to run the default shell\n    match/str: The tab to open the new window in\n    title/str: Title for the new window\n    cwd/str: Working directory for the new window\n    keep_focus/bool: Boolean indicating whether the current window should retain focus or not\n    window_type/choices.kitty.os: One of :code:`kitty` or :code:`os`\n    new_tab/bool: Boolean indicating whether to open a new tab\n    tab_title/str: Title for the new tab\n    '
    short_desc = 'Open new window'
    desc = 'DEPRECATED: Use the :ref:`launch <at-launch>` command instead.\n\nOpen a new window in the specified tab. If you use the :option:`kitten @ new-window --match` option the first matching tab is used. Otherwise the currently active tab is used. Prints out the id of the newly opened window (unless :option:`--no-response` is used). Any command line arguments are assumed to be the command line used to run in the new window, if none are provided, the default shell is run. For example::\n\n    kitten @ new-window --title Email mutt'
    options_spec = MATCH_TAB_OPTION + "\n\n--title\nThe title for the new window. By default it will use the title set by the\nprogram running in it.\n\n\n--cwd\nThe initial working directory for the new window. Defaults to whatever\nthe working directory for the kitty process you are talking to is.\n\n\n--keep-focus --dont-take-focus\ntype=bool-set\nKeep the current window focused instead of switching to the newly opened window.\n\n\n--window-type\ndefault=kitty\nchoices=kitty,os\nWhat kind of window to open. A kitty window or a top-level OS window.\n\n\n--new-tab\ntype=bool-set\nOpen a new tab.\n\n\n--tab-title\nSet the title of the tab, when open a new tab.\n\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response giving the id of the newly opened window. Note that\nusing this option means that you will not be notified of failures and that\nthe id of the new window will not be printed out.\n"
    args = RemoteCommand.Args(spec='[CMD ...]', json_field='args')

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            return 10
        ans = {'args': args or [], 'type': 'window'}
        for (attr, val) in opts.__dict__.items():
            if attr == 'new_tab':
                if val:
                    ans['type'] = 'tab'
            elif attr == 'window_type':
                if val == 'os' and ans['type'] != 'tab':
                    ans['type'] = 'os-window'
            else:
                ans[attr] = val
        return ans

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            i = 10
            return i + 15
        from .launch import launch
        return launch.response_from_kitty(boss, window, payload_get)
new_window = NewWindow()