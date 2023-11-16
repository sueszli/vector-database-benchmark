"""A HintManager to draw hints over links."""
import collections
import functools
import os
import re
import html
import enum
import dataclasses
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator, List, Mapping, MutableSequence, Optional, Sequence, Set
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, QObject, Qt, QUrl
from qutebrowser.qt.widgets import QLabel
from qutebrowser.config import config, configexc
from qutebrowser.keyinput import modeman, modeparsers, basekeyparser
from qutebrowser.browser import webelem, history
from qutebrowser.commands import runners
from qutebrowser.api import cmdutils
from qutebrowser.utils import usertypes, log, qtutils, message, objreg, utils, urlutils
if TYPE_CHECKING:
    from qutebrowser.browser import browsertab

class Target(enum.Enum):
    """What action to take on a hint."""
    normal = enum.auto()
    current = enum.auto()
    tab = enum.auto()
    tab_fg = enum.auto()
    tab_bg = enum.auto()
    window = enum.auto()
    yank = enum.auto()
    yank_primary = enum.auto()
    run = enum.auto()
    fill = enum.auto()
    hover = enum.auto()
    download = enum.auto()
    userscript = enum.auto()
    spawn = enum.auto()
    delete = enum.auto()
    right_click = enum.auto()

class HintingError(Exception):
    """Exception raised on errors during hinting."""

def on_mode_entered(mode: usertypes.KeyMode, win_id: int) -> None:
    if False:
        i = 10
        return i + 15
    'Stop hinting when insert mode was entered.'
    if mode == usertypes.KeyMode.insert:
        modeman.leave(win_id, usertypes.KeyMode.hint, 'insert mode', maybe=True)

class HintLabel(QLabel):
    """A label for a link.

    Attributes:
        elem: The element this label belongs to.
        _context: The current hinting context.
    """

    def __init__(self, elem: webelem.AbstractWebElement, context: 'HintContext') -> None:
        if False:
            return 10
        super().__init__(parent=context.tab)
        self._context = context
        self.elem = elem
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setIndent(0)
        self._context.tab.contents_size_changed.connect(self._move_to_elem)
        self._move_to_elem()
        self.show()

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        try:
            text = self.text()
        except RuntimeError:
            text = '<deleted>'
        return utils.get_repr(self, elem=self.elem, text=text)

    def update_text(self, matched: str, unmatched: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the text for the hint.\n\n        Args:\n            matched: The part of the text which was typed.\n            unmatched: The part of the text which was not typed yet.\n        '
        if config.cache['hints.uppercase'] and self._context.hint_mode in ['letter', 'word']:
            matched = html.escape(matched.upper())
            unmatched = html.escape(unmatched.upper())
        else:
            matched = html.escape(matched)
            unmatched = html.escape(unmatched)
        if matched:
            match_color = config.cache['colors.hints.match.fg'].name()
            self.setText('<font color="{}">{}</font>{}'.format(match_color, matched, unmatched))
        else:
            self.setText(unmatched)
        self.adjustSize()

    @pyqtSlot()
    def _move_to_elem(self) -> None:
        if False:
            i = 10
            return i + 15
        'Reposition the label to its element.'
        if not self.elem.has_frame():
            log.hints.debug('Frame for {!r} vanished!'.format(self))
            self.hide()
            return
        no_js = config.cache['hints.find_implementation'] != 'javascript'
        rect = self.elem.rect_on_view(no_js=no_js)
        self.move(rect.x(), rect.y())

    def cleanup(self) -> None:
        if False:
            i = 10
            return i + 15
        'Clean up this element and hide it.'
        self.hide()
        self.deleteLater()

@dataclasses.dataclass
class HintContext:
    """Context namespace used for hinting.

    Attributes:
        all_labels: A list of all HintLabel objects ever created.
        labels: A mapping from key strings to HintLabel objects.
                May contain less elements than `all_labels` due to filtering.
        baseurl: The URL of the current page.
        target: What to do with the opened links.
                normal/current/tab/tab_fg/tab_bg/window: Get passed to
                                                         BrowserTab.
                right_click: Right-click the selected element.
                yank/yank_primary: Yank to clipboard/primary selection.
                run: Run a command.
                fill: Fill commandline with link.
                download: Download the link.
                userscript: Call a custom userscript.
                spawn: Spawn a simple command.
                delete: Delete the selected element.
        to_follow: The link to follow when enter is pressed.
        args: Custom arguments for userscript/spawn
        rapid: Whether to do rapid hinting.
        first_run: Whether the action is run for the 1st time in rapid hinting.
        add_history: Whether to add yanked or spawned link to the history.
        filterstr: Used to save the filter string for restoring in rapid mode.
        tab: The WebTab object we started hinting in.
        group: The group of web elements to hint.
    """
    tab: 'browsertab.AbstractTab'
    target: Target
    rapid: bool
    hint_mode: str
    add_history: bool
    first: bool
    baseurl: QUrl
    args: List[str]
    group: str
    all_labels: List[HintLabel] = dataclasses.field(default_factory=list)
    labels: Dict[str, HintLabel] = dataclasses.field(default_factory=dict)
    to_follow: Optional[str] = None
    first_run: bool = True
    filterstr: Optional[str] = None

    def get_args(self, urlstr: str) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        'Get the arguments, with {hint-url} replaced by the given URL.'
        args = []
        for arg in self.args:
            arg = arg.replace('{hint-url}', urlstr)
            args.append(arg)
        return args

class HintActions:
    """Actions which can be done after selecting a hint."""

    def __init__(self, win_id: int) -> None:
        if False:
            while True:
                i = 10
        self._win_id = win_id

    def click(self, elem: webelem.AbstractWebElement, context: HintContext) -> None:
        if False:
            while True:
                i = 10
        'Click an element.'
        target_mapping = {Target.normal: usertypes.ClickTarget.normal, Target.current: usertypes.ClickTarget.normal, Target.tab_fg: usertypes.ClickTarget.tab, Target.tab_bg: usertypes.ClickTarget.tab_bg, Target.window: usertypes.ClickTarget.window}
        if config.val.tabs.background:
            target_mapping[Target.tab] = usertypes.ClickTarget.tab_bg
        else:
            target_mapping[Target.tab] = usertypes.ClickTarget.tab
        if context.target in [Target.normal, Target.current]:
            context.tab.scroller.before_jump_requested.emit()
        try:
            if context.target == Target.hover:
                elem.hover()
            elif context.target == Target.right_click:
                elem.right_click()
            elif context.target == Target.current:
                elem.remove_blank_target()
                elem.click(target_mapping[context.target])
            else:
                elem.click(target_mapping[context.target])
        except webelem.Error as e:
            raise HintingError(str(e))

    def yank(self, url: QUrl, context: HintContext) -> None:
        if False:
            print('Hello World!')
        'Yank an element to the clipboard or primary selection.'
        sel = context.target == Target.yank_primary and utils.supports_selection()
        flags = urlutils.FormatOption.ENCODED | urlutils.FormatOption.REMOVE_PASSWORD
        if url.scheme() == 'mailto':
            flags |= urlutils.FormatOption.REMOVE_SCHEME
        urlstr = url.toString(flags)
        new_content = urlstr
        if context.rapid and (not context.first_run):
            try:
                old_content = utils.get_clipboard(selection=sel)
            except utils.ClipboardEmptyError:
                pass
            else:
                new_content = os.linesep.join([old_content, new_content])
        utils.set_clipboard(new_content, selection=sel)
        msg = 'Yanked URL to {}: {}'.format('primary selection' if sel else 'clipboard', urlstr)
        message.info(msg, replace='rapid-hints' if context.rapid else None)

    def run_cmd(self, url: QUrl, context: HintContext) -> None:
        if False:
            while True:
                i = 10
        'Run the command based on a hint URL.'
        urlstr = url.toString(urlutils.FormatOption.ENCODED)
        args = context.get_args(urlstr)
        commandrunner = runners.CommandRunner(self._win_id)
        commandrunner.run_safely(' '.join(args))

    def preset_cmd_text(self, url: QUrl, context: HintContext) -> None:
        if False:
            return 10
        'Preset a commandline text based on a hint URL.'
        urlstr = url.toDisplayString(urlutils.FormatOption.ENCODED)
        args = context.get_args(urlstr)
        text = ' '.join(args)
        if text[0] not in modeparsers.STARTCHARS:
            raise HintingError("Invalid command text '{}'.".format(text))
        cmd = objreg.get('status-command', scope='window', window=self._win_id)
        cmd.cmd_set_text(text)

    def download(self, elem: webelem.AbstractWebElement, context: HintContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Download a hint URL.\n\n        Args:\n            elem: The QWebElement to download.\n            context: The HintContext to use.\n        '
        url = elem.resolve_url(context.baseurl)
        if url is None:
            raise HintingError('No suitable link found for this element.')
        prompt = False if context.rapid else None
        qnam = context.tab.private_api.networkaccessmanager()
        download_manager = objreg.get('qtnetwork-download-manager')
        download_manager.get(url, qnam=qnam, prompt_download_directory=prompt)

    def call_userscript(self, elem: webelem.AbstractWebElement, context: HintContext) -> None:
        if False:
            return 10
        'Call a userscript from a hint.\n\n        Args:\n            elem: The QWebElement to use in the userscript.\n            context: The HintContext to use.\n        '
        from qutebrowser.commands import userscripts
        cmd = context.args[0]
        args = context.args[1:]
        flags = urlutils.FormatOption.ENCODED
        env = {'QUTE_MODE': 'hints', 'QUTE_SELECTED_TEXT': str(elem), 'QUTE_SELECTED_HTML': elem.outer_xml(), 'QUTE_CURRENT_URL': context.baseurl.toString(flags)}
        url = elem.resolve_url(context.baseurl)
        if url is not None:
            env['QUTE_URL'] = url.toString(flags)
        try:
            userscripts.run_async(context.tab, cmd, *args, win_id=self._win_id, env=env)
        except userscripts.Error as e:
            raise HintingError(str(e))

    def delete(self, elem: webelem.AbstractWebElement, _context: HintContext) -> None:
        if False:
            return 10
        elem.delete()

    def spawn(self, url: QUrl, context: HintContext) -> None:
        if False:
            while True:
                i = 10
        'Spawn a simple command from a hint.\n\n        Args:\n            url: The URL to open as a QUrl.\n            context: The HintContext to use.\n        '
        urlstr = url.toString(QUrl.ComponentFormattingOption.FullyEncoded | QUrl.UrlFormattingOption.RemovePassword)
        args = context.get_args(urlstr)
        commandrunner = runners.CommandRunner(self._win_id)
        commandrunner.run_safely('spawn ' + ' '.join(args))
_ElemsType = Sequence[webelem.AbstractWebElement]
_HintStringsType = MutableSequence[str]

class HintManager(QObject):
    """Manage drawing hints over links or other elements.

    Class attributes:
        HINT_TEXTS: Text displayed for different hinting modes.

    Attributes:
        _context: The HintContext for the current invocation.
        _win_id: The window ID this HintManager is associated with.
        _tab_id: The tab ID this HintManager is associated with.

    Signals:
        set_text: Request for the statusbar to change its text.
    """
    HINT_TEXTS = {Target.normal: 'Follow hint', Target.current: 'Follow hint in current tab', Target.tab: 'Follow hint in new tab', Target.tab_fg: 'Follow hint in foreground tab', Target.tab_bg: 'Follow hint in background tab', Target.window: 'Follow hint in new window', Target.yank: 'Yank hint to clipboard', Target.yank_primary: 'Yank hint to primary selection', Target.run: 'Run a command on a hint', Target.fill: 'Set hint in commandline', Target.hover: 'Hover over a hint', Target.right_click: 'Right-click hint', Target.download: 'Download hint', Target.userscript: 'Call userscript via hint', Target.spawn: 'Spawn command via hint', Target.delete: 'Delete an element'}
    set_text = pyqtSignal(str)

    def __init__(self, win_id: int, parent: QObject=None) -> None:
        if False:
            while True:
                i = 10
        'Constructor.'
        super().__init__(parent)
        self._win_id = win_id
        self._context: Optional[HintContext] = None
        self._word_hinter = WordHinter()
        self._actions = HintActions(win_id)
        mode_manager = modeman.instance(self._win_id)
        mode_manager.left.connect(self.on_mode_left)

    def _get_text(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get a hint text based on the current context.'
        assert self._context is not None
        text = self.HINT_TEXTS[self._context.target]
        if self._context.rapid:
            text += ' (rapid mode)'
        text += '...'
        return text

    def _cleanup(self) -> None:
        if False:
            return 10
        'Clean up after hinting.'
        assert self._context is not None
        for label in self._context.all_labels:
            label.cleanup()
        self.set_text.emit('')
        self._context = None

    def _hint_strings(self, elems: _ElemsType) -> _HintStringsType:
        if False:
            i = 10
            return i + 15
        'Calculate the hint strings for elems.\n\n        Inspired by Vimium.\n\n        Args:\n            elems: The elements to get hint strings for.\n\n        Return:\n            A list of hint strings, in the same order as the elements.\n        '
        if not elems:
            return []
        assert self._context is not None
        hint_mode = self._context.hint_mode
        if hint_mode == 'word':
            try:
                return self._word_hinter.hint(elems)
            except HintingError as e:
                message.error(str(e))
        if hint_mode == 'number':
            chars = '0123456789'
        else:
            chars = config.val.hints.chars
        min_chars = config.val.hints.min_chars
        if config.val.hints.scatter and hint_mode != 'number':
            return self._hint_scattered(min_chars, chars, elems)
        else:
            return self._hint_linear(min_chars, chars, elems)

    def _hint_scattered(self, min_chars: int, chars: str, elems: _ElemsType) -> _HintStringsType:
        if False:
            print('Hello World!')
        'Produce scattered hint labels with variable length (like Vimium).\n\n        Args:\n            min_chars: The minimum length of labels.\n            chars: The alphabet to use for labels.\n            elems: The elements to generate labels for.\n        '
        needed = max(min_chars, utils.ceil_log(len(elems), len(chars)))
        if needed > min_chars and needed > 1:
            total_space = len(chars) ** needed
            short_count = (total_space - len(elems)) // (len(chars) - 1)
        else:
            short_count = 0
        long_count = len(elems) - short_count
        strings = []
        if needed > 1:
            for i in range(short_count):
                strings.append(self._number_to_hint_str(i, chars, needed - 1))
        start = short_count * len(chars)
        for i in range(start, start + long_count):
            strings.append(self._number_to_hint_str(i, chars, needed))
        return self._shuffle_hints(strings, len(chars))

    def _hint_linear(self, min_chars: int, chars: str, elems: _ElemsType) -> _HintStringsType:
        if False:
            return 10
        'Produce linear hint labels with constant length (like dwb).\n\n        Args:\n            min_chars: The minimum length of labels.\n            chars: The alphabet to use for labels.\n            elems: The elements to generate labels for.\n        '
        strings = []
        needed = max(min_chars, utils.ceil_log(len(elems), len(chars)))
        for i in range(len(elems)):
            strings.append(self._number_to_hint_str(i, chars, needed))
        return strings

    def _shuffle_hints(self, hints: _HintStringsType, length: int) -> _HintStringsType:
        if False:
            return 10
        "Shuffle the given set of hints so that they're scattered.\n\n        Hints starting with the same character will be spread evenly throughout\n        the array.\n\n        Inspired by Vimium.\n\n        Args:\n            hints: A list of hint strings.\n            length: Length of the available charset.\n\n        Return:\n            A list of shuffled hint strings.\n        "
        buckets: Sequence[_HintStringsType] = [[] for i in range(length)]
        for (i, hint) in enumerate(hints):
            buckets[i % len(buckets)].append(hint)
        result: _HintStringsType = []
        for bucket in buckets:
            result += bucket
        return result

    def _number_to_hint_str(self, number: int, chars: str, digits: int=0) -> str:
        if False:
            print('Hello World!')
        'Convert a number like "8" into a hint string like "JK".\n\n        This is used to sequentially generate all of the hint text.\n        The hint string will be "padded with zeroes" to ensure its length is >=\n        digits.\n\n        Inspired by Vimium.\n\n        Args:\n            number: The hint number.\n            chars: The charset to use.\n            digits: The minimum output length.\n\n        Return:\n            A hint string.\n        '
        base = len(chars)
        hintstr: MutableSequence[str] = []
        remainder = 0
        while True:
            remainder = number % base
            hintstr.insert(0, chars[remainder])
            number -= remainder
            number //= base
            if number <= 0:
                break
        for _ in range(0, digits - len(hintstr)):
            hintstr.insert(0, chars[0])
        return ''.join(hintstr)

    def _check_args(self, target: Target, *args: str) -> None:
        if False:
            print('Hello World!')
        "Check the arguments passed to start() and raise if they're wrong.\n\n        Args:\n            target: A Target enum member.\n            args: Arguments for userscript/download\n        "
        if not isinstance(target, Target):
            raise TypeError('Target {} is no Target member!'.format(target))
        if target in [Target.userscript, Target.spawn, Target.run, Target.fill]:
            if not args:
                raise cmdutils.CommandError("'args' is required with target userscript/spawn/run/fill.")
        elif args:
            raise cmdutils.CommandError("'args' is only allowed with target userscript/spawn.")

    def _filter_matches(self, filterstr: Optional[str], elemstr: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if `filterstr` matches `elemstr`.'
        if not filterstr:
            return True
        filterstr = filterstr.casefold()
        elemstr = elemstr.casefold()
        return all((word in elemstr for word in filterstr.split()))

    def _filter_matches_exactly(self, filterstr: str, elemstr: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return True if `filterstr` exactly matches `elemstr`.'
        if not filterstr:
            return False
        filterstr = filterstr.casefold()
        elemstr = elemstr.casefold()
        return filterstr == elemstr

    def _get_keyparser(self, mode: usertypes.KeyMode) -> basekeyparser.BaseKeyParser:
        if False:
            return 10
        mode_manager = modeman.instance(self._win_id)
        return mode_manager.parsers[mode]

    def _start_cb(self, elems: _ElemsType) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the elements and labels based on the context set.'
        if self._context is None:
            log.hints.debug('In _start_cb without context!')
            return
        if not elems:
            message.error('No elements found.')
            return
        tabbed_browser = objreg.get('tabbed-browser', default=None, scope='window', window=self._win_id)
        tab = tabbed_browser.widget.currentWidget()
        if tab.tab_id != self._context.tab.tab_id:
            log.hints.debug('Current tab changed ({} -> {}) before _start_cb is run.'.format(self._context.tab.tab_id, tab.tab_id))
            return
        strings = self._hint_strings(elems)
        log.hints.debug('hints: {}'.format(', '.join(strings)))
        for (elem, string) in zip(elems, strings):
            label = HintLabel(elem, self._context)
            label.update_text('', string)
            self._context.all_labels.append(label)
            self._context.labels[string] = label
        keyparser = self._get_keyparser(usertypes.KeyMode.hint)
        assert isinstance(keyparser, modeparsers.HintKeyParser), keyparser
        keyparser.update_bindings(strings)
        modeman.enter(self._win_id, usertypes.KeyMode.hint, 'HintManager.start')
        self.set_text.emit(self._get_text())
        if self._context.first:
            self._fire(strings[0])
            return
        self._handle_auto_follow()

    @cmdutils.register(instance='hintmanager', scope='window', name='hint', star_args_optional=True, maxsplit=2)
    def start(self, group: str='all', target: Target=Target.normal, *args: str, mode: str=None, add_history: bool=False, rapid: bool=False, first: bool=False) -> None:
        if False:
            print('Hello World!')
        "Start hinting.\n\n        Args:\n            rapid: Whether to do rapid hinting. With rapid hinting, the hint\n                   mode isn't left after a hint is followed, so you can easily\n                   open multiple links. Note this won't work with targets\n                   `tab-fg`, `fill`, `delete` and `right-click`.\n            add_history: Whether to add the spawned or yanked link to the\n                         browsing history.\n            first: Click the first hinted element without prompting.\n            group: The element types to hint.\n\n                - `all`: All clickable elements.\n                - `links`: Only links.\n                - `images`: Only images.\n                - `inputs`: Only input fields.\n\n                Custom groups can be added via the `hints.selectors` setting\n                and also used here.\n\n            target: What to do with the selected element.\n\n                - `normal`: Open the link.\n                - `current`: Open the link in the current tab.\n                - `tab`: Open the link in a new tab (honoring the\n                         `tabs.background` setting).\n                - `tab-fg`: Open the link in a new foreground tab.\n                - `tab-bg`: Open the link in a new background tab.\n                - `window`: Open the link in a new window.\n                - `hover` : Hover over the link.\n                - `right-click`: Right-click the element.\n                - `yank`: Yank the link to the clipboard.\n                - `yank-primary`: Yank the link to the primary selection.\n                - `run`: Run the argument as command.\n                - `fill`: Fill the commandline with the command given as\n                          argument.\n                - `download`: Download the link.\n                - `userscript`: Call a userscript with `$QUTE_URL` set to the\n                                link.\n                - `spawn`: Spawn a command.\n                - `delete`: Delete the selected element.\n\n            mode: The hinting mode to use.\n\n                - `number`: Use numeric hints.\n                - `letter`: Use the chars in the hints.chars setting.\n                - `word`: Use hint words based on the html elements and the\n                          extra words.\n\n            *args: Arguments for spawn/userscript/run/fill.\n\n                - With `spawn`: The executable and arguments to spawn.\n                                `{hint-url}` will get replaced by the selected\n                                URL.\n                - With `userscript`: The userscript to execute. Either store\n                                     the userscript in\n                                     `~/.local/share/qutebrowser/userscripts`\n                                     (or `$XDG_DATA_HOME`), or use an absolute\n                                     path.\n                - With `fill`: The command to fill the statusbar with.\n                                `{hint-url}` will get replaced by the selected\n                                URL.\n                - With `run`: Same as `fill`.\n        "
        tabbed_browser = objreg.get('tabbed-browser', scope='window', window=self._win_id)
        tab = tabbed_browser.widget.currentWidget()
        if tab is None:
            raise cmdutils.CommandError('No WebView available yet!')
        mode_manager = modeman.instance(self._win_id)
        if mode_manager.mode == usertypes.KeyMode.hint:
            modeman.leave(self._win_id, usertypes.KeyMode.hint, 're-hinting')
        no_rapid_targets = [Target.tab_fg, Target.fill, Target.right_click, Target.delete]
        if rapid and target in no_rapid_targets:
            name = target.name.replace('_', '-')
            raise cmdutils.CommandError(f'Rapid hinting makes no sense with target {name}!')
        self._check_args(target, *args)
        try:
            baseurl = tabbed_browser.current_url()
        except qtutils.QtValueError:
            raise cmdutils.CommandError('No URL set for this page yet!')
        self._context = HintContext(tab=tab, target=target, rapid=rapid, hint_mode=self._get_hint_mode(mode), add_history=add_history, first=first, baseurl=baseurl, args=list(args), group=group)
        try:
            selector = webelem.css_selector(self._context.group, self._context.baseurl)
        except webelem.Error as e:
            raise cmdutils.CommandError(str(e))
        self._context.tab.elements.find_css(selector, callback=self._start_cb, error_cb=lambda err: message.error(str(err)), only_visible=True)

    def _get_hint_mode(self, mode: Optional[str]) -> str:
        if False:
            i = 10
            return i + 15
        'Get the hinting mode to use based on a mode argument.'
        if mode is None:
            return config.val.hints.mode
        opt = config.instance.get_opt('hints.mode')
        try:
            opt.typ.to_py(mode)
        except configexc.ValidationError as e:
            raise cmdutils.CommandError('Invalid mode: {}'.format(e))
        return mode

    def current_mode(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'Return the currently active hinting mode (or None otherwise).'
        if self._context is None:
            return None
        return self._context.hint_mode

    def _handle_auto_follow(self, keystr: str='', filterstr: str='', visible: Mapping[str, HintLabel]=None) -> None:
        if False:
            print('Hello World!')
        'Handle the auto_follow option.'
        assert self._context is not None
        if visible is None:
            visible = {string: label for (string, label) in self._context.labels.items() if label.isVisible()}
        if len(visible) != 1:
            return
        auto_follow = config.val.hints.auto_follow
        if auto_follow == 'always':
            follow = True
        elif auto_follow == 'unique-match':
            follow = bool(keystr or filterstr)
        elif auto_follow == 'full-match':
            elemstr = str(list(visible.values())[0].elem)
            filter_match = self._filter_matches_exactly(filterstr, elemstr)
            follow = keystr in visible or filter_match
        else:
            follow = False
            self._context.to_follow = list(visible.keys())[0]
        if follow:
            timeout = config.val.hints.auto_follow_timeout
            normal_parser = self._get_keyparser(usertypes.KeyMode.normal)
            assert isinstance(normal_parser, modeparsers.NormalKeyParser), normal_parser
            normal_parser.set_inhibited_timeout(timeout)
            self._fire(*visible)

    @pyqtSlot(str)
    def handle_partial_key(self, keystr: str) -> None:
        if False:
            return 10
        'Handle a new partial keypress.'
        if self._context is None:
            log.hints.debug('Got key without context!')
            return
        log.hints.debug("Handling new keystring: '{}'".format(keystr))
        for (string, label) in self._context.labels.items():
            try:
                if string.startswith(keystr):
                    matched = string[:len(keystr)]
                    rest = string[len(keystr):]
                    label.update_text(matched, rest)
                    label.show()
                elif not self._context.rapid or config.val.hints.hide_unmatched_rapid_hints:
                    label.hide()
            except webelem.Error:
                pass
        self._handle_auto_follow(keystr=keystr)

    def filter_hints(self, filterstr: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Filter displayed hints according to a text.\n\n        Args:\n            filterstr: The string to filter with, or None to use the filter\n                       from previous call (saved in `self._context.filterstr`).\n                       If `filterstr` is an empty string or if both `filterstr`\n                       and `self._context.filterstr` are None, all hints are\n                       shown.\n        '
        assert self._context is not None
        if filterstr is None:
            filterstr = self._context.filterstr
        else:
            self._context.filterstr = filterstr
        log.hints.debug('Filtering hints on {!r}'.format(filterstr))
        visible = []
        for label in self._context.all_labels:
            try:
                if self._filter_matches(filterstr, str(label.elem)):
                    visible.append(label)
                    label.show()
                else:
                    label.hide()
            except webelem.Error:
                pass
        if not visible:
            modeman.leave(self._win_id, usertypes.KeyMode.hint, 'all filtered')
            return
        if self._context.hint_mode == 'number':
            strings = self._hint_strings([label.elem for label in visible])
            self._context.labels = {}
            for (label, string) in zip(visible, strings):
                label.update_text('', string)
                self._context.labels[string] = label
            keyparser = self._get_keyparser(usertypes.KeyMode.hint)
            assert isinstance(keyparser, modeparsers.HintKeyParser), keyparser
            keyparser.update_bindings(strings, preserve_filter=True)
            if filterstr is not None:
                self._handle_auto_follow(filterstr=filterstr, visible=self._context.labels)

    def _fire(self, keystr: str) -> None:
        if False:
            while True:
                i = 10
        'Fire a completed hint.\n\n        Args:\n            keystr: The keychain string to follow.\n        '
        assert self._context is not None
        elem_handlers = {Target.normal: self._actions.click, Target.current: self._actions.click, Target.tab: self._actions.click, Target.tab_fg: self._actions.click, Target.tab_bg: self._actions.click, Target.window: self._actions.click, Target.hover: self._actions.click, Target.right_click: self._actions.click, Target.download: self._actions.download, Target.userscript: self._actions.call_userscript, Target.delete: self._actions.delete}
        url_handlers = {Target.yank: self._actions.yank, Target.yank_primary: self._actions.yank, Target.run: self._actions.run_cmd, Target.fill: self._actions.preset_cmd_text, Target.spawn: self._actions.spawn}
        elem = self._context.labels[keystr].elem
        if not elem.has_frame():
            message.error('This element has no webframe.')
            return
        if self._context.target in elem_handlers:
            handler = functools.partial(elem_handlers[self._context.target], elem, self._context)
        elif self._context.target in url_handlers:
            url = elem.resolve_url(self._context.baseurl)
            if url is None:
                message.error('No suitable link found for this element.')
                return
            handler = functools.partial(url_handlers[self._context.target], url, self._context)
            if self._context.add_history:
                history.web_history.add_url(url, '')
        else:
            raise ValueError('No suitable handler found!')
        if not self._context.rapid:
            modeman.leave(self._win_id, usertypes.KeyMode.hint, 'followed', maybe=True)
        else:
            self.filter_hints(None)
            for (string, label) in self._context.labels.items():
                label.update_text('', string)
        try:
            handler()
        except HintingError as e:
            message.error(str(e))
        if self._context is not None:
            self._context.first_run = False

    @cmdutils.register(instance='hintmanager', scope='window', modes=[usertypes.KeyMode.hint])
    def hint_follow(self, select: bool=False, keystring: str=None) -> None:
        if False:
            print('Hello World!')
        "Follow a hint.\n\n        Args:\n            select: Only select the given hint, don't necessarily follow it.\n            keystring: The hint to follow, or None.\n        "
        assert self._context is not None
        if keystring is None:
            if self._context.to_follow is None:
                raise cmdutils.CommandError('No hint to follow')
            if select:
                raise cmdutils.CommandError("Can't use --select without hint.")
            keystring = self._context.to_follow
        elif keystring not in self._context.labels:
            raise cmdutils.CommandError('No hint {}!'.format(keystring))
        if select:
            self.handle_partial_key(keystring)
        else:
            self._fire(keystring)

    @pyqtSlot(usertypes.KeyMode)
    def on_mode_left(self, mode: usertypes.KeyMode) -> None:
        if False:
            while True:
                i = 10
        'Stop hinting when hinting mode was left.'
        if mode != usertypes.KeyMode.hint or self._context is None:
            return
        self._cleanup()

class WordHinter:
    """Generator for word hints.

    Attributes:
        words: A set of words to be used when no "smart hint" can be
            derived from the hinted element.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.words: Set[str] = set()
        self.dictionary = None

    def ensure_initialized(self) -> None:
        if False:
            return 10
        'Generate the used words if yet uninitialized.'
        dictionary = config.val.hints.dictionary
        if not self.words or self.dictionary != dictionary:
            self.words.clear()
            self.dictionary = dictionary
            try:
                with open(dictionary, encoding='UTF-8') as wordfile:
                    alphabet = set(ascii_lowercase)
                    hints = set()
                    lines = (line.rstrip().lower() for line in wordfile)
                    for word in lines:
                        if set(word) - alphabet:
                            continue
                        if len(word) > 4:
                            continue
                        for i in range(len(word)):
                            hints.discard(word[:i + 1])
                        hints.add(word)
                    self.words.update(hints)
            except OSError as e:
                error = 'Word hints requires reading the file at {}: {}'
                raise HintingError(error.format(dictionary, str(e)))
            except UnicodeDecodeError as e:
                error = 'Word hints expects the file at {} to be encoded as UTF-8: {}'
                raise HintingError(error.format(dictionary, str(e)))

    def extract_tag_words(self, elem: webelem.AbstractWebElement) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        'Extract tag words form the given element.'
        _extractor_type = Callable[[webelem.AbstractWebElement], str]
        attr_extractors: Mapping[str, _extractor_type] = {'alt': lambda elem: elem['alt'], 'name': lambda elem: elem['name'], 'title': lambda elem: elem['title'], 'placeholder': lambda elem: elem['placeholder'], 'src': lambda elem: elem['src'].split('/')[-1], 'href': lambda elem: elem['href'].split('/')[-1], 'text': str}
        extractable_attrs = collections.defaultdict(list, {'img': ['alt', 'title', 'src'], 'a': ['title', 'href', 'text'], 'input': ['name', 'placeholder'], 'textarea': ['name', 'placeholder'], 'button': ['text']})
        return (attr_extractors[attr](elem) for attr in extractable_attrs[elem.tag_name()] if attr in elem or attr == 'text')

    def tag_words_to_hints(self, words: Iterable[str]) -> Iterator[str]:
        if False:
            return 10
        'Take words and transform them to proper hints if possible.'
        for candidate in words:
            if not candidate:
                continue
            match = re.search('[A-Za-z]{3,}', candidate)
            if not match:
                continue
            if 4 < match.end() - match.start() < 8:
                yield candidate[match.start():match.end()].lower()

    def any_prefix(self, hint: str, existing: Iterable[str]) -> bool:
        if False:
            i = 10
            return i + 15
        return any((hint.startswith(e) or e.startswith(hint) for e in existing))

    def filter_prefixes(self, hints: Iterable[str], existing: Iterable[str]) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        "Filter hints which don't start with the given prefix."
        return (h for h in hints if not self.any_prefix(h, existing))

    def new_hint_for(self, elem: webelem.AbstractWebElement, existing: Iterable[str], fallback: Iterable[str]) -> Optional[str]:
        if False:
            while True:
                i = 10
        'Return a hint for elem, not conflicting with the existing.'
        new = self.tag_words_to_hints(self.extract_tag_words(elem))
        new_no_prefixes = self.filter_prefixes(new, existing)
        fallback_no_prefixes = self.filter_prefixes(fallback, existing)
        return next(new_no_prefixes, None) or next(fallback_no_prefixes, None)

    def hint(self, elems: _ElemsType) -> _HintStringsType:
        if False:
            for i in range(10):
                print('nop')
        'Produce hint labels based on the html tags.\n\n        Produce hint words based on the link text and random words\n        from the words arg as fallback.\n\n        Args:\n            elems: The elements to get hint strings for.\n\n        Return:\n            A list of hint strings, in the same order as the elements.\n        '
        self.ensure_initialized()
        hints = []
        used_hints: Set[str] = set()
        words = iter(self.words)
        for elem in elems:
            hint = self.new_hint_for(elem, used_hints, words)
            if not hint:
                raise HintingError('Not enough words in the dictionary.')
            used_hints.add(hint)
            hints.append(hint)
        return hints