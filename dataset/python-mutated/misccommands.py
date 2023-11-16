"""Various commands."""
import os
import signal
import functools
import logging
import pathlib
from typing import Optional, Sequence, Callable
try:
    import hunter
except ImportError:
    hunter = None
from qutebrowser.qt.core import Qt
from qutebrowser.qt.printsupport import QPrintPreviewDialog
from qutebrowser.api import cmdutils, apitypes, message, config
from qutebrowser.completion.models import miscmodels
from qutebrowser.utils import utils
_LOGGER = logging.getLogger('misc')

@cmdutils.register(name='reload')
@cmdutils.argument('tab', value=cmdutils.Value.count_tab)
def reloadpage(tab: Optional[apitypes.Tab], force: bool=False) -> None:
    if False:
        return 10
    'Reload the current/[count]th tab.\n\n    Args:\n        count: The tab index to reload, or None.\n        force: Bypass the page cache.\n    '
    if tab is not None:
        tab.reload(force=force)

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.count_tab)
def stop(tab: Optional[apitypes.Tab]) -> None:
    if False:
        i = 10
        return i + 15
    'Stop loading in the current/[count]th tab.\n\n    Args:\n        count: The tab index to stop, or None.\n    '
    if tab is not None:
        tab.stop()

def _print_preview(tab: apitypes.Tab) -> None:
    if False:
        return 10
    'Show a print preview.'

    def print_callback(ok: bool) -> None:
        if False:
            print('Hello World!')
        if not ok:
            message.error('Printing failed!')
    tab.printing.check_preview_support()
    diag = QPrintPreviewDialog(tab)
    diag.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    diag.setWindowFlags(diag.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
    diag.paintRequested.connect(functools.partial(tab.printing.to_printer, callback=print_callback))
    diag.exec()

def _print_pdf(tab: apitypes.Tab, path: pathlib.Path) -> None:
    if False:
        print('Hello World!')
    'Print to the given PDF file.'
    tab.printing.check_pdf_support()
    path = path.expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise cmdutils.CommandError(e)
    tab.printing.to_pdf(path)
    _LOGGER.debug(f'Print to file: {path}')

@cmdutils.register(name='print')
@cmdutils.argument('tab', value=cmdutils.Value.count_tab)
@cmdutils.argument('pdf', flag='f', metavar='file')
def printpage(tab: Optional[apitypes.Tab], preview: bool=False, *, pdf: Optional[pathlib.Path]=None) -> None:
    if False:
        i = 10
        return i + 15
    'Print the current/[count]th tab.\n\n    Args:\n        preview: Show preview instead of printing.\n        count: The tab index to print, or None.\n        pdf: The file path to write the PDF to.\n    '
    if tab is None:
        return
    try:
        if preview:
            _print_preview(tab)
        elif pdf:
            _print_pdf(tab, pdf)
        else:
            tab.printing.show_dialog()
    except apitypes.WebTabError as e:
        raise cmdutils.CommandError(e)

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def home(tab: apitypes.Tab) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Open main startpage in current tab.'
    if tab.navigation_blocked():
        message.info('Tab is pinned!')
    else:
        tab.load_url(config.val.url.start_pages[0])

@cmdutils.register(debug=True)
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def debug_dump_page(tab: apitypes.Tab, dest: str, plain: bool=False) -> None:
    if False:
        return 10
    "Dump the current page's content to a file.\n\n    Args:\n        dest: Where to write the file to.\n        plain: Write plain text instead of HTML.\n    "
    dest = os.path.expanduser(dest)

    def callback(data: str) -> None:
        if False:
            return 10
        'Write the data to disk.'
        try:
            with open(dest, 'w', encoding='utf-8') as f:
                f.write(data)
        except OSError as e:
            message.error('Could not write page: {}'.format(e))
        else:
            message.info('Dumped page to {}.'.format(dest))
    tab.dump_async(callback, plain=plain)

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def screenshot(tab: apitypes.Tab, filename: pathlib.Path, *, rect: str=None, force: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    'Take a screenshot of the currently shown part of the page.\n\n    The file format is automatically determined based on the given file extension.\n\n    Args:\n        filename: The file to save the screenshot to (~ gets expanded).\n        rect: The rectangle to save, as a string like WxH+X+Y.\n        force: Overwrite existing files.\n    '
    expanded = filename.expanduser()
    if expanded.exists() and (not force):
        raise cmdutils.CommandError(f'File {filename} already exists (use --force to overwrite)')
    try:
        qrect = None if rect is None else utils.parse_rect(rect)
    except ValueError as e:
        raise cmdutils.CommandError(str(e))
    pic = tab.grab_pixmap(qrect)
    if pic is None:
        raise cmdutils.CommandError('Getting screenshot failed')
    ok = pic.save(str(expanded))
    if not ok:
        raise cmdutils.CommandError(f'Saving to {filename} failed')
    _LOGGER.debug(f'Screenshot saved to {filename}')

@cmdutils.register(maxsplit=0)
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def insert_text(tab: apitypes.Tab, text: str) -> None:
    if False:
        while True:
            i = 10
    'Insert text at cursor position.\n\n    Args:\n        text: The text to insert.\n    '

    def _insert_text_cb(elem: Optional[apitypes.WebElement]) -> None:
        if False:
            i = 10
            return i + 15
        if elem is None:
            message.error('No element focused!')
            return
        try:
            elem.insert_text(text)
        except apitypes.WebElemError as e:
            message.error(str(e))
            return
    tab.elements.find_focused(_insert_text_cb)

def _wrap_find_at_pos(value: str, tab: apitypes.Tab, callback: Callable[[Optional[apitypes.WebElement]], None]) -> None:
    if False:
        while True:
            i = 10
    try:
        point = utils.parse_point(value)
    except ValueError as e:
        message.error(str(e))
        return
    tab.elements.find_at_pos(point, callback)
_FILTER_ERRORS = {'id': lambda x: f'with ID "{x}"', 'css': lambda x: f'matching CSS selector "{x}"', 'focused': lambda _: 'with focus', 'position': lambda x: 'at position {x}'}

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('filter_', choices=['id', 'css', 'position', 'focused'])
def click_element(tab: apitypes.Tab, filter_: str, value: str=None, *, target: apitypes.ClickTarget=apitypes.ClickTarget.normal, force_event: bool=False, select_first: bool=False) -> None:
    if False:
        while True:
            i = 10
    "Click the element matching the given filter.\n\n    The given filter needs to result in exactly one element, otherwise, an\n    error is shown.\n\n    Args:\n        filter_: How to filter the elements.\n\n            - id: Get an element based on its ID.\n            - css: Filter by a CSS selector.\n            - position: Click the element at specified position.\n               Specify `value` as 'x,y'.\n            - focused: Click the currently focused element.\n        value: The value to filter for. Optional for 'focused' filter.\n        target: How to open the clicked element (normal/tab/tab-bg/window).\n        force_event: Force generating a fake click event.\n        select_first: Select first matching element if there are multiple.\n    "

    def do_click(elem: apitypes.WebElement) -> None:
        if False:
            i = 10
            return i + 15
        try:
            elem.click(target, force_event=force_event)
        except apitypes.WebElemError as e:
            message.error(str(e))

    def single_cb(elem: Optional[apitypes.WebElement]) -> None:
        if False:
            i = 10
            return i + 15
        'Click a single element.'
        if elem is None:
            message.error(f'No element found {_FILTER_ERRORS[filter_](value)}!')
            return
        do_click(elem)

    def multiple_cb(elems: Sequence[apitypes.WebElement]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not elems:
            message.error(f'No element found {_FILTER_ERRORS[filter_](value)}!')
            return
        if not select_first and len(elems) > 1:
            message.error(f'Multiple elements found {_FILTER_ERRORS[filter_](value)}!')
            return
        do_click(elems[0])
    if value is None and filter_ != 'focused':
        raise cmdutils.CommandError("Argument 'value' is only optional with filter 'focused'!")
    if filter_ == 'id':
        assert value is not None
        tab.elements.find_id(elem_id=value, callback=single_cb)
    elif filter_ == 'css':
        assert value is not None
        tab.elements.find_css(value, callback=multiple_cb, error_cb=lambda exc: message.error(str(exc)))
    elif filter_ == 'position':
        assert value is not None
        _wrap_find_at_pos(value, tab=tab, callback=single_cb)
    elif filter_ == 'focused':
        tab.elements.find_focused(callback=single_cb)
    else:
        raise utils.Unreachable(filter_)

@cmdutils.register(debug=True)
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def debug_webaction(tab: apitypes.Tab, action: str, count: int=1) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Execute a webaction.\n\n    Available actions:\n    https://doc.qt.io/archives/qt-5.5/qwebpage.html#WebAction-enum (WebKit)\n    https://doc.qt.io/qt-6/qwebenginepage.html#WebAction-enum (WebEngine)\n\n    Args:\n        action: The action to execute, e.g. MoveToNextChar.\n        count: How many times to repeat the action.\n    '
    for _ in range(count):
        try:
            tab.action.run_string(action)
        except apitypes.WebTabError as e:
            raise cmdutils.CommandError(str(e))

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.count_tab)
def tab_mute(tab: Optional[apitypes.Tab]) -> None:
    if False:
        print('Hello World!')
    'Mute/Unmute the current/[count]th tab.\n\n    Args:\n        count: The tab index to mute or unmute, or None\n    '
    if tab is None:
        return
    try:
        tab.audio.set_muted(not tab.audio.is_muted(), override=True)
    except apitypes.WebTabError as e:
        raise cmdutils.CommandError(e)

@cmdutils.register()
def nop() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Do nothing.'

@cmdutils.register()
def message_error(text: str, rich: bool=False) -> None:
    if False:
        while True:
            i = 10
    'Show an error message in the statusbar.\n\n    Args:\n        text: The text to show.\n        rich: Render the given text as\n              https://doc.qt.io/qt-6/richtext-html-subset.html[Qt Rich Text].\n    '
    message.error(text, rich=rich)

@cmdutils.register()
@cmdutils.argument('count', value=cmdutils.Value.count)
def message_info(text: str, count: int=1, rich: bool=False) -> None:
    if False:
        print('Hello World!')
    'Show an info message in the statusbar.\n\n    Args:\n        text: The text to show.\n        count: How many times to show the message.\n        rich: Render the given text as\n              https://doc.qt.io/qt-6/richtext-html-subset.html[Qt Rich Text].\n    '
    for _ in range(count):
        message.info(text, rich=rich)

@cmdutils.register()
def message_warning(text: str, rich: bool=False) -> None:
    if False:
        return 10
    'Show a warning message in the statusbar.\n\n    Args:\n        text: The text to show.\n        rich: Render the given text as\n              https://doc.qt.io/qt-6/richtext-html-subset.html[Qt Rich Text].\n    '
    message.warning(text, rich=rich)

@cmdutils.register(debug=True)
@cmdutils.argument('typ', choices=['exception', 'segfault'])
def debug_crash(typ: str='exception') -> None:
    if False:
        while True:
            i = 10
    "Crash for debugging purposes.\n\n    Args:\n        typ: either 'exception' or 'segfault'.\n    "
    if typ == 'segfault':
        os.kill(os.getpid(), signal.SIGSEGV)
        raise Exception('Segfault failed (wat.)')
    raise Exception('Forced crash')

@cmdutils.register(debug=True, maxsplit=0, no_cmd_split=True)
def debug_trace(expr: str='') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Trace executed code via hunter.\n\n    Args:\n        expr: What to trace, passed to hunter.\n    '
    if hunter is None:
        raise cmdutils.CommandError("You need to install 'hunter' to use this command!")
    try:
        eval('hunter.trace({})'.format(expr))
    except Exception as e:
        raise cmdutils.CommandError('{}: {}'.format(e.__class__.__name__, e))

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('position', completion=miscmodels.inspector_position)
def devtools(tab: apitypes.Tab, position: apitypes.InspectorPosition=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Toggle the developer tools (web inspector).\n\n    Args:\n        position: Where to open the devtools\n                  (right/left/top/bottom/window).\n    '
    try:
        tab.private_api.toggle_inspector(position)
    except apitypes.InspectorError as e:
        raise cmdutils.CommandError(e)

@cmdutils.register()
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def devtools_focus(tab: apitypes.Tab) -> None:
    if False:
        while True:
            i = 10
    'Toggle focus between the devtools/tab.'
    assert tab.data.splitter is not None
    try:
        tab.data.splitter.cycle_focus()
    except apitypes.InspectorError as e:
        raise cmdutils.CommandError(e)

@cmdutils.register(name='Ni!')
def knights_who_say_ni() -> None:
    if False:
        print('Hello World!')
    "We are the Knights Who Say... 'Ni'!"
    raise cmdutils.CommandError('Do you demand a shrubbery?')