"""Implementation of :navigate."""
import re
import posixpath
from typing import Optional, Set
from qutebrowser.qt.core import QUrl
from qutebrowser.browser import webelem
from qutebrowser.config import config
from qutebrowser.utils import objreg, urlutils, log, message, qtutils
from qutebrowser.mainwindow import mainwindow

class Error(Exception):
    """Raised when the navigation can't be done."""
_URL_SEGMENTS = [('host', lambda url: url.host(QUrl.ComponentFormattingOption.FullyEncoded), lambda url, host: url.setHost(host, QUrl.ParsingMode.StrictMode)), ('port', lambda url: str(url.port()) if url.port() > 0 else '', lambda url, x: url.setPort(int(x))), ('path', lambda url: url.path(QUrl.ComponentFormattingOption.FullyEncoded), lambda url, path: url.setPath(path, QUrl.ParsingMode.StrictMode)), ('query', lambda url: url.query(QUrl.ComponentFormattingOption.FullyEncoded), lambda url, query: url.setQuery(query, QUrl.ParsingMode.StrictMode)), ('anchor', lambda url: url.fragment(QUrl.ComponentFormattingOption.FullyEncoded), lambda url, fragment: url.setFragment(fragment, QUrl.ParsingMode.StrictMode))]

def _get_incdec_value(match, inc_or_dec, count):
    if False:
        print('Hello World!')
    'Get an incremented/decremented URL based on a URL match.'
    (pre, zeroes, number, post) = match.groups()
    val = int(number)
    if inc_or_dec == 'decrement':
        if val < count:
            raise Error("Can't decrement {} by {}!".format(val, count))
        val -= count
    elif inc_or_dec == 'increment':
        val += count
    else:
        raise ValueError('Invalid value {} for inc_or_dec!'.format(inc_or_dec))
    if zeroes:
        if len(number) < len(str(val)):
            zeroes = zeroes[1:]
        elif len(number) > len(str(val)):
            zeroes += '0'
    return ''.join([pre, zeroes, str(val), post])

def incdec(url, count, inc_or_dec):
    if False:
        print('Hello World!')
    "Helper method for :navigate when `where' is increment/decrement.\n\n    Args:\n        url: The current url.\n        count: How much to increment or decrement by.\n        inc_or_dec: Either 'increment' or 'decrement'.\n    "
    urlutils.ensure_valid(url)
    segments: Optional[Set[str]] = set(config.val.url.incdec_segments)
    if segments is None:
        segments = {'path', 'query'}
    url = QUrl(url)
    for (segment, getter, setter) in reversed(_URL_SEGMENTS):
        if segment not in segments:
            continue
        match = re.fullmatch('(.*\\D|^)(?<!%)(?<!%.)(0*)(\\d+)(.*)', getter(url))
        if not match:
            continue
        setter(url, _get_incdec_value(match, inc_or_dec, count))
        qtutils.ensure_valid(url)
        return url
    raise Error('No number found in URL!')

def path_up(url, count):
    if False:
        i = 10
        return i + 15
    "Helper method for :navigate when `where' is up.\n\n    Args:\n        url: The current url.\n        count: The number of levels to go up in the url.\n    "
    urlutils.ensure_valid(url)
    url = url.adjusted(QUrl.UrlFormattingOption.RemoveFragment | QUrl.UrlFormattingOption.RemoveQuery)
    path = url.path(QUrl.ComponentFormattingOption.FullyEncoded)
    if not path or path == '/':
        raise Error("Can't go up!")
    for _i in range(0, min(count, path.count('/'))):
        path = posixpath.join(path, posixpath.pardir)
    path = posixpath.normpath(path)
    url.setPath(path, QUrl.ParsingMode.StrictMode)
    return url

def strip(url, count):
    if False:
        i = 10
        return i + 15
    'Strip fragment/query from a URL.'
    if count != 1:
        raise Error('Count is not supported when stripping URL components')
    urlutils.ensure_valid(url)
    return url.adjusted(QUrl.UrlFormattingOption.RemoveFragment | QUrl.UrlFormattingOption.RemoveQuery)

def _find_prevnext(prev, elems):
    if False:
        i = 10
        return i + 15
    'Find a prev/next element in the given list of elements.'
    rel_values = {'prev', 'previous'} if prev else {'next'}
    classes = {'nav-prev'} if prev else {'nav-next'}
    for e in elems:
        if e.tag_name() not in ['link', 'a']:
            continue
        if 'rel' in e and set(e['rel'].split(' ')) & rel_values:
            log.hints.debug('Found {!r} with rel={}'.format(e, e['rel']))
            return e
        elif e.classes() & classes:
            log.hints.debug('Found {!r} with class={}'.format(e, e.classes()))
            return e
    elems = [e for e in elems if e.tag_name() != 'link']
    option = 'prev_regexes' if prev else 'next_regexes'
    if not elems:
        return None
    for regex in getattr(config.val.hints, option):
        log.hints.vdebug("== Checking regex '{}'.".format(regex.pattern))
        for e in elems:
            text = str(e)
            if not text:
                continue
            if regex.search(text):
                log.hints.debug("Regex '{}' matched on '{}'.".format(regex.pattern, text))
                return e
            else:
                log.hints.vdebug("No match on '{}'!".format(text))
    return None

def prevnext(*, browsertab, win_id, baseurl, prev=False, tab=False, background=False, window=False):
    if False:
        while True:
            i = 10
    'Click a "previous"/"next" element on the page.\n\n    Args:\n        browsertab: The WebKitTab/WebEngineTab of the page.\n        baseurl: The base URL of the current tab.\n        prev: True to open a "previous" link, False to open a "next" link.\n        tab: True to open in a new tab, False for the current tab.\n        background: True to open in a background tab.\n        window: True to open in a new window, False for the current one.\n    '

    def _prevnext_cb(elems):
        if False:
            i = 10
            return i + 15
        elem = _find_prevnext(prev, elems)
        word = 'prev' if prev else 'forward'
        if elem is None:
            message.error('No {} links found!'.format(word))
            return
        url = elem.resolve_url(baseurl)
        if url is None:
            message.error('No {} links found!'.format(word))
            return
        qtutils.ensure_valid(url)
        cur_tabbed_browser = objreg.get('tabbed-browser', scope='window', window=win_id)
        if window:
            new_window = mainwindow.MainWindow(private=cur_tabbed_browser.is_private)
            tabbed_browser = objreg.get('tabbed-browser', scope='window', window=new_window.win_id)
            tabbed_browser.tabopen(url, background=False)
            new_window.show()
        elif tab:
            cur_tabbed_browser.tabopen(url, background=background)
        else:
            browsertab.load_url(url)
    try:
        link_selector = webelem.css_selector('links', baseurl)
    except webelem.Error as e:
        raise Error(str(e))
    browsertab.elements.find_css(link_selector, callback=_prevnext_cb, error_cb=lambda err: message.error(str(err)))