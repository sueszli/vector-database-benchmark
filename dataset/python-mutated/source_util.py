import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast
from blinker import Signal
from streamlit.logger import get_logger
from streamlit.string_util import extract_leading_emoji
from streamlit.util import calc_md5
LOGGER = get_logger(__name__)

def open_python_file(filename):
    if False:
        for i in range(10):
            print('nop')
    "Open a read-only Python file taking proper care of its encoding.\n\n    In Python 3, we would like all files to be opened with utf-8 encoding.\n    However, some author like to specify PEP263 headers in their source files\n    with their own encodings. In that case, we should respect the author's\n    encoding.\n    "
    import tokenize
    if hasattr(tokenize, 'open'):
        return tokenize.open(filename)
    else:
        return open(filename, 'r', encoding='utf-8')
PAGE_FILENAME_REGEX = re.compile('([0-9]*)[_ -]*(.*)\\.py')

def page_sort_key(script_path: Path) -> Tuple[float, str]:
    if False:
        while True:
            i = 10
    matches = re.findall(PAGE_FILENAME_REGEX, script_path.name)
    assert len(matches) > 0, f'{script_path} is not a Python file'
    [(number, label)] = matches
    label = label.lower()
    if number == '':
        return (float('inf'), label)
    return (float(number), label)

def page_icon_and_name(script_path: Path) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    'Compute the icon and name of a page from its script path.\n\n    This is *almost* the page name displayed in the nav UI, but it has\n    underscores instead of spaces. The reason we do this is because having\n    spaces in URLs both looks bad and is hard to deal with due to the need to\n    URL-encode them. To solve this, we only swap the underscores for spaces\n    right before we render page names.\n    '
    extraction = re.search(PAGE_FILENAME_REGEX, script_path.name)
    if extraction is None:
        return ('', '')
    extraction: re.Match[str] = cast(Any, extraction)
    icon_and_name = re.sub('[_ ]+', '_', extraction.group(2)).strip() or extraction.group(1)
    return extract_leading_emoji(icon_and_name)
_pages_cache_lock = threading.RLock()
_cached_pages: Optional[Dict[str, Dict[str, str]]] = None
_on_pages_changed = Signal(doc='Emitted when the pages directory is changed')

def invalidate_pages_cache():
    if False:
        print('Hello World!')
    global _cached_pages
    LOGGER.debug('Pages directory changed')
    with _pages_cache_lock:
        _cached_pages = None
    _on_pages_changed.send()

def get_pages(main_script_path_str: str) -> Dict[str, Dict[str, str]]:
    if False:
        print('Hello World!')
    global _cached_pages
    pages = _cached_pages
    if pages is not None:
        return pages
    with _pages_cache_lock:
        if _cached_pages is not None:
            return _cached_pages
        main_script_path = Path(main_script_path_str)
        (main_page_icon, main_page_name) = page_icon_and_name(main_script_path)
        main_page_script_hash = calc_md5(main_script_path_str)
        pages = {main_page_script_hash: {'page_script_hash': main_page_script_hash, 'page_name': main_page_name, 'icon': main_page_icon, 'script_path': str(main_script_path.resolve())}}
        pages_dir = main_script_path.parent / 'pages'
        page_scripts = sorted([f for f in pages_dir.glob('*.py') if not f.name.startswith('.') and (not f.name == '__init__.py')], key=page_sort_key)
        for script_path in page_scripts:
            script_path_str = str(script_path.resolve())
            (pi, pn) = page_icon_and_name(script_path)
            psh = calc_md5(script_path_str)
            pages[psh] = {'page_script_hash': psh, 'page_name': pn, 'icon': pi, 'script_path': script_path_str}
        _cached_pages = pages
        return pages

def register_pages_changed_callback(callback: Callable[[str], None]):
    if False:
        print('Hello World!')

    def disconnect():
        if False:
            i = 10
            return i + 15
        _on_pages_changed.disconnect(callback)
    _on_pages_changed.connect(callback, weak=False)
    return disconnect