"""A payload based version of page."""
import warnings
from IPython.core.getipython import get_ipython

def page(strng, start=0, screen_lines=0, pager_cmd=None):
    if False:
        while True:
            i = 10
    "Print a string, piping through a pager.\n\n    This version ignores the screen_lines and pager_cmd arguments and uses\n    IPython's payload system instead.\n\n    Parameters\n    ----------\n    strng : str or mime-dict\n        Text to page, or a mime-type keyed dict of already formatted data.\n    start : int\n        Starting line at which to place the display.\n    "
    start = max(0, start)
    shell = get_ipython()
    if isinstance(strng, dict):
        data = strng
    else:
        data = {'text/plain': strng}
    payload = dict(source='page', data=data, start=start)
    shell.payload_manager.write_payload(payload)

def install_payload_page():
    if False:
        i = 10
        return i + 15
    'DEPRECATED, use show_in_pager hook\n\n    Install this version of page as IPython.core.page.page.\n    '
    warnings.warn("install_payload_page is deprecated.\n    Use `ip.set_hook('show_in_pager, page.as_hook(payloadpage.page))`\n    ")
    from IPython.core import page as corepage
    corepage.page = page