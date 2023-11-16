""" Utility functions for helping with operations involving browsers.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import webbrowser
from os.path import abspath
from typing import Literal, Protocol, cast
from ..settings import settings
BrowserTarget = Literal['same', 'window', 'tab']
TargetCode = Literal[0, 1, 2]
NEW_PARAM: dict[BrowserTarget, TargetCode] = {'same': 0, 'window': 1, 'tab': 2}
__all__ = ('DummyWebBrowser', 'get_browser_controller', 'view')

class BrowserLike(Protocol):
    """ Interface for browser-like objects.

    """

    def open(self, url: str, new: TargetCode=..., autoraise: bool=...) -> bool:
        if False:
            print('Hello World!')
        ...

class DummyWebBrowser:
    """ A "no-op" web-browser controller.

    """

    def open(self, url: str, new: TargetCode=0, autoraise: bool=True) -> bool:
        if False:
            i = 10
            return i + 15
        ' Receive standard arguments and take no action. '
        return True

def get_browser_controller(browser: str | None=None) -> BrowserLike:
    if False:
        for i in range(10):
            print('nop')
    " Return a browser controller.\n\n    Args:\n        browser (str or None) : browser name, or ``None`` (default: ``None``)\n            If passed the string ``'none'``, a dummy web browser controller\n            is returned.\n\n            Otherwise, use the value to select an appropriate controller using\n            the :doc:`webbrowser <python:library/webbrowser>` standard library\n            module. If the value is ``None``, a system default is used.\n\n    Returns:\n        controller : a web browser controller\n\n    "
    browser = settings.browser(browser)
    if browser is None:
        controller = cast(BrowserLike, webbrowser)
    elif browser == 'none':
        controller = DummyWebBrowser()
    else:
        controller = webbrowser.get(browser)
    return controller

def view(location: str, browser: str | None=None, new: BrowserTarget='same', autoraise: bool=True) -> None:
    if False:
        print('Hello World!')
    ' Open a browser to view the specified location.\n\n    Args:\n        location (str) : Location to open\n            If location does not begin with "http:" it is assumed\n            to be a file path on the local filesystem.\n        browser (str or None) : what browser to use (default: None)\n            If ``None``, use the system default browser.\n        new (str) : How to open the location. Valid values are:\n\n            ``\'same\'`` - open in the current tab\n\n            ``\'tab\'`` - open a new tab in the current window\n\n            ``\'window\'`` - open in a new window\n        autoraise (bool) : Whether to automatically raise the location\n            in a new browser window (default: True)\n\n    Returns:\n        None\n\n    '
    try:
        new_id = NEW_PARAM[new]
    except KeyError:
        raise RuntimeError(f"invalid 'new' value passed to view: {new!r}, valid values are: 'same', 'window', or 'tab'")
    if location.startswith('http'):
        url = location
    else:
        url = 'file://' + abspath(location)
    try:
        controller = get_browser_controller(browser)
        controller.open(url, new=new_id, autoraise=autoraise)
    except Exception:
        pass