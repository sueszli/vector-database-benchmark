"""Bridge from QWeb(Engine)Settings to our own settings."""
import re
import argparse
import functools
import dataclasses
from typing import Any, Callable, Dict, Optional, Union
from qutebrowser.qt.core import QUrl, pyqtSlot, qVersion
from qutebrowser.qt.gui import QFont
import qutebrowser
from qutebrowser.config import config
from qutebrowser.utils import usertypes, urlmatch, qtutils, utils
from qutebrowser.misc import objects, debugcachestats
UNSET = object()

@dataclasses.dataclass
class UserAgent:
    """A parsed user agent."""
    os_info: str
    webkit_version: str
    upstream_browser_key: str
    upstream_browser_version: str
    qt_key: str
    qt_version: Optional[str]

    @classmethod
    def parse(cls, ua: str) -> 'UserAgent':
        if False:
            i = 10
            return i + 15
        'Parse a user agent string into its components.'
        comment_matches = re.finditer('\\(([^)]*)\\)', ua)
        os_info = list(comment_matches)[0].group(1)
        version_matches = re.finditer('(\\S+)/(\\S+)', ua)
        versions = {}
        for match in version_matches:
            versions[match.group(1)] = match.group(2)
        webkit_version = versions['AppleWebKit']
        if 'Chrome' in versions:
            upstream_browser_key = 'Chrome'
            qt_key = 'QtWebEngine'
        elif 'Version' in versions:
            upstream_browser_key = 'Version'
            qt_key = 'Qt'
        else:
            raise ValueError('Invalid upstream browser key: {}'.format(ua))
        upstream_browser_version = versions[upstream_browser_key]
        qt_version = versions.get(qt_key)
        return cls(os_info=os_info, webkit_version=webkit_version, upstream_browser_key=upstream_browser_key, upstream_browser_version=upstream_browser_version, qt_key=qt_key, qt_version=qt_version)

class AttributeInfo:
    """Info about a settings attribute."""

    def __init__(self, *attributes: Any, converter: Callable[[Any], bool]=None) -> None:
        if False:
            return 10
        self.attributes = attributes
        if converter is None:
            self.converter = lambda val: val
        else:
            self.converter = converter

class AbstractSettings:
    """Abstract base class for settings set via QWeb(Engine)Settings."""
    _ATTRIBUTES: Dict[str, AttributeInfo] = {}
    _FONT_SIZES: Dict[str, Any] = {}
    _FONT_FAMILIES: Dict[str, Any] = {}
    _FONT_TO_QFONT: Dict[Any, QFont.StyleHint] = {}

    def __init__(self, settings: Any) -> None:
        if False:
            return 10
        self._settings = settings

    def set_attribute(self, name: str, value: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Set the given QWebSettings/QWebEngineSettings attribute.\n\n        If the value is usertypes.UNSET, the value is reset instead.\n        '
        info = self._ATTRIBUTES[name]
        for attribute in info.attributes:
            if value is usertypes.UNSET:
                self._settings.resetAttribute(attribute)
            else:
                self._settings.setAttribute(attribute, info.converter(value))

    def test_attribute(self, name: str) -> bool:
        if False:
            return 10
        'Get the value for the given attribute.\n\n        If the setting resolves to a list of attributes, only the first\n        attribute is tested.\n        '
        info = self._ATTRIBUTES[name]
        return self._settings.testAttribute(info.attributes[0])

    def set_font_size(self, name: str, value: Union[int, usertypes.Unset]) -> None:
        if False:
            i = 10
            return i + 15
        'Set the given QWebSettings/QWebEngineSettings font size.'
        family = self._FONT_SIZES[name]
        if value is usertypes.UNSET:
            self._settings.resetFontSize(family)
        else:
            self._settings.setFontSize(family, value)

    def set_font_family(self, name: str, value: Union[str, None, usertypes.Unset]) -> None:
        if False:
            print('Hello World!')
        'Set the given QWebSettings/QWebEngineSettings font family.\n\n        With None (the default), QFont is used to get the default font for the\n        family.\n        '
        family = self._FONT_FAMILIES[name]
        if value is usertypes.UNSET:
            self._settings.resetFontFamily(family)
        elif value is None:
            font = QFont()
            font.setStyleHint(self._FONT_TO_QFONT[family])
            value = font.defaultFamily()
            self._settings.setFontFamily(family, value)
        else:
            self._settings.setFontFamily(family, value)

    def set_default_text_encoding(self, encoding: Union[str, usertypes.Unset]) -> None:
        if False:
            while True:
                i = 10
        'Set the default text encoding to use.'
        assert encoding is not usertypes.UNSET
        self._settings.setDefaultTextEncoding(encoding)

    def _update_setting(self, setting: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the given setting/value.\n\n        Unknown settings are ignored.\n        '
        if setting in self._ATTRIBUTES:
            self.set_attribute(setting, value)
        elif setting in self._FONT_SIZES:
            self.set_font_size(setting, value)
        elif setting in self._FONT_FAMILIES:
            self.set_font_family(setting, value)
        elif setting == 'content.default_encoding':
            self.set_default_text_encoding(value)

    def update_setting(self, setting: str) -> None:
        if False:
            print('Hello World!')
        'Update the given setting.'
        value = config.instance.get(setting)
        self._update_setting(setting, value)

    def update_for_url(self, url: QUrl) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update settings customized for the given tab.'
        qtutils.ensure_valid(url)
        for values in config.instance:
            if not values.opt.supports_pattern:
                continue
            value = values.get_for_url(url, fallback=False)
            self._update_setting(values.opt.name, value)

    def init_settings(self) -> None:
        if False:
            return 10
        'Set all supported settings correctly.'
        for setting in list(self._ATTRIBUTES) + list(self._FONT_SIZES) + list(self._FONT_FAMILIES):
            self.update_setting(setting)
        self.update_setting('content.default_encoding')

@debugcachestats.register(name='user agent cache')
@functools.lru_cache
def _format_user_agent(template: str, backend: usertypes.Backend) -> str:
    if False:
        i = 10
        return i + 15
    if backend == usertypes.Backend.QtWebEngine:
        from qutebrowser.browser.webengine import webenginesettings
        parsed = webenginesettings.parsed_user_agent
    else:
        from qutebrowser.browser.webkit import webkitsettings
        parsed = webkitsettings.parsed_user_agent
    assert parsed is not None
    return template.format(os_info=parsed.os_info, webkit_version=parsed.webkit_version, qt_key=parsed.qt_key, qt_version=qVersion(), upstream_browser_key=parsed.upstream_browser_key, upstream_browser_version=parsed.upstream_browser_version, qutebrowser_version=qutebrowser.__version__)

def user_agent(url: QUrl=None) -> str:
    if False:
        i = 10
        return i + 15
    'Get the user agent for the given URL, or the global one if URL is None.\n\n    Note that the given URL should always be valid.\n    '
    template = config.instance.get('content.headers.user_agent', url=url)
    return _format_user_agent(template=template, backend=objects.backend)

def init(args: argparse.Namespace) -> None:
    if False:
        i = 10
        return i + 15
    'Initialize all QWeb(Engine)Settings.'
    utils.unused(args)
    if objects.backend == usertypes.Backend.QtWebEngine:
        from qutebrowser.browser.webengine import webenginesettings
        webenginesettings.init()
    elif objects.backend == usertypes.Backend.QtWebKit:
        from qutebrowser.browser.webkit import webkitsettings
        webkitsettings.init()
    else:
        raise utils.Unreachable(objects.backend)
    for pattern in ['chrome://*/*', 'qute://*/*']:
        config.instance.set_obj('content.javascript.enabled', True, pattern=urlmatch.UrlPattern(pattern), hide_userconfig=True)

def clear_private_data() -> None:
    if False:
        return 10
    'Clear cookies, cache and related data for private browsing sessions.'
    if objects.backend == usertypes.Backend.QtWebEngine:
        from qutebrowser.browser.webengine import webenginesettings
        webenginesettings.init_private_profile()
    elif objects.backend == usertypes.Backend.QtWebKit:
        from qutebrowser.browser.webkit import cookies
        assert cookies.ram_cookie_jar is not None
        cookies.ram_cookie_jar.setAllCookies([])
    else:
        raise utils.Unreachable(objects.backend)

@pyqtSlot()
def shutdown() -> None:
    if False:
        return 10
    'Shut down QWeb(Engine)Settings.'
    if objects.backend == usertypes.Backend.QtWebEngine:
        from qutebrowser.browser.webengine import webenginesettings
        webenginesettings.shutdown()
    elif objects.backend == usertypes.Backend.QtWebKit:
        from qutebrowser.browser.webkit import webkitsettings
        webkitsettings.shutdown()
    else:
        raise utils.Unreachable(objects.backend)