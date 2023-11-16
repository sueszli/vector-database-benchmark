"""Bridge from QWebEngineSettings to our own settings.

Module attributes:
    ATTRIBUTES: A mapping from internal setting names to QWebEngineSetting enum
                constants.
"""
import os
import operator
import pathlib
from typing import cast, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from qutebrowser.qt import machinery
from qutebrowser.qt.gui import QFont
from qutebrowser.qt.widgets import QApplication
from qutebrowser.qt.webenginecore import QWebEngineSettings, QWebEngineProfile
from qutebrowser.browser import history
from qutebrowser.browser.webengine import spell, webenginequtescheme, cookies, webenginedownloads, notification
from qutebrowser.config import config, websettings
from qutebrowser.config.websettings import AttributeInfo as Attr
from qutebrowser.utils import standarddir, qtutils, message, log, urlmatch, usertypes, objreg, version
if TYPE_CHECKING:
    from qutebrowser.browser.webengine import interceptor
default_profile = cast(QWebEngineProfile, None)
private_profile: Optional[QWebEngineProfile] = None
_global_settings = cast('WebEngineSettings', None)
parsed_user_agent = None
_qute_scheme_handler = cast(webenginequtescheme.QuteSchemeHandler, None)
_req_interceptor = cast('interceptor.RequestInterceptor', None)
_download_manager = cast(webenginedownloads.DownloadManager, None)

class _SettingsWrapper:
    """Expose a QWebEngineSettings interface which acts on all profiles.

    For read operations, the default profile value is always used.
    """

    def default_profile(self):
        if False:
            for i in range(10):
                print('nop')
        assert default_profile is not None
        return default_profile

    def _settings(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.default_profile().settings()
        if private_profile:
            yield private_profile.settings()

    def setAttribute(self, attribute, on):
        if False:
            while True:
                i = 10
        for settings in self._settings():
            settings.setAttribute(attribute, on)

    def setFontFamily(self, which, family):
        if False:
            return 10
        for settings in self._settings():
            settings.setFontFamily(which, family)

    def setFontSize(self, fonttype, size):
        if False:
            i = 10
            return i + 15
        for settings in self._settings():
            settings.setFontSize(fonttype, size)

    def setDefaultTextEncoding(self, encoding):
        if False:
            return 10
        for settings in self._settings():
            settings.setDefaultTextEncoding(encoding)

    def setUnknownUrlSchemePolicy(self, policy):
        if False:
            i = 10
            return i + 15
        for settings in self._settings():
            settings.setUnknownUrlSchemePolicy(policy)

    def testAttribute(self, attribute):
        if False:
            for i in range(10):
                print('nop')
        return self.default_profile().settings().testAttribute(attribute)

    def fontSize(self, fonttype):
        if False:
            for i in range(10):
                print('nop')
        return self.default_profile().settings().fontSize(fonttype)

    def fontFamily(self, which):
        if False:
            while True:
                i = 10
        return self.default_profile().settings().fontFamily(which)

    def defaultTextEncoding(self):
        if False:
            return 10
        return self.default_profile().settings().defaultTextEncoding()

    def unknownUrlSchemePolicy(self):
        if False:
            print('Hello World!')
        return self.default_profile().settings().unknownUrlSchemePolicy()

class WebEngineSettings(websettings.AbstractSettings):
    """A wrapper for the config for QWebEngineSettings."""
    _ATTRIBUTES = {'content.xss_auditing': Attr(QWebEngineSettings.WebAttribute.XSSAuditingEnabled), 'content.images': Attr(QWebEngineSettings.WebAttribute.AutoLoadImages), 'content.javascript.enabled': Attr(QWebEngineSettings.WebAttribute.JavascriptEnabled), 'content.javascript.can_open_tabs_automatically': Attr(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows), 'content.plugins': Attr(QWebEngineSettings.WebAttribute.PluginsEnabled), 'content.hyperlink_auditing': Attr(QWebEngineSettings.WebAttribute.HyperlinkAuditingEnabled), 'content.local_content_can_access_remote_urls': Attr(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls), 'content.local_content_can_access_file_urls': Attr(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls), 'content.webgl': Attr(QWebEngineSettings.WebAttribute.WebGLEnabled), 'content.local_storage': Attr(QWebEngineSettings.WebAttribute.LocalStorageEnabled), 'content.desktop_capture': Attr(QWebEngineSettings.WebAttribute.ScreenCaptureEnabled, converter=lambda val: True if val == 'ask' else val), 'input.spatial_navigation': Attr(QWebEngineSettings.WebAttribute.SpatialNavigationEnabled), 'input.links_included_in_focus_chain': Attr(QWebEngineSettings.WebAttribute.LinksIncludedInFocusChain), 'scrolling.smooth': Attr(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled), 'content.print_element_backgrounds': Attr(QWebEngineSettings.WebAttribute.PrintElementBackgrounds), 'content.autoplay': Attr(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, converter=operator.not_), 'content.dns_prefetch': Attr(QWebEngineSettings.WebAttribute.DnsPrefetchEnabled), 'tabs.favicons.show': Attr(QWebEngineSettings.WebAttribute.AutoLoadIconsForPage, converter=lambda val: val != 'never')}
    _FONT_SIZES = {'fonts.web.size.minimum': QWebEngineSettings.FontSize.MinimumFontSize, 'fonts.web.size.minimum_logical': QWebEngineSettings.FontSize.MinimumLogicalFontSize, 'fonts.web.size.default': QWebEngineSettings.FontSize.DefaultFontSize, 'fonts.web.size.default_fixed': QWebEngineSettings.FontSize.DefaultFixedFontSize}
    _FONT_FAMILIES = {'fonts.web.family.standard': QWebEngineSettings.FontFamily.StandardFont, 'fonts.web.family.fixed': QWebEngineSettings.FontFamily.FixedFont, 'fonts.web.family.serif': QWebEngineSettings.FontFamily.SerifFont, 'fonts.web.family.sans_serif': QWebEngineSettings.FontFamily.SansSerifFont, 'fonts.web.family.cursive': QWebEngineSettings.FontFamily.CursiveFont, 'fonts.web.family.fantasy': QWebEngineSettings.FontFamily.FantasyFont}
    _UNKNOWN_URL_SCHEME_POLICY = {'disallow': QWebEngineSettings.UnknownUrlSchemePolicy.DisallowUnknownUrlSchemes, 'allow-from-user-interaction': QWebEngineSettings.UnknownUrlSchemePolicy.AllowUnknownUrlSchemesFromUserInteraction, 'allow-all': QWebEngineSettings.UnknownUrlSchemePolicy.AllowAllUnknownUrlSchemes}
    _FONT_TO_QFONT = {QWebEngineSettings.FontFamily.StandardFont: QFont.StyleHint.Serif, QWebEngineSettings.FontFamily.FixedFont: QFont.StyleHint.Monospace, QWebEngineSettings.FontFamily.SerifFont: QFont.StyleHint.Serif, QWebEngineSettings.FontFamily.SansSerifFont: QFont.StyleHint.SansSerif, QWebEngineSettings.FontFamily.CursiveFont: QFont.StyleHint.Cursive, QWebEngineSettings.FontFamily.FantasyFont: QFont.StyleHint.Fantasy}
    _JS_CLIPBOARD_SETTINGS = {'none': {QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard: False, QWebEngineSettings.WebAttribute.JavascriptCanPaste: False}, 'access': {QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard: True, QWebEngineSettings.WebAttribute.JavascriptCanPaste: False}, 'access-paste': {QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard: True, QWebEngineSettings.WebAttribute.JavascriptCanPaste: True}}

    def set_unknown_url_scheme_policy(self, policy: Union[str, usertypes.Unset]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the UnknownUrlSchemePolicy to use.'
        if isinstance(policy, usertypes.Unset):
            self._settings.resetUnknownUrlSchemePolicy()
        else:
            new_value = self._UNKNOWN_URL_SCHEME_POLICY[policy]
            self._settings.setUnknownUrlSchemePolicy(new_value)

    def _set_js_clipboard(self, value: Union[str, usertypes.Unset]) -> None:
        if False:
            while True:
                i = 10
        attr_access = QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard
        attr_paste = QWebEngineSettings.WebAttribute.JavascriptCanPaste
        if isinstance(value, usertypes.Unset):
            self._settings.resetAttribute(attr_access)
            self._settings.resetAttribute(attr_paste)
        else:
            for (attr, attr_val) in self._JS_CLIPBOARD_SETTINGS[value].items():
                self._settings.setAttribute(attr, attr_val)

    def _update_setting(self, setting, value):
        if False:
            for i in range(10):
                print('nop')
        if setting == 'content.unknown_url_scheme_policy':
            self.set_unknown_url_scheme_policy(value)
        elif setting == 'content.javascript.clipboard':
            self._set_js_clipboard(value)
        super()._update_setting(setting, value)

    def init_settings(self):
        if False:
            return 10
        super().init_settings()
        self.update_setting('content.unknown_url_scheme_policy')
        self.update_setting('content.javascript.clipboard')

class ProfileSetter:
    """Helper to set various settings on a profile."""

    def __init__(self, profile):
        if False:
            print('Hello World!')
        self._profile = profile
        self._name_to_method = {'content.cache.size': self.set_http_cache_size, 'content.cookies.store': self.set_persistent_cookie_policy, 'spellcheck.languages': self.set_dictionary_language, 'content.headers.user_agent': self.set_http_headers, 'content.headers.accept_language': self.set_http_headers}

    def update_setting(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Update a setting based on its name.'
        try:
            meth = self._name_to_method[name]
        except KeyError:
            return
        meth()

    def init_profile(self):
        if False:
            return 10
        'Initialize settings on the given profile.'
        self.set_http_headers()
        self.set_http_cache_size()
        self._set_hardcoded_settings()
        self.set_persistent_cookie_policy()
        self.set_dictionary_language()

    def _set_hardcoded_settings(self):
        if False:
            print('Hello World!')
        'Set up settings with a fixed value.'
        settings = self._profile.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.FocusOnNavigationEnabled, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PdfViewerEnabled, False)

    def set_http_headers(self):
        if False:
            for i in range(10):
                print('nop')
        'Set the user agent and accept-language for the given profile.\n\n        We override those per request in the URL interceptor (to allow for\n        per-domain values), but this one still gets used for things like\n        window.navigator.userAgent/.languages in JS.\n        '
        user_agent = websettings.user_agent()
        self._profile.setHttpUserAgent(user_agent)
        accept_language = config.val.content.headers.accept_language
        if accept_language is not None:
            self._profile.setHttpAcceptLanguage(accept_language)

    def set_http_cache_size(self):
        if False:
            while True:
                i = 10
        'Initialize the HTTP cache size for the given profile.'
        size = config.val.content.cache.size
        if size is None:
            size = 0
        else:
            size = qtutils.check_overflow(size, 'int', fatal=False)
        self._profile.setHttpCacheMaximumSize(size)

    def set_persistent_cookie_policy(self):
        if False:
            while True:
                i = 10
        'Set the HTTP Cookie size for the given profile.'
        if self._profile.isOffTheRecord():
            return
        if config.val.content.cookies.store:
            value = QWebEngineProfile.PersistentCookiesPolicy.AllowPersistentCookies
        else:
            value = QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies
        self._profile.setPersistentCookiesPolicy(value)

    def set_dictionary_language(self):
        if False:
            i = 10
            return i + 15
        'Load the given dictionaries.'
        filenames = []
        for code in config.val.spellcheck.languages or []:
            local_filename = spell.local_filename(code)
            if not local_filename:
                if not self._profile.isOffTheRecord():
                    message.warning("Language {} is not installed - see scripts/dictcli.py in qutebrowser's sources".format(code))
                continue
            filenames.append(os.path.splitext(local_filename)[0])
        log.config.debug('Found dicts: {}'.format(filenames))
        self._profile.setSpellCheckLanguages(filenames)
        self._profile.setSpellCheckEnabled(bool(filenames))

def _update_settings(option):
    if False:
        return 10
    'Update global settings when qwebsettings changed.'
    _global_settings.update_setting(option)
    default_profile.setter.update_setting(option)
    if private_profile:
        private_profile.setter.update_setting(option)

def _init_user_agent_str(ua):
    if False:
        return 10
    global parsed_user_agent
    parsed_user_agent = websettings.UserAgent.parse(ua)

def init_user_agent():
    if False:
        for i in range(10):
            print('nop')
    'Make the default WebEngine user agent available via parsed_user_agent.'
    actual_default_profile = QWebEngineProfile.defaultProfile()
    assert actual_default_profile is not None
    _init_user_agent_str(actual_default_profile.httpUserAgent())

def _init_profile(profile: QWebEngineProfile) -> None:
    if False:
        print('Hello World!')
    'Initialize a new QWebEngineProfile.\n\n    This currently only contains the steps which are shared between a private and a\n    non-private profile (at the moment, only the default profile).\n    '
    profile.setter = ProfileSetter(profile)
    profile.setter.init_profile()
    _qute_scheme_handler.install(profile)
    _req_interceptor.install(profile)
    _download_manager.install(profile)
    cookies.install_filter(profile)
    if notification.bridge is not None:
        notification.bridge.install(profile)
    history.web_history.history_cleared.connect(profile.clearAllVisitedLinks)
    history.web_history.url_cleared.connect(lambda url: profile.clearVisitedLinks([url]))
    _global_settings.init_settings()

def _init_default_profile():
    if False:
        i = 10
        return i + 15
    'Init the default QWebEngineProfile.'
    global default_profile
    if machinery.IS_QT6:
        default_profile = QWebEngineProfile('Default')
    else:
        default_profile = QWebEngineProfile.defaultProfile()
    assert not default_profile.isOffTheRecord()
    assert parsed_user_agent is None
    non_ua_version = version.qtwebengine_versions(avoid_init=True)
    init_user_agent()
    ua_version = version.qtwebengine_versions()
    if ua_version.webengine != non_ua_version.webengine:
        log.init.warning(f'QtWebEngine version mismatch - unexpected behavior might occur, please open a bug about this.\n  Early version: {non_ua_version}\n  Real version:  {ua_version}')
    default_profile.setCachePath(os.path.join(standarddir.cache(), 'webengine'))
    default_profile.setPersistentStoragePath(os.path.join(standarddir.data(), 'webengine'))
    _init_profile(default_profile)

def init_private_profile():
    if False:
        print('Hello World!')
    'Init the private QWebEngineProfile.'
    global private_profile
    if qtutils.is_single_process():
        return
    private_profile = QWebEngineProfile()
    assert private_profile.isOffTheRecord()
    _init_profile(private_profile)

def _init_site_specific_quirks():
    if False:
        return 10
    'Add custom user-agent settings for problematic sites.\n\n    See https://github.com/qutebrowser/qutebrowser/issues/4810\n    '
    if not config.val.content.site_specific_quirks.enabled:
        return
    no_qtwe_ua = 'Mozilla/5.0 ({os_info}) AppleWebKit/{webkit_version} (KHTML, like Gecko) {upstream_browser_key}/{upstream_browser_version} Safari/{webkit_version}'
    firefox_ua = 'Mozilla/5.0 ({os_info}; rv:90.0) Gecko/20100101 Firefox/90.0'

    def maybe_newer_chrome_ua(at_least_version):
        if False:
            return 10
        "Return a new UA if our current chrome version isn't at least at_least_version."
        current_chome_version = version.qtwebengine_versions().chromium_major
        if current_chome_version >= at_least_version:
            return None
        return f'Mozilla/5.0 ({{os_info}}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{at_least_version} Safari/537.36'
    user_agents = [('ua-whatsapp', 'https://web.whatsapp.com/', no_qtwe_ua), ('ua-google', 'https://accounts.google.com/*', firefox_ua), ('ua-slack', 'https://*.slack.com/*', maybe_newer_chrome_ua(112))]
    for (name, pattern, ua) in user_agents:
        if not ua:
            continue
        if name not in config.val.content.site_specific_quirks.skip:
            config.instance.set_obj('content.headers.user_agent', ua, pattern=urlmatch.UrlPattern(pattern), hide_userconfig=True)
    if 'misc-krunker' not in config.val.content.site_specific_quirks.skip:
        config.instance.set_obj('content.headers.accept_language', '', pattern=urlmatch.UrlPattern('https://matchmaker.krunker.io/*'), hide_userconfig=True)

def _init_default_settings():
    if False:
        for i in range(10):
            print('nop')
    'Set permissions required for internal functionality.\n\n    - Make sure the devtools always get images/JS permissions.\n    - On Qt 6, make sure files in the data path can load external resources.\n    '
    devtools_settings: List[Tuple[str, Any]] = [('content.javascript.enabled', True), ('content.images', True), ('content.cookies.accept', 'all')]
    for (setting, value) in devtools_settings:
        for pattern in ['chrome-devtools://*', 'devtools://*']:
            config.instance.set_obj(setting, value, pattern=urlmatch.UrlPattern(pattern), hide_userconfig=True)
    if machinery.IS_QT6:
        userscripts_settings: List[Tuple[str, Any]] = [('content.local_content_can_access_remote_urls', True), ('content.local_content_can_access_file_urls', False)]
        url = pathlib.Path(standarddir.data(), 'userscripts').as_uri()
        for (setting, value) in userscripts_settings:
            config.instance.set_obj(setting, value, pattern=urlmatch.UrlPattern(f'{url}/*'), hide_userconfig=True)

def init():
    if False:
        return 10
    'Initialize the global QWebSettings.'
    webenginequtescheme.init()
    spell.init()
    global _qute_scheme_handler
    app = QApplication.instance()
    log.init.debug('Initializing qute://* handler...')
    _qute_scheme_handler = webenginequtescheme.QuteSchemeHandler(parent=app)
    global _req_interceptor
    log.init.debug('Initializing request interceptor...')
    from qutebrowser.browser.webengine import interceptor
    _req_interceptor = interceptor.RequestInterceptor(parent=app)
    global _download_manager
    log.init.debug('Initializing QtWebEngine downloads...')
    _download_manager = webenginedownloads.DownloadManager(parent=app)
    objreg.register('webengine-download-manager', _download_manager)
    from qutebrowser.misc import quitter
    quitter.instance.shutting_down.connect(_download_manager.shutdown)
    log.init.debug('Initializing notification presenter...')
    notification.init()
    log.init.debug('Initializing global settings...')
    global _global_settings
    _global_settings = WebEngineSettings(_SettingsWrapper())
    log.init.debug('Initializing profiles...')
    _init_default_profile()
    init_private_profile()
    config.instance.changed.connect(_update_settings)
    log.init.debug('Misc initialization...')
    _init_site_specific_quirks()
    _init_default_settings()

def shutdown():
    if False:
        i = 10
        return i + 15
    pass