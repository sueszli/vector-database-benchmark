"""Translation helper functions."""
import functools
import gettext as gettext_module
import os
import re
import sys
import warnings
from asgiref.local import Local
from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, mark_safe
from . import to_language, to_locale
_translations = {}
_active = Local()
_default = None
CONTEXT_SEPARATOR = '\x04'
ACCEPT_LANGUAGE_HEADER_MAX_LENGTH = 500
accept_language_re = _lazy_re_compile('\n        # "en", "en-au", "x-y-z", "es-419", "*"\n        ([A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*|\\*)\n        # Optional "q=1.00", "q=0.8"\n        (?:\\s*;\\s*q=(0(?:\\.[0-9]{,3})?|1(?:\\.0{,3})?))?\n        # Multiple accepts per header.\n        (?:\\s*,\\s*|$)\n    ', re.VERBOSE)
language_code_re = _lazy_re_compile('^[a-z]{1,8}(?:-[a-z0-9]{1,8})*(?:@[a-z0-9]{1,20})?$', re.IGNORECASE)
language_code_prefix_re = _lazy_re_compile('^/(\\w+([@-]\\w+){0,2})(/|$)')

@receiver(setting_changed)
def reset_cache(*, setting, **kwargs):
    if False:
        return 10
    '\n    Reset global state when LANGUAGES setting has been changed, as some\n    languages should no longer be accepted.\n    '
    if setting in ('LANGUAGES', 'LANGUAGE_CODE'):
        check_for_language.cache_clear()
        get_languages.cache_clear()
        get_supported_language_variant.cache_clear()

class TranslationCatalog:
    """
    Simulate a dict for DjangoTranslation._catalog so as multiple catalogs
    with different plural equations are kept separate.
    """

    def __init__(self, trans=None):
        if False:
            print('Hello World!')
        self._catalogs = [trans._catalog.copy()] if trans else [{}]
        self._plurals = [trans.plural] if trans else [lambda n: int(n != 1)]

    def __getitem__(self, key):
        if False:
            return 10
        for cat in self._catalogs:
            try:
                return cat[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        self._catalogs[0][key] = value

    def __contains__(self, key):
        if False:
            print('Hello World!')
        return any((key in cat for cat in self._catalogs))

    def items(self):
        if False:
            return 10
        for cat in self._catalogs:
            yield from cat.items()

    def keys(self):
        if False:
            while True:
                i = 10
        for cat in self._catalogs:
            yield from cat.keys()

    def update(self, trans):
        if False:
            print('Hello World!')
        for (cat, plural) in zip(self._catalogs, self._plurals):
            if trans.plural.__code__ == plural.__code__:
                cat.update(trans._catalog)
                break
        else:
            self._catalogs.insert(0, trans._catalog.copy())
            self._plurals.insert(0, trans.plural)

    def get(self, key, default=None):
        if False:
            return 10
        missing = object()
        for cat in self._catalogs:
            result = cat.get(key, missing)
            if result is not missing:
                return result
        return default

    def plural(self, msgid, num):
        if False:
            i = 10
            return i + 15
        for (cat, plural) in zip(self._catalogs, self._plurals):
            tmsg = cat.get((msgid, plural(num)))
            if tmsg is not None:
                return tmsg
        raise KeyError

class DjangoTranslation(gettext_module.GNUTranslations):
    """
    Set up the GNUTranslations context with regard to output charset.

    This translation object will be constructed out of multiple GNUTranslations
    objects by merging their catalogs. It will construct an object for the
    requested language and add a fallback to the default language, if it's
    different from the requested language.
    """
    domain = 'django'

    def __init__(self, language, domain=None, localedirs=None):
        if False:
            i = 10
            return i + 15
        'Create a GNUTranslations() using many locale directories'
        gettext_module.GNUTranslations.__init__(self)
        if domain is not None:
            self.domain = domain
        self.__language = language
        self.__to_language = to_language(language)
        self.__locale = to_locale(language)
        self._catalog = None
        self.plural = lambda n: int(n != 1)
        if self.domain == 'django':
            if localedirs is not None:
                warnings.warn("localedirs is ignored when domain is 'django'.", RuntimeWarning)
                localedirs = None
            self._init_translation_catalog()
        if localedirs:
            for localedir in localedirs:
                translation = self._new_gnu_trans(localedir)
                self.merge(translation)
        else:
            self._add_installed_apps_translations()
        self._add_local_translations()
        if self.__language == settings.LANGUAGE_CODE and self.domain == 'django' and (self._catalog is None):
            raise OSError('No translation files found for default language %s.' % settings.LANGUAGE_CODE)
        self._add_fallback(localedirs)
        if self._catalog is None:
            self._catalog = TranslationCatalog()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<DjangoTranslation lang:%s>' % self.__language

    def _new_gnu_trans(self, localedir, use_null_fallback=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a mergeable gettext.GNUTranslations instance.\n\n        A convenience wrapper. By default gettext uses 'fallback=False'.\n        Using param `use_null_fallback` to avoid confusion with any other\n        references to 'fallback'.\n        "
        return gettext_module.translation(domain=self.domain, localedir=localedir, languages=[self.__locale], fallback=use_null_fallback)

    def _init_translation_catalog(self):
        if False:
            return 10
        'Create a base catalog using global django translations.'
        settingsfile = sys.modules[settings.__module__].__file__
        localedir = os.path.join(os.path.dirname(settingsfile), 'locale')
        translation = self._new_gnu_trans(localedir)
        self.merge(translation)

    def _add_installed_apps_translations(self):
        if False:
            print('Hello World!')
        'Merge translations from each installed app.'
        try:
            app_configs = reversed(apps.get_app_configs())
        except AppRegistryNotReady:
            raise AppRegistryNotReady("The translation infrastructure cannot be initialized before the apps registry is ready. Check that you don't make non-lazy gettext calls at import time.")
        for app_config in app_configs:
            localedir = os.path.join(app_config.path, 'locale')
            if os.path.exists(localedir):
                translation = self._new_gnu_trans(localedir)
                self.merge(translation)

    def _add_local_translations(self):
        if False:
            i = 10
            return i + 15
        'Merge translations defined in LOCALE_PATHS.'
        for localedir in reversed(settings.LOCALE_PATHS):
            translation = self._new_gnu_trans(localedir)
            self.merge(translation)

    def _add_fallback(self, localedirs=None):
        if False:
            while True:
                i = 10
        'Set the GNUTranslations() fallback with the default language.'
        if self.__language == settings.LANGUAGE_CODE or self.__language.startswith('en'):
            return
        if self.domain == 'django':
            default_translation = translation(settings.LANGUAGE_CODE)
        else:
            default_translation = DjangoTranslation(settings.LANGUAGE_CODE, domain=self.domain, localedirs=localedirs)
        self.add_fallback(default_translation)

    def merge(self, other):
        if False:
            print('Hello World!')
        'Merge another translation into this catalog.'
        if not getattr(other, '_catalog', None):
            return
        if self._catalog is None:
            self.plural = other.plural
            self._info = other._info.copy()
            self._catalog = TranslationCatalog(other)
        else:
            self._catalog.update(other)
        if other._fallback:
            self.add_fallback(other._fallback)

    def language(self):
        if False:
            print('Hello World!')
        'Return the translation language.'
        return self.__language

    def to_language(self):
        if False:
            i = 10
            return i + 15
        'Return the translation language name.'
        return self.__to_language

    def ngettext(self, msgid1, msgid2, n):
        if False:
            print('Hello World!')
        try:
            tmsg = self._catalog.plural(msgid1, n)
        except KeyError:
            if self._fallback:
                return self._fallback.ngettext(msgid1, msgid2, n)
            if n == 1:
                tmsg = msgid1
            else:
                tmsg = msgid2
        return tmsg

def translation(language):
    if False:
        return 10
    "\n    Return a translation object in the default 'django' domain.\n    "
    global _translations
    if language not in _translations:
        _translations[language] = DjangoTranslation(language)
    return _translations[language]

def activate(language):
    if False:
        i = 10
        return i + 15
    '\n    Fetch the translation object for a given language and install it as the\n    current translation object for the current thread.\n    '
    if not language:
        return
    _active.value = translation(language)

def deactivate():
    if False:
        i = 10
        return i + 15
    '\n    Uninstall the active translation object so that further _() calls resolve\n    to the default translation object.\n    '
    if hasattr(_active, 'value'):
        del _active.value

def deactivate_all():
    if False:
        print('Hello World!')
    '\n    Make the active translation object a NullTranslations() instance. This is\n    useful when we want delayed translations to appear as the original string\n    for some reason.\n    '
    _active.value = gettext_module.NullTranslations()
    _active.value.to_language = lambda *args: None

def get_language():
    if False:
        return 10
    'Return the currently selected language.'
    t = getattr(_active, 'value', None)
    if t is not None:
        try:
            return t.to_language()
        except AttributeError:
            pass
    return settings.LANGUAGE_CODE

def get_language_bidi():
    if False:
        i = 10
        return i + 15
    "\n    Return selected language's BiDi layout.\n\n    * False = left-to-right layout\n    * True = right-to-left layout\n    "
    lang = get_language()
    if lang is None:
        return False
    else:
        base_lang = get_language().split('-')[0]
        return base_lang in settings.LANGUAGES_BIDI

def catalog():
    if False:
        print('Hello World!')
    '\n    Return the current active catalog for further processing.\n    This can be used if you need to modify the catalog or want to access the\n    whole message catalog instead of just translating one string.\n    '
    global _default
    t = getattr(_active, 'value', None)
    if t is not None:
        return t
    if _default is None:
        _default = translation(settings.LANGUAGE_CODE)
    return _default

def gettext(message):
    if False:
        for i in range(10):
            print('nop')
    "\n    Translate the 'message' string. It uses the current thread to find the\n    translation object to use. If no current translation is activated, the\n    message will be run through the default translation object.\n    "
    global _default
    eol_message = message.replace('\r\n', '\n').replace('\r', '\n')
    if eol_message:
        _default = _default or translation(settings.LANGUAGE_CODE)
        translation_object = getattr(_active, 'value', _default)
        result = translation_object.gettext(eol_message)
    else:
        result = type(message)('')
    if isinstance(message, SafeData):
        return mark_safe(result)
    return result

def pgettext(context, message):
    if False:
        while True:
            i = 10
    msg_with_ctxt = '%s%s%s' % (context, CONTEXT_SEPARATOR, message)
    result = gettext(msg_with_ctxt)
    if CONTEXT_SEPARATOR in result:
        result = message
    elif isinstance(message, SafeData):
        result = mark_safe(result)
    return result

def gettext_noop(message):
    if False:
        while True:
            i = 10
    "\n    Mark strings for translation but don't translate them now. This can be\n    used to store strings in global variables that should stay in the base\n    language (because they might be used externally) and will be translated\n    later.\n    "
    return message

def do_ntranslate(singular, plural, number, translation_function):
    if False:
        for i in range(10):
            print('nop')
    global _default
    t = getattr(_active, 'value', None)
    if t is not None:
        return getattr(t, translation_function)(singular, plural, number)
    if _default is None:
        _default = translation(settings.LANGUAGE_CODE)
    return getattr(_default, translation_function)(singular, plural, number)

def ngettext(singular, plural, number):
    if False:
        i = 10
        return i + 15
    '\n    Return a string of the translation of either the singular or plural,\n    based on the number.\n    '
    return do_ntranslate(singular, plural, number, 'ngettext')

def npgettext(context, singular, plural, number):
    if False:
        print('Hello World!')
    msgs_with_ctxt = ('%s%s%s' % (context, CONTEXT_SEPARATOR, singular), '%s%s%s' % (context, CONTEXT_SEPARATOR, plural), number)
    result = ngettext(*msgs_with_ctxt)
    if CONTEXT_SEPARATOR in result:
        result = ngettext(singular, plural, number)
    return result

def all_locale_paths():
    if False:
        while True:
            i = 10
    '\n    Return a list of paths to user-provides languages files.\n    '
    globalpath = os.path.join(os.path.dirname(sys.modules[settings.__module__].__file__), 'locale')
    app_paths = []
    for app_config in apps.get_app_configs():
        locale_path = os.path.join(app_config.path, 'locale')
        if os.path.exists(locale_path):
            app_paths.append(locale_path)
    return [globalpath, *settings.LOCALE_PATHS, *app_paths]

@functools.lru_cache(maxsize=1000)
def check_for_language(lang_code):
    if False:
        print('Hello World!')
    '\n    Check whether there is a global language file for the given language\n    code. This is used to decide whether a user-provided language is\n    available.\n\n    lru_cache should have a maxsize to prevent from memory exhaustion attacks,\n    as the provided language codes are taken from the HTTP request. See also\n    <https://www.djangoproject.com/weblog/2007/oct/26/security-fix/>.\n    '
    if lang_code is None or not language_code_re.search(lang_code):
        return False
    return any((gettext_module.find('django', path, [to_locale(lang_code)]) is not None for path in all_locale_paths()))

@functools.lru_cache
def get_languages():
    if False:
        i = 10
        return i + 15
    '\n    Cache of settings.LANGUAGES in a dictionary for easy lookups by key.\n    Convert keys to lowercase as they should be treated as case-insensitive.\n    '
    return {key.lower(): value for (key, value) in dict(settings.LANGUAGES).items()}

@functools.lru_cache(maxsize=1000)
def get_supported_language_variant(lang_code, strict=False):
    if False:
        i = 10
        return i + 15
    "\n    Return the language code that's listed in supported languages, possibly\n    selecting a more generic variant. Raise LookupError if nothing is found.\n\n    If `strict` is False (the default), look for a country-specific variant\n    when neither the language code nor its generic variant is found.\n\n    lru_cache should have a maxsize to prevent from memory exhaustion attacks,\n    as the provided language codes are taken from the HTTP request. See also\n    <https://www.djangoproject.com/weblog/2007/oct/26/security-fix/>.\n    "
    if lang_code:
        possible_lang_codes = [lang_code]
        try:
            possible_lang_codes.extend(LANG_INFO[lang_code]['fallback'])
        except KeyError:
            pass
        i = None
        while (i := lang_code.rfind('-', 0, i)) > -1:
            possible_lang_codes.append(lang_code[:i])
        generic_lang_code = possible_lang_codes[-1]
        supported_lang_codes = get_languages()
        for code in possible_lang_codes:
            if code.lower() in supported_lang_codes and check_for_language(code):
                return code
        if not strict:
            for supported_code in supported_lang_codes:
                if supported_code.startswith(generic_lang_code + '-'):
                    return supported_code
    raise LookupError(lang_code)

def get_language_from_path(path, strict=False):
    if False:
        print('Hello World!')
    "\n    Return the language code if there's a valid language code found in `path`.\n\n    If `strict` is False (the default), look for a country-specific variant\n    when neither the language code nor its generic variant is found.\n    "
    regex_match = language_code_prefix_re.match(path)
    if not regex_match:
        return None
    lang_code = regex_match[1]
    try:
        return get_supported_language_variant(lang_code, strict=strict)
    except LookupError:
        return None

def get_language_from_request(request, check_path=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Analyze the request to find what language the user wants the system to\n    show. Only languages listed in settings.LANGUAGES are taken into account.\n    If the user requests a sublanguage where we have a main language, we send\n    out the main language.\n\n    If check_path is True, the URL path prefix will be checked for a language\n    code, otherwise this is skipped for backwards compatibility.\n    '
    if check_path:
        lang_code = get_language_from_path(request.path_info)
        if lang_code is not None:
            return lang_code
    lang_code = request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME)
    if lang_code is not None and lang_code in get_languages() and check_for_language(lang_code):
        return lang_code
    try:
        return get_supported_language_variant(lang_code)
    except LookupError:
        pass
    accept = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
    for (accept_lang, unused) in parse_accept_lang_header(accept):
        if accept_lang == '*':
            break
        if not language_code_re.search(accept_lang):
            continue
        try:
            return get_supported_language_variant(accept_lang)
        except LookupError:
            continue
    try:
        return get_supported_language_variant(settings.LANGUAGE_CODE)
    except LookupError:
        return settings.LANGUAGE_CODE

@functools.lru_cache(maxsize=1000)
def _parse_accept_lang_header(lang_string):
    if False:
        return 10
    "\n    Parse the lang_string, which is the body of an HTTP Accept-Language\n    header, and return a tuple of (lang, q-value), ordered by 'q' values.\n\n    Return an empty tuple if there are any format errors in lang_string.\n    "
    result = []
    pieces = accept_language_re.split(lang_string.lower())
    if pieces[-1]:
        return ()
    for i in range(0, len(pieces) - 1, 3):
        (first, lang, priority) = pieces[i:i + 3]
        if first:
            return ()
        if priority:
            priority = float(priority)
        else:
            priority = 1.0
        result.append((lang, priority))
    result.sort(key=lambda k: k[1], reverse=True)
    return tuple(result)

def parse_accept_lang_header(lang_string):
    if False:
        i = 10
        return i + 15
    '\n    Parse the value of the Accept-Language header up to a maximum length.\n\n    The value of the header is truncated to a maximum length to avoid potential\n    denial of service and memory exhaustion attacks. Excessive memory could be\n    used if the raw value is very large as it would be cached due to the use of\n    functools.lru_cache() to avoid repetitive parsing of common header values.\n    '
    if len(lang_string) <= ACCEPT_LANGUAGE_HEADER_MAX_LENGTH:
        return _parse_accept_lang_header(lang_string)
    if (index := lang_string.rfind(',', 0, ACCEPT_LANGUAGE_HEADER_MAX_LENGTH)) > 0:
        return _parse_accept_lang_header(lang_string[:index])
    return ()