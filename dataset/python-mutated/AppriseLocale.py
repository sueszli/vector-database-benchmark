import ctypes
import locale
import contextlib
import os
import re
from os.path import join
from os.path import dirname
from os.path import abspath
from .logger import logger
GETTEXT_LOADED = False
try:
    import gettext
    GETTEXT_LOADED = True
except ImportError:
    pass

class AppriseLocale:
    """
    A wrapper class to gettext so that we can manipulate multiple lanaguages
    on the fly if required.

    """
    _domain = 'apprise'
    _locale_dir = abspath(join(dirname(__file__), 'i18n'))
    _local_re = re.compile('^((?P<ansii>C)|(?P<lang>([a-z]{2}))([_:](?P<country>[a-z]{2}))?)(\\.(?P<enc>[a-z0-9-]+))?$', re.IGNORECASE)
    _default_encoding = 'utf-8'
    _fn = 'gettext'
    _default_language = 'en'

    def __init__(self, language=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes our object, if a language is specified, then we\n        initialize ourselves to that, otherwise we use whatever we detect\n        from the local operating system. If all else fails, we resort to the\n        defined default_language.\n\n        '
        self._gtobjs = {}
        self.lang = AppriseLocale.detect_language(language)
        self.__fn_map = None
        if GETTEXT_LOADED is False:
            return
        self.add(self.lang)

    def add(self, lang=None, set_default=True):
        if False:
            return 10
        '\n        Add a language to our list\n        '
        lang = lang if lang else self._default_language
        if lang not in self._gtobjs:
            try:
                self._gtobjs[lang] = gettext.translation(self._domain, localedir=self._locale_dir, languages=[lang], fallback=False)
                self.__fn_map = getattr(self._gtobjs[lang], self._fn)
            except FileNotFoundError:
                logger.debug('Could not load translation path: %s', join(self._locale_dir, lang))
                if self.lang not in self._gtobjs:
                    self._gtobjs[self.lang] = gettext
                    self.__fn_map = getattr(self._gtobjs[self.lang], self._fn)
                return False
            logger.trace('Loaded language %s', lang)
        if set_default:
            logger.debug('Language set to %s', lang)
            self.lang = lang
        return True

    @contextlib.contextmanager
    def lang_at(self, lang, mapto=_fn):
        if False:
            i = 10
            return i + 15
        "\n        The syntax works as:\n            with at.lang_at('fr'):\n                # apprise works as though the french language has been\n                # defined. afterwards, the language falls back to whatever\n                # it was.\n        "
        if GETTEXT_LOADED is False:
            yield None
            return
        lang = AppriseLocale.detect_language(lang, detect_fallback=False)
        if lang not in self._gtobjs and (not self.add(lang, set_default=False)):
            yield getattr(self._gtobjs[self.lang], mapto)
        else:
            yield getattr(self._gtobjs[lang], mapto)
        return

    @property
    def gettext(self):
        if False:
            return 10
        '\n        Return the current language gettext() function\n\n        Useful for assigning to `_`\n        '
        return self._gtobjs[self.lang].gettext

    @staticmethod
    def detect_language(lang=None, detect_fallback=True):
        if False:
            print('Hello World!')
        "\n        Returns the language (if it's retrievable)\n        "
        if not isinstance(lang, str):
            if detect_fallback is False:
                return None
            lookup = os.environ.get
            localename = None
            for variable in ('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'):
                localename = lookup(variable, None)
                if localename:
                    result = AppriseLocale._local_re.match(localename)
                    if result and result.group('lang'):
                        return result.group('lang').lower()
            if hasattr(ctypes, 'windll'):
                windll = ctypes.windll.kernel32
                try:
                    lang = locale.windows_locale[windll.GetUserDefaultUILanguage()]
                    return lang[0:2].lower()
                except (TypeError, KeyError):
                    pass
            try:
                lang = locale.getlocale()[0]
            except (ValueError, TypeError) as e:
                logger.warning('Language detection failure / {}'.format(str(e)))
                return None
        return None if not lang else lang[0:2].lower()

    def __getstate__(self):
        if False:
            print('Hello World!')
        '\n        Pickle Support dumps()\n        '
        state = self.__dict__.copy()
        del state['_gtobjs']
        del state['_AppriseLocale__fn_map']
        return state

    def __setstate__(self, state):
        if False:
            return 10
        '\n        Pickle Support loads()\n        '
        self.__dict__.update(state)
        self.__fn_map = None
        self._gtobjs = {}
        self.add(state['lang'], set_default=True)
LOCALE = AppriseLocale()

class LazyTranslation:
    """
    Doesn't translate anything until str() or unicode() references
    are made.

    """

    def __init__(self, text, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Store our text\n        '
        self.text = text
        super().__init__(*args, **kwargs)

    def __str__(self):
        if False:
            return 10
        return LOCALE.gettext(self.text) if GETTEXT_LOADED else self.text

def gettext_lazy(text):
    if False:
        while True:
            i = 10
    '\n    A dummy function that can be referenced\n    '
    return LazyTranslation(text=text)