from __future__ import annotations
import contextlib
import functools
import io
import os
import logging
import discord
from pathlib import Path
from typing import Callable, TYPE_CHECKING, Union, Dict, Optional, TypeVar
from contextvars import ContextVar
import babel.localedata
from babel.core import Locale
if TYPE_CHECKING:
    from redbot.core.bot import Red
__all__ = ['get_locale', 'get_regional_format', 'get_locale_from_guild', 'get_regional_format_from_guild', 'set_contextual_locales_from_guild', 'Translator', 'get_babel_locale', 'get_babel_regional_format', 'cog_i18n']
log = logging.getLogger('red.i18n')
_current_locale = ContextVar('_current_locale', default='en-US')
_current_regional_format = ContextVar('_current_regional_format', default=None)
WAITING_FOR_MSGID = 1
IN_MSGID = 2
WAITING_FOR_MSGSTR = 3
IN_MSGSTR = 4
MSGID = 'msgid "'
MSGSTR = 'msgstr "'
_translators = []

def get_locale() -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get locale in a current context.\n\n    Returns\n    -------\n    str\n        Current locale\'s language code with country code included, e.g. "en-US".\n    '
    return str(_current_locale.get())

def set_locale(locale: str) -> None:
    if False:
        while True:
            i = 10
    global _current_locale
    _current_locale = ContextVar('_current_locale', default=locale)
    reload_locales()

def set_contextual_locale(locale: str) -> None:
    if False:
        print('Hello World!')
    _current_locale.set(locale)
    reload_locales()

def get_regional_format() -> str:
    if False:
        i = 10
        return i + 15
    '\n    Get regional format in a current context.\n\n    Returns\n    -------\n    str\n        Current regional format\'s language code with country code included, e.g. "en-US".\n    '
    if _current_regional_format.get() is None:
        return str(_current_locale.get())
    return str(_current_regional_format.get())

def set_regional_format(regional_format: Optional[str]) -> None:
    if False:
        i = 10
        return i + 15
    global _current_regional_format
    _current_regional_format = ContextVar('_current_regional_format', default=regional_format)

def set_contextual_regional_format(regional_format: Optional[str]) -> None:
    if False:
        while True:
            i = 10
    _current_regional_format.set(regional_format)

def reload_locales() -> None:
    if False:
        return 10
    for translator in _translators:
        translator.load_translations()

async def get_locale_from_guild(bot: Red, guild: Optional[discord.Guild]) -> str:
    """
    Get locale set for the given guild.

    Parameters
    ----------
    bot: Red
         The bot's instance.
    guild: Optional[discord.Guild]
         The guild contextual locale is set for.
         Use `None` if the context doesn't involve guild.

    Returns
    -------
    str
        Guild locale's language code with country code included, e.g. "en-US".
    """
    return await bot._i18n_cache.get_locale(guild)

async def get_regional_format_from_guild(bot: Red, guild: Optional[discord.Guild]) -> str:
    """
    Get regional format for the given guild.

    Parameters
    ----------
    bot: Red
         The bot's instance.
    guild: Optional[discord.Guild]
         The guild contextual locale is set for.
         Use `None` if the context doesn't involve guild.

    Returns
    -------
    str
        Guild regional format's language code with country code included, e.g. "en-US".
    """
    return await bot._i18n_cache.get_regional_format(guild)

async def set_contextual_locales_from_guild(bot: Red, guild: Optional[discord.Guild]) -> None:
    """
    Set contextual locales (locale and regional format) for given guild context.

    Parameters
    ----------
    bot: Red
         The bot's instance.
    guild: Optional[discord.Guild]
         The guild contextual locale is set for.
         Use `None` if the context doesn't involve guild.
    """
    locale = await get_locale_from_guild(bot, guild)
    regional_format = await get_regional_format_from_guild(bot, guild)
    set_contextual_locale(locale)
    set_contextual_regional_format(regional_format)

def _parse(translation_file: io.TextIOWrapper) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    '\n    Custom gettext parsing of translation files.\n\n    Parameters\n    ----------\n    translation_file : io.TextIOWrapper\n        An open text file containing translations.\n\n    Returns\n    -------\n    Dict[str, str]\n        A dict mapping the original strings to their translations. Empty\n        translated strings are omitted.\n\n    '
    step = None
    untranslated = ''
    translated = ''
    translations = {}
    locale = get_locale()
    translations[locale] = {}
    for line in translation_file:
        line = line.strip()
        if line.startswith(MSGID):
            if step is IN_MSGSTR and translated:
                translations[locale][_unescape(untranslated)] = _unescape(translated)
            step = IN_MSGID
            untranslated = line[len(MSGID):-1]
        elif line.startswith('"') and line.endswith('"'):
            if step is IN_MSGID:
                untranslated += line[1:-1]
            elif step is IN_MSGSTR:
                translated += line[1:-1]
        elif line.startswith(MSGSTR):
            step = IN_MSGSTR
            translated = line[len(MSGSTR):-1]
    if step is IN_MSGSTR and translated:
        translations[locale][_unescape(untranslated)] = _unescape(translated)
    return translations

def _unescape(string):
    if False:
        for i in range(10):
            print('nop')
    string = string.replace('\\\\', '\\')
    string = string.replace('\\t', '\t')
    string = string.replace('\\r', '\r')
    string = string.replace('\\n', '\n')
    string = string.replace('\\"', '"')
    return string

def get_locale_path(cog_folder: Path, extension: str) -> Path:
    if False:
        return 10
    '\n    Gets the folder path containing localization files.\n\n    :param Path cog_folder:\n        The cog folder that we want localizations for.\n    :param str extension:\n        Extension of localization files.\n    :return:\n        Path of possible localization file, it may not exist.\n    '
    return cog_folder / 'locales' / '{}.{}'.format(get_locale(), extension)

class Translator(Callable[[str], str]):
    """Function to get translated strings at runtime."""

    def __init__(self, name: str, file_location: Union[str, Path, os.PathLike]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes an internationalization object.\n\n        Parameters\n        ----------\n        name : str\n            Your cog name.\n        file_location : `str` or `pathlib.Path`\n            This should always be ``__file__`` otherwise your localizations\n            will not load.\n\n        '
        self.cog_folder = Path(file_location).resolve().parent
        self.cog_name = name
        self.translations = {}
        _translators.append(self)
        self.load_translations()

    def __call__(self, untranslated: str) -> str:
        if False:
            i = 10
            return i + 15
        "Translate the given string.\n\n        This will look for the string in the translator's :code:`.pot` file,\n        with respect to the current locale.\n        "
        locale = get_locale()
        try:
            return self.translations[locale][untranslated]
        except KeyError:
            return untranslated

    def load_translations(self):
        if False:
            print('Hello World!')
        '\n        Loads the current translations.\n        '
        locale = get_locale()
        if locale.lower() == 'en-us':
            return
        if locale in self.translations:
            return
        locale_path = get_locale_path(self.cog_folder, 'po')
        with contextlib.suppress(IOError, FileNotFoundError):
            with locale_path.open(encoding='utf-8') as file:
                self._parse(file)

    def _parse(self, translation_file):
        if False:
            return 10
        self.translations.update(_parse(translation_file))

    def _add_translation(self, untranslated, translated):
        if False:
            return 10
        untranslated = _unescape(untranslated)
        translated = _unescape(translated)
        if translated:
            self.translations[untranslated] = translated

@functools.lru_cache()
def _get_babel_locale(red_locale: str) -> babel.core.Locale:
    if False:
        return 10
    supported_locales = babel.localedata.locale_identifiers()
    try:
        babel_locale = Locale(*babel.parse_locale(red_locale))
    except (ValueError, babel.core.UnknownLocaleError):
        try:
            babel_locale = Locale(*babel.parse_locale(red_locale, sep='-'))
        except (ValueError, babel.core.UnknownLocaleError):
            try:
                babel_locale = Locale(Locale.negotiate([red_locale], supported_locales, sep='-'))
            except (ValueError, TypeError, babel.core.UnknownLocaleError):
                babel_locale = Locale('en', 'US')
    return babel_locale

def get_babel_locale(locale: Optional[str]=None) -> babel.core.Locale:
    if False:
        print('Hello World!')
    "Function to convert a locale to a `babel.core.Locale`.\n\n    Parameters\n    ----------\n    locale : Optional[str]\n        The locale to convert, if not specified it defaults to the bot's locale.\n\n    Returns\n    -------\n    babel.core.Locale\n        The babel locale object.\n    "
    if locale is None:
        locale = get_locale()
    return _get_babel_locale(locale)

def get_babel_regional_format(regional_format: Optional[str]=None) -> babel.core.Locale:
    if False:
        return 10
    "Function to convert a regional format to a `babel.core.Locale`.\n\n    If ``regional_format`` parameter is passed, this behaves the same as `get_babel_locale`.\n\n    Parameters\n    ----------\n    regional_format : Optional[str]\n        The regional format to convert, if not specified it defaults to the bot's regional format.\n\n    Returns\n    -------\n    babel.core.Locale\n        The babel locale object.\n    "
    if regional_format is None:
        regional_format = get_regional_format()
    return _get_babel_locale(regional_format)
from . import commands
_TypeT = TypeVar('_TypeT', bound=type)

def cog_i18n(translator: Translator) -> Callable[[_TypeT], _TypeT]:
    if False:
        return 10
    'Get a class decorator to link the translator to this cog.'

    def decorator(cog_class: _TypeT) -> _TypeT:
        if False:
            return 10
        cog_class.__translator__ = translator
        for (name, attr) in cog_class.__dict__.items():
            if isinstance(attr, (commands.Group, commands.Command)):
                attr.translator = translator
                setattr(cog_class, name, attr)
        return cog_class
    return decorator