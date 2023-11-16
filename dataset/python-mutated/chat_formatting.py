from __future__ import annotations
import datetime
import itertools
import math
import textwrap
from io import BytesIO
from typing import Iterator, List, Optional, Sequence, SupportsInt, Union
import discord
from babel.lists import format_list as babel_list
from babel.numbers import format_decimal
from redbot.core.i18n import Translator, get_babel_locale, get_babel_regional_format
__all__ = ('error', 'warning', 'info', 'success', 'question', 'bold', 'box', 'inline', 'italics', 'spoiler', 'pagify', 'strikethrough', 'underline', 'quote', 'escape', 'humanize_list', 'format_perms_list', 'humanize_timedelta', 'humanize_number', 'text_to_file')
_ = Translator('UtilsChatFormatting', __file__)

def error(text: str) -> str:
    if False:
        print('Hello World!')
    'Get text prefixed with an error emoji.\n\n    Parameters\n    ----------\n    text : str\n        The text to be prefixed.\n\n    Returns\n    -------\n    str\n        The new message.\n\n    '
    return f'🚫 {text}'

def warning(text: str) -> str:
    if False:
        print('Hello World!')
    'Get text prefixed with a warning emoji.\n\n    Parameters\n    ----------\n    text : str\n        The text to be prefixed.\n\n    Returns\n    -------\n    str\n        The new message.\n\n    '
    return f'⚠️ {text}'

def info(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get text prefixed with an info emoji.\n\n    Parameters\n    ----------\n    text : str\n        The text to be prefixed.\n\n    Returns\n    -------\n    str\n        The new message.\n\n    '
    return f'ℹ️ {text}'

def success(text: str) -> str:
    if False:
        print('Hello World!')
    'Get text prefixed with a success emoji.\n\n    Parameters\n    ----------\n    text : str\n        The text to be prefixed.\n\n    Returns\n    -------\n    str\n        The new message.\n\n    '
    return f'✅ {text}'

def question(text: str) -> str:
    if False:
        while True:
            i = 10
    'Get text prefixed with a question emoji.\n\n    Parameters\n    ----------\n    text : str\n        The text to be prefixed.\n\n    Returns\n    -------\n    str\n        The new message.\n\n    '
    return f'❓️ {text}'

def bold(text: str, escape_formatting: bool=True) -> str:
    if False:
        print('Hello World!')
    'Get the given text in bold.\n\n    Note: By default, this function will escape ``text`` prior to emboldening.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n    escape_formatting : `bool`, optional\n        Set to :code:`False` to not escape markdown formatting in the text.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return f'**{escape(text, formatting=escape_formatting)}**'

def box(text: str, lang: str='') -> str:
    if False:
        while True:
            i = 10
    'Get the given text in a code block.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n    lang : `str`, optional\n        The syntax highlighting language for the codeblock.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return f'```{lang}\n{text}\n```'

def inline(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the given text as inline code.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    if '`' in text:
        return f'``{text}``'
    else:
        return f'`{text}`'

def italics(text: str, escape_formatting: bool=True) -> str:
    if False:
        print('Hello World!')
    'Get the given text in italics.\n\n    Note: By default, this function will escape ``text`` prior to italicising.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n    escape_formatting : `bool`, optional\n        Set to :code:`False` to not escape markdown formatting in the text.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return f'*{escape(text, formatting=escape_formatting)}*'

def spoiler(text: str, escape_formatting: bool=True) -> str:
    if False:
        print('Hello World!')
    'Get the given text as a spoiler.\n\n    Note: By default, this function will escape ``text`` prior to making the text a spoiler.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n    escape_formatting : `bool`, optional\n        Set to :code:`False` to not escape markdown formatting in the text.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return f'||{escape(text, formatting=escape_formatting)}||'

class pagify(Iterator[str]):
    """Generate multiple pages from the given text.

    The returned iterator supports length estimation with :func:`operator.length_hint()`.

    Note
    ----
    This does not respect code blocks or inline code.

    Parameters
    ----------
    text : str
        The content to pagify and send.
    delims : `sequence` of `str`, optional
        Characters where page breaks will occur. If no delimiters are found
        in a page, the page will break after ``page_length`` characters.
        By default this only contains the newline.

    Other Parameters
    ----------------
    priority : `bool`
        Set to :code:`True` to choose the page break delimiter based on the
        order of ``delims``. Otherwise, the page will always break at the
        last possible delimiter.
    escape_mass_mentions : `bool`
        If :code:`True`, any mass mentions (here or everyone) will be
        silenced.
    shorten_by : `int`
        How much to shorten each page by. Defaults to 8.
    page_length : `int`
        The maximum length of each page. Defaults to 2000.

    Yields
    ------
    `str`
        Pages of the given text.

    """

    def __init__(self, text: str, delims: Sequence[str]=('\n',), *, priority: bool=False, escape_mass_mentions: bool=True, shorten_by: int=8, page_length: int=2000) -> None:
        if False:
            return 10
        self._text = text
        self._delims = delims
        self._priority = priority
        self._escape_mass_mentions = escape_mass_mentions
        self._shorten_by = shorten_by
        self._page_length = page_length - shorten_by
        self._start = 0
        self._end = len(text)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        text = self._text
        if len(text) > 20:
            text = f'{text[:19]}…'
        return f'pagify({text!r}, {self._delims!r}, priority={self._priority!r}, escape_mass_mentions={self._escape_mass_mentions!r}, shorten_by={self._shorten_by!r}, page_length={self._page_length + self._shorten_by!r})'

    def __length_hint__(self) -> int:
        if False:
            while True:
                i = 10
        return math.ceil((self._end - self._start) / self._page_length)

    def __iter__(self) -> pagify:
        if False:
            while True:
                i = 10
        return self

    def __next__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        text = self._text
        escape_mass_mentions = self._escape_mass_mentions
        page_length = self._page_length
        start = self._start
        end = self._end
        while end - start > page_length:
            stop = start + page_length
            if escape_mass_mentions:
                stop -= text.count('@here', start, stop) + text.count('@everyone', start, stop)
            closest_delim_it = (text.rfind(d, start + 1, stop) for d in self._delims)
            if self._priority:
                closest_delim = next((x for x in closest_delim_it if x > 0), -1)
            else:
                closest_delim = max(closest_delim_it)
            stop = closest_delim if closest_delim != -1 else stop
            if escape_mass_mentions:
                to_send = escape(text[start:stop], mass_mentions=True)
            else:
                to_send = text[start:stop]
            start = self._start = stop
            if len(to_send.strip()) > 0:
                return to_send
        if len(text[start:end].strip()) > 0:
            self._start = end
            if escape_mass_mentions:
                return escape(text[start:end], mass_mentions=True)
            else:
                return text[start:end]
        raise StopIteration

def strikethrough(text: str, escape_formatting: bool=True) -> str:
    if False:
        i = 10
        return i + 15
    'Get the given text with a strikethrough.\n\n    Note: By default, this function will escape ``text`` prior to applying a strikethrough.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n    escape_formatting : `bool`, optional\n        Set to :code:`False` to not escape markdown formatting in the text.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return f'~~{escape(text, formatting=escape_formatting)}~~'

def underline(text: str, escape_formatting: bool=True) -> str:
    if False:
        i = 10
        return i + 15
    'Get the given text with an underline.\n\n    Note: By default, this function will escape ``text`` prior to underlining.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n    escape_formatting : `bool`, optional\n        Set to :code:`False` to not escape markdown formatting in the text.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return f'__{escape(text, formatting=escape_formatting)}__'

def quote(text: str) -> str:
    if False:
        return 10
    'Quotes the given text.\n\n    Parameters\n    ----------\n    text : str\n        The text to be marked up.\n\n    Returns\n    -------\n    str\n        The marked up text.\n\n    '
    return textwrap.indent(text, '> ', lambda l: True)

def escape(text: str, *, mass_mentions: bool=False, formatting: bool=False) -> str:
    if False:
        return 10
    'Get text with all mass mentions or markdown escaped.\n\n    Parameters\n    ----------\n    text : str\n        The text to be escaped.\n    mass_mentions : `bool`, optional\n        Set to :code:`True` to escape mass mentions in the text.\n    formatting : `bool`, optional\n        Set to :code:`True` to escape any markdown formatting in the text.\n\n    Returns\n    -------\n    str\n        The escaped text.\n\n    '
    if mass_mentions:
        text = text.replace('@everyone', '@\u200beveryone')
        text = text.replace('@here', '@\u200bhere')
    if formatting:
        text = discord.utils.escape_markdown(text)
    return text

def humanize_list(items: Sequence[str], *, locale: Optional[str]=None, style: str='standard') -> str:
    if False:
        i = 10
        return i + 15
    'Get comma-separated list, with the last element joined with *and*.\n\n    Parameters\n    ----------\n    items : Sequence[str]\n        The items of the list to join together.\n    locale : Optional[str]\n        The locale to convert, if not specified it defaults to the bot\'s locale.\n    style : str\n        The style to format the list with.\n\n        Note: Not all styles are necessarily available in all locales,\n        see documentation of `babel.lists.format_list` for more details.\n\n        standard\n            A typical \'and\' list for arbitrary placeholders.\n            eg. "January, February, and March"\n        standard-short\n             A short version of a \'and\' list, suitable for use with short or\n             abbreviated placeholder values.\n             eg. "Jan., Feb., and Mar."\n        or\n            A typical \'or\' list for arbitrary placeholders.\n            eg. "January, February, or March"\n        or-short\n            A short version of an \'or\' list.\n            eg. "Jan., Feb., or Mar."\n        unit\n            A list suitable for wide units.\n            eg. "3 feet, 7 inches"\n        unit-short\n            A list suitable for short units\n            eg. "3 ft, 7 in"\n        unit-narrow\n            A list suitable for narrow units, where space on the screen is very limited.\n            eg. "3′ 7″"\n\n    Raises\n    ------\n    ValueError\n        The locale does not support the specified style.\n\n    Examples\n    --------\n    .. testsetup::\n\n        from redbot.core.utils.chat_formatting import humanize_list\n\n    .. doctest::\n\n        >>> humanize_list([\'One\', \'Two\', \'Three\'])\n        \'One, Two, and Three\'\n        >>> humanize_list([\'One\'])\n        \'One\'\n        >>> humanize_list([\'omena\', \'peruna\', \'aplari\'], style=\'or\', locale=\'fi\')\n        \'omena, peruna tai aplari\'\n\n    '
    return babel_list(items, style=style, locale=get_babel_locale(locale))

def format_perms_list(perms: discord.Permissions) -> str:
    if False:
        while True:
            i = 10
    'Format a list of permission names.\n\n    This will return a humanized list of the names of all enabled\n    permissions in the provided `discord.Permissions` object.\n\n    Parameters\n    ----------\n    perms : discord.Permissions\n        The permissions object with the requested permissions to list\n        enabled.\n\n    Returns\n    -------\n    str\n        The humanized list.\n\n    '
    perm_names: List[str] = []
    for (perm, value) in perms:
        if value is True:
            perm_name = '"' + perm.replace('_', ' ').title() + '"'
            perm_names.append(perm_name)
    return humanize_list(perm_names).replace('Guild', 'Server')

def humanize_timedelta(*, timedelta: Optional[datetime.timedelta]=None, seconds: Optional[SupportsInt]=None) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Get a locale aware human timedelta representation.\n\n    This works with either a timedelta object or a number of seconds.\n\n    Fractional values will be omitted, and values less than 1 second\n    an empty string.\n\n    Parameters\n    ----------\n    timedelta: Optional[datetime.timedelta]\n        A timedelta object\n    seconds: Optional[SupportsInt]\n        A number of seconds\n\n    Returns\n    -------\n    str\n        A locale aware representation of the timedelta or seconds.\n\n    Raises\n    ------\n    ValueError\n        The function was called with neither a number of seconds nor a timedelta object\n    '
    try:
        obj = seconds if seconds is not None else timedelta.total_seconds()
    except AttributeError:
        raise ValueError('You must provide either a timedelta or a number of seconds')
    seconds = int(obj)
    periods = [(_('year'), _('years'), 60 * 60 * 24 * 365), (_('month'), _('months'), 60 * 60 * 24 * 30), (_('day'), _('days'), 60 * 60 * 24), (_('hour'), _('hours'), 60 * 60), (_('minute'), _('minutes'), 60), (_('second'), _('seconds'), 1)]
    strings = []
    for (period_name, plural_period_name, period_seconds) in periods:
        if seconds >= period_seconds:
            (period_value, seconds) = divmod(seconds, period_seconds)
            if period_value == 0:
                continue
            unit = plural_period_name if period_value > 1 else period_name
            strings.append(f'{period_value} {unit}')
    return ', '.join(strings)

def humanize_number(val: Union[int, float], override_locale=None) -> str:
    if False:
        print('Hello World!')
    "\n    Convert an int or float to a str with digit separators based on bot locale\n\n    Parameters\n    ----------\n    val : Union[int, float]\n        The int/float to be formatted.\n    override_locale: Optional[str]\n        A value to override bot's regional format.\n\n    Returns\n    -------\n    str\n        locale aware formatted number.\n    "
    return format_decimal(val, locale=get_babel_regional_format(override_locale))

def text_to_file(text: str, filename: str='file.txt', *, spoiler: bool=False, encoding: str='utf-8'):
    if False:
        return 10
    'Prepares text to be sent as a file on Discord, without character limit.\n\n    This writes text into a bytes object that can be used for the ``file`` or ``files`` parameters\n    of :meth:`discord.abc.Messageable.send`.\n\n    Parameters\n    ----------\n    text: str\n        The text to put in your file.\n    filename: str\n        The name of the file sent. Defaults to ``file.txt``.\n    spoiler: bool\n        Whether the attachment is a spoiler. Defaults to ``False``.\n\n    Returns\n    -------\n    discord.File\n        The file containing your text.\n\n    '
    file = BytesIO(text.encode(encoding))
    return discord.File(file, filename, spoiler=spoiler)