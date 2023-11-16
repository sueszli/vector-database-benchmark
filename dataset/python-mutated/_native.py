"""
markupsafe._native
~~~~~~~~~~~~~~~~~~

Native Python implementation used when the C module is not compiled.

:copyright: 2010 Pallets
:license: BSD-3-Clause
"""
from . import Markup
from ._compat import text_type

def escape(s):
    if False:
        return 10
    'Replace the characters ``&``, ``<``, ``>``, ``\'``, and ``"`` in\n    the string with HTML-safe sequences. Use this if you need to display\n    text that might contain such characters in HTML.\n\n    If the object has an ``__html__`` method, it is called and the\n    return value is assumed to already be safe for HTML.\n\n    :param s: An object to be converted to a string and escaped.\n    :return: A :class:`Markup` string with the escaped text.\n    '
    if hasattr(s, '__html__'):
        return Markup(s.__html__())
    return Markup(text_type(s).replace('&', '&amp;').replace('>', '&gt;').replace('<', '&lt;').replace("'", '&#39;').replace('"', '&#34;'))

def escape_silent(s):
    if False:
        i = 10
        return i + 15
    "Like :func:`escape` but treats ``None`` as the empty string.\n    Useful with optional values, as otherwise you get the string\n    ``'None'`` when the value is ``None``.\n\n    >>> escape(None)\n    Markup('None')\n    >>> escape_silent(None)\n    Markup('')\n    "
    if s is None:
        return Markup()
    return escape(s)

def soft_unicode(s):
    if False:
        while True:
            i = 10
    "Convert an object to a string if it isn't already. This preserves\n    a :class:`Markup` string rather than converting it back to a basic\n    string, so it will still be marked as safe and won't be escaped\n    again.\n\n    >>> value = escape('<User 1>')\n    >>> value\n    Markup('&lt;User 1&gt;')\n    >>> escape(str(value))\n    Markup('&amp;lt;User 1&amp;gt;')\n    >>> escape(soft_unicode(value))\n    Markup('&lt;User 1&gt;')\n    "
    if not isinstance(s, text_type):
        s = text_type(s)
    return s