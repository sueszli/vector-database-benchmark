from __future__ import annotations
import codecs
import re
from .structures import ImmutableList

class Accept(ImmutableList):
    """An :class:`Accept` object is just a list subclass for lists of
    ``(value, quality)`` tuples.  It is automatically sorted by specificity
    and quality.

    All :class:`Accept` objects work similar to a list but provide extra
    functionality for working with the data.  Containment checks are
    normalized to the rules of that header:

    >>> a = CharsetAccept([('ISO-8859-1', 1), ('utf-8', 0.7)])
    >>> a.best
    'ISO-8859-1'
    >>> 'iso-8859-1' in a
    True
    >>> 'UTF8' in a
    True
    >>> 'utf7' in a
    False

    To get the quality for an item you can use normal item lookup:

    >>> print a['utf-8']
    0.7
    >>> a['utf7']
    0

    .. versionchanged:: 0.5
       :class:`Accept` objects are forced immutable now.

    .. versionchanged:: 1.0.0
       :class:`Accept` internal values are no longer ordered
       alphabetically for equal quality tags. Instead the initial
       order is preserved.

    """

    def __init__(self, values=()):
        if False:
            while True:
                i = 10
        if values is None:
            list.__init__(self)
            self.provided = False
        elif isinstance(values, Accept):
            self.provided = values.provided
            list.__init__(self, values)
        else:
            self.provided = True
            values = sorted(values, key=lambda x: (self._specificity(x[0]), x[1]), reverse=True)
            list.__init__(self, values)

    def _specificity(self, value):
        if False:
            while True:
                i = 10
        "Returns a tuple describing the value's specificity."
        return (value != '*',)

    def _value_matches(self, value, item):
        if False:
            i = 10
            return i + 15
        'Check if a value matches a given accept item.'
        return item == '*' or item.lower() == value.lower()

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Besides index lookup (getting item n) you can also pass it a string\n        to get the quality for the item.  If the item is not in the list, the\n        returned quality is ``0``.\n        '
        if isinstance(key, str):
            return self.quality(key)
        return list.__getitem__(self, key)

    def quality(self, key):
        if False:
            print('Hello World!')
        'Returns the quality of the key.\n\n        .. versionadded:: 0.6\n           In previous versions you had to use the item-lookup syntax\n           (eg: ``obj[key]`` instead of ``obj.quality(key)``)\n        '
        for (item, quality) in self:
            if self._value_matches(key, item):
                return quality
        return 0

    def __contains__(self, value):
        if False:
            i = 10
            return i + 15
        for (item, _quality) in self:
            if self._value_matches(value, item):
                return True
        return False

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        pairs_str = ', '.join((f'({x!r}, {y})' for (x, y) in self))
        return f'{type(self).__name__}([{pairs_str}])'

    def index(self, key):
        if False:
            print('Hello World!')
        'Get the position of an entry or raise :exc:`ValueError`.\n\n        :param key: The key to be looked up.\n\n        .. versionchanged:: 0.5\n           This used to raise :exc:`IndexError`, which was inconsistent\n           with the list API.\n        '
        if isinstance(key, str):
            for (idx, (item, _quality)) in enumerate(self):
                if self._value_matches(key, item):
                    return idx
            raise ValueError(key)
        return list.index(self, key)

    def find(self, key):
        if False:
            i = 10
            return i + 15
        'Get the position of an entry or return -1.\n\n        :param key: The key to be looked up.\n        '
        try:
            return self.index(key)
        except ValueError:
            return -1

    def values(self):
        if False:
            return 10
        'Iterate over all values.'
        for item in self:
            yield item[0]

    def to_header(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert the header set into an HTTP header string.'
        result = []
        for (value, quality) in self:
            if quality != 1:
                value = f'{value};q={quality}'
            result.append(value)
        return ','.join(result)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.to_header()

    def _best_single_match(self, match):
        if False:
            while True:
                i = 10
        for (client_item, quality) in self:
            if self._value_matches(match, client_item):
                return (client_item, quality)
        return None

    def best_match(self, matches, default=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the best match from a list of possible matches based\n        on the specificity and quality of the client. If two items have the\n        same quality and specificity, the one is returned that comes first.\n\n        :param matches: a list of matches to check for\n        :param default: the value that is returned if none match\n        '
        result = default
        best_quality = -1
        best_specificity = (-1,)
        for server_item in matches:
            match = self._best_single_match(server_item)
            if not match:
                continue
            (client_item, quality) = match
            specificity = self._specificity(client_item)
            if quality <= 0 or quality < best_quality:
                continue
            if quality > best_quality or specificity > best_specificity:
                result = server_item
                best_quality = quality
                best_specificity = specificity
        return result

    @property
    def best(self):
        if False:
            print('Hello World!')
        'The best match as value.'
        if self:
            return self[0][0]
_mime_split_re = re.compile('/|(?:\\s*;\\s*)')

def _normalize_mime(value):
    if False:
        while True:
            i = 10
    return _mime_split_re.split(value.lower())

class MIMEAccept(Accept):
    """Like :class:`Accept` but with special methods and behavior for
    mimetypes.
    """

    def _specificity(self, value):
        if False:
            while True:
                i = 10
        return tuple((x != '*' for x in _mime_split_re.split(value)))

    def _value_matches(self, value, item):
        if False:
            while True:
                i = 10
        if '/' not in item:
            return False
        if '/' not in value:
            raise ValueError(f'invalid mimetype {value!r}')
        normalized_value = _normalize_mime(value)
        (value_type, value_subtype) = normalized_value[:2]
        value_params = sorted(normalized_value[2:])
        if value_type == '*' and value_subtype != '*':
            raise ValueError(f'invalid mimetype {value!r}')
        normalized_item = _normalize_mime(item)
        (item_type, item_subtype) = normalized_item[:2]
        item_params = sorted(normalized_item[2:])
        if item_type == '*' and item_subtype != '*':
            return False
        return (item_type == '*' and item_subtype == '*' or (value_type == '*' and value_subtype == '*')) or (item_type == value_type and (item_subtype == '*' or value_subtype == '*' or (item_subtype == value_subtype and item_params == value_params)))

    @property
    def accept_html(self):
        if False:
            print('Hello World!')
        'True if this object accepts HTML.'
        return 'text/html' in self or 'application/xhtml+xml' in self or self.accept_xhtml

    @property
    def accept_xhtml(self):
        if False:
            for i in range(10):
                print('nop')
        'True if this object accepts XHTML.'
        return 'application/xhtml+xml' in self or 'application/xml' in self

    @property
    def accept_json(self):
        if False:
            for i in range(10):
                print('nop')
        'True if this object accepts JSON.'
        return 'application/json' in self
_locale_delim_re = re.compile('[_-]')

def _normalize_lang(value):
    if False:
        for i in range(10):
            print('nop')
    'Process a language tag for matching.'
    return _locale_delim_re.split(value.lower())

class LanguageAccept(Accept):
    """Like :class:`Accept` but with normalization for language tags."""

    def _value_matches(self, value, item):
        if False:
            return 10
        return item == '*' or _normalize_lang(value) == _normalize_lang(item)

    def best_match(self, matches, default=None):
        if False:
            while True:
                i = 10
        'Given a list of supported values, finds the best match from\n        the list of accepted values.\n\n        Language tags are normalized for the purpose of matching, but\n        are returned unchanged.\n\n        If no exact match is found, this will fall back to matching\n        the first subtag (primary language only), first with the\n        accepted values then with the match values. This partial is not\n        applied to any other language subtags.\n\n        The default is returned if no exact or fallback match is found.\n\n        :param matches: A list of supported languages to find a match.\n        :param default: The value that is returned if none match.\n        '
        result = super().best_match(matches)
        if result is not None:
            return result
        fallback = Accept([(_locale_delim_re.split(item[0], 1)[0], item[1]) for item in self])
        result = fallback.best_match(matches)
        if result is not None:
            return result
        fallback_matches = [_locale_delim_re.split(item, 1)[0] for item in matches]
        result = super().best_match(fallback_matches)
        if result is not None:
            return next((item for item in matches if item.startswith(result)))
        return default

class CharsetAccept(Accept):
    """Like :class:`Accept` but with normalization for charsets."""

    def _value_matches(self, value, item):
        if False:
            print('Hello World!')

        def _normalize(name):
            if False:
                return 10
            try:
                return codecs.lookup(name).name
            except LookupError:
                return name.lower()
        return item == '*' or _normalize(value) == _normalize(item)