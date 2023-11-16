from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import string
import os
update_translations = 'RENPY_UPDATE_TRANSLATIONS' in os.environ

class Formatter(string.Formatter):
    """
    A string formatter that uses Ren'Py's formatting rules. Ren'Py uses
    square brackets to introduce formatting, and it supports a q conversion
    that quotes the text being shown to the user.
    """

    def parse(self, s):
        if False:
            i = 10
            return i + 15
        "\n        Parses s according to Ren'Py string formatting rules. Returns a list\n        of (literal_text, field_name, format, replacement) tuples, just like\n        the method we're overriding.\n        "
        LITERAL = 0
        OPEN_BRACKET = 1
        VALUE = 3
        FORMAT = 4
        CONVERSION = 5
        bracket_depth = 0
        literal = ''
        value = ''
        format = ''
        conversion = None
        state = LITERAL
        for c in s:
            if state == LITERAL:
                if c == '[':
                    state = OPEN_BRACKET
                    continue
                else:
                    literal += c
                    continue
            elif state == OPEN_BRACKET:
                if c == '[':
                    literal += c
                    state = LITERAL
                    continue
                else:
                    value = c
                    state = VALUE
                    bracket_depth = 0
                    continue
            elif state == VALUE:
                if c == '[':
                    bracket_depth += 1
                    value += c
                    continue
                elif c == ']':
                    if bracket_depth:
                        bracket_depth -= 1
                        value += c
                        continue
                    else:
                        yield (literal, value, format, conversion)
                        state = LITERAL
                        literal = ''
                        value = ''
                        format = ''
                        conversion = None
                        continue
                elif c == ':':
                    state = FORMAT
                    continue
                elif c == '!':
                    state = CONVERSION
                    conversion = ''
                    continue
                else:
                    value += c
                    continue
            elif state == FORMAT:
                if c == ']':
                    yield (literal, value, format, conversion)
                    state = LITERAL
                    literal = ''
                    value = ''
                    format = ''
                    conversion = None
                    continue
                elif c == '!':
                    state = CONVERSION
                    conversion = ''
                    continue
                else:
                    format += c
                    continue
            elif state == CONVERSION:
                if c == ']':
                    yield (literal, value, format, conversion)
                    state = LITERAL
                    literal = ''
                    value = ''
                    format = ''
                    conversion = None
                    continue
                else:
                    conversion += c
                    continue
        if state != LITERAL:
            raise Exception('String {0!r} ends with an open format operation.'.format(s))
        if literal:
            yield (literal, None, None, None)

    def get_field(self, field_name, args, kwargs):
        if False:
            i = 10
            return i + 15
        (obj, arg_used) = super(Formatter, self).get_field(field_name, args, kwargs)
        return ((obj, kwargs), arg_used)

    def convert_field(self, value, conversion):
        if False:
            i = 10
            return i + 15
        (value, kwargs) = value
        if conversion is None:
            return value
        if not conversion:
            raise ValueError("Conversion specifier can't be empty.")
        if set(conversion) - set('rstqulci!'):
            raise ValueError('Unknown symbols in conversion specifier, this must use only the "rstqulci".')
        if 'r' in conversion:
            value = repr(value)
            conversion = conversion.replace('r', '')
        elif 's' in conversion:
            value = str(value)
            conversion = conversion.replace('s', '')
        if not conversion:
            return value
        if not isinstance(value, basestring):
            value = str(value)
        if 't' in conversion:
            value = renpy.translation.translate_string(value)
        if 'i' in conversion:
            try:
                value = self.vformat(value, (), kwargs)
            except RuntimeError:
                raise ValueError('Substitution {!r} refers to itself in a loop.'.format(value))
        if 'q' in conversion:
            value = value.replace('{', '{{')
        if 'u' in conversion:
            value = value.upper()
        if 'l' in conversion:
            value = value.lower()
        if 'c' in conversion and value:
            value = value[0].upper() + value[1:]
        return value
formatter = Formatter()

class MultipleDict(object):

    def __init__(self, *dicts):
        if False:
            for i in range(10):
                print('nop')
        self.dicts = dicts

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        for d in self.dicts:
            if key in d:
                return d[key]
        raise NameError("Name '{}' is not defined.".format(key))

def substitute(s, scope=None, force=False, translate=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Performs translation and formatting on `s`, as necessary.\n\n    `scope`\n        The scope which is used in formatting, in addition to the default\n        store.\n\n    `force`\n        Force substitution to occur, even if it's disabled in the config.\n\n    `translate`\n        Determines if translation occurs.\n\n    Returns the substituted string, and a flag that is True if substitution\n    occurred, or False if no substitution occurred.\n    "
    if not isinstance(s, basestring):
        s = str(s)
    if translate:
        s = renpy.translation.translate_string(s)
    if not renpy.config.new_substitutions and (not force):
        return (s, False)
    if '[' not in s:
        return (s, False)
    old_s = s
    dicts = [renpy.store.__dict__]
    if 'store.interpolate' in renpy.python.store_dicts:
        dicts.insert(0, renpy.python.store_dicts['store.interpolate'])
    if scope is not None:
        dicts.insert(0, scope)
    if dicts:
        kwargs = MultipleDict(*dicts)
    else:
        kwargs = dicts[0]
    try:
        s = formatter.vformat(s, (), kwargs)
    except Exception:
        if renpy.display.predict.predicting:
            return (' ', True)
        raise
    return (s, s != old_s)