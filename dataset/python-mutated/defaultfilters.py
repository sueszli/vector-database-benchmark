"""Default variable filters."""
import random as random_module
import re
import types
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation, getcontext
from functools import wraps
from inspect import unwrap
from operator import itemgetter
from pprint import pformat
from urllib.parse import quote
from django.utils import formats
from django.utils.dateformat import format, time_format
from django.utils.encoding import iri_to_uri
from django.utils.html import avoid_wrapping, conditional_escape, escape, escapejs
from django.utils.html import json_script as _json_script
from django.utils.html import linebreaks, strip_tags
from django.utils.html import urlize as _urlize
from django.utils.safestring import SafeData, mark_safe
from django.utils.text import Truncator, normalize_newlines, phone2numeric
from django.utils.text import slugify as _slugify
from django.utils.text import wrap
from django.utils.timesince import timesince, timeuntil
from django.utils.translation import gettext, ngettext
from .base import VARIABLE_ATTRIBUTE_SEPARATOR
from .library import Library
register = Library()

def stringfilter(func):
    if False:
        return 10
    '\n    Decorator for filters which should only receive strings. The object\n    passed as the first positional argument will be converted to a string.\n    '

    @wraps(func)
    def _dec(first, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        first = str(first)
        result = func(first, *args, **kwargs)
        if isinstance(first, SafeData) and getattr(unwrap(func), 'is_safe', False):
            result = mark_safe(result)
        return result
    return _dec

@register.filter(is_safe=True)
@stringfilter
def addslashes(value):
    if False:
        i = 10
        return i + 15
    '\n    Add slashes before quotes. Useful for escaping strings in CSV, for\n    example. Less useful for escaping JavaScript; use the ``escapejs``\n    filter instead.\n    '
    return value.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")

@register.filter(is_safe=True)
@stringfilter
def capfirst(value):
    if False:
        while True:
            i = 10
    'Capitalize the first character of the value.'
    return value and value[0].upper() + value[1:]

@register.filter('escapejs')
@stringfilter
def escapejs_filter(value):
    if False:
        return 10
    'Hex encode characters for use in JavaScript strings.'
    return escapejs(value)

@register.filter(is_safe=True)
def json_script(value, element_id=None):
    if False:
        while True:
            i = 10
    '\n    Output value JSON-encoded, wrapped in a <script type="application/json">\n    tag (with an optional id).\n    '
    return _json_script(value, element_id)

@register.filter(is_safe=True)
def floatformat(text, arg=-1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Display a float to a specified number of decimal places.\n\n    If called without an argument, display the floating point number with one\n    decimal place -- but only if there\'s a decimal place to be displayed:\n\n    * num1 = 34.23234\n    * num2 = 34.00000\n    * num3 = 34.26000\n    * {{ num1|floatformat }} displays "34.2"\n    * {{ num2|floatformat }} displays "34"\n    * {{ num3|floatformat }} displays "34.3"\n\n    If arg is positive, always display exactly arg number of decimal places:\n\n    * {{ num1|floatformat:3 }} displays "34.232"\n    * {{ num2|floatformat:3 }} displays "34.000"\n    * {{ num3|floatformat:3 }} displays "34.260"\n\n    If arg is negative, display arg number of decimal places -- but only if\n    there are places to be displayed:\n\n    * {{ num1|floatformat:"-3" }} displays "34.232"\n    * {{ num2|floatformat:"-3" }} displays "34"\n    * {{ num3|floatformat:"-3" }} displays "34.260"\n\n    If arg has the \'g\' suffix, force the result to be grouped by the\n    THOUSAND_SEPARATOR for the active locale. When the active locale is\n    en (English):\n\n    * {{ 6666.6666|floatformat:"2g" }} displays "6,666.67"\n    * {{ 10000|floatformat:"g" }} displays "10,000"\n\n    If arg has the \'u\' suffix, force the result to be unlocalized. When the\n    active locale is pl (Polish):\n\n    * {{ 66666.6666|floatformat:"2" }} displays "66666,67"\n    * {{ 66666.6666|floatformat:"2u" }} displays "66666.67"\n\n    If the input float is infinity or NaN, display the string representation\n    of that value.\n    '
    force_grouping = False
    use_l10n = True
    if isinstance(arg, str):
        last_char = arg[-1]
        if arg[-2:] in {'gu', 'ug'}:
            force_grouping = True
            use_l10n = False
            arg = arg[:-2] or -1
        elif last_char == 'g':
            force_grouping = True
            arg = arg[:-1] or -1
        elif last_char == 'u':
            use_l10n = False
            arg = arg[:-1] or -1
    try:
        input_val = str(text)
        d = Decimal(input_val)
    except InvalidOperation:
        try:
            d = Decimal(str(float(text)))
        except (ValueError, InvalidOperation, TypeError):
            return ''
    try:
        p = int(arg)
    except ValueError:
        return input_val
    try:
        m = int(d) - d
    except (ValueError, OverflowError, InvalidOperation):
        return input_val
    if not m and p <= 0:
        return mark_safe(formats.number_format('%d' % int(d), 0, use_l10n=use_l10n, force_grouping=force_grouping))
    exp = Decimal(1).scaleb(-abs(p))
    tupl = d.as_tuple()
    units = len(tupl[1])
    units += -tupl[2] if m else tupl[2]
    prec = abs(p) + units + 1
    prec = max(getcontext().prec, prec)
    rounded_d = d.quantize(exp, ROUND_HALF_UP, Context(prec=prec))
    (sign, digits, exponent) = rounded_d.as_tuple()
    digits = [str(digit) for digit in reversed(digits)]
    while len(digits) <= abs(exponent):
        digits.append('0')
    digits.insert(-exponent, '.')
    if sign and rounded_d:
        digits.append('-')
    number = ''.join(reversed(digits))
    return mark_safe(formats.number_format(number, abs(p), use_l10n=use_l10n, force_grouping=force_grouping))

@register.filter(is_safe=True)
@stringfilter
def iriencode(value):
    if False:
        print('Hello World!')
    'Escape an IRI value for use in a URL.'
    return iri_to_uri(value)

@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def linenumbers(value, autoescape=True):
    if False:
        while True:
            i = 10
    'Display text with line numbers.'
    lines = value.split('\n')
    width = str(len(str(len(lines))))
    if not autoescape or isinstance(value, SafeData):
        for (i, line) in enumerate(lines):
            lines[i] = ('%0' + width + 'd. %s') % (i + 1, line)
    else:
        for (i, line) in enumerate(lines):
            lines[i] = ('%0' + width + 'd. %s') % (i + 1, escape(line))
    return mark_safe('\n'.join(lines))

@register.filter(is_safe=True)
@stringfilter
def lower(value):
    if False:
        print('Hello World!')
    'Convert a string into all lowercase.'
    return value.lower()

@register.filter(is_safe=False)
@stringfilter
def make_list(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the value turned into a list.\n\n    For an integer, it's a list of digits.\n    For a string, it's a list of characters.\n    "
    return list(value)

@register.filter(is_safe=True)
@stringfilter
def slugify(value):
    if False:
        print('Hello World!')
    "\n    Convert to ASCII. Convert spaces to hyphens. Remove characters that aren't\n    alphanumerics, underscores, or hyphens. Convert to lowercase. Also strip\n    leading and trailing whitespace.\n    "
    return _slugify(value)

@register.filter(is_safe=True)
def stringformat(value, arg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Format the variable according to the arg, a string formatting specifier.\n\n    This specifier uses Python string formatting syntax, with the exception\n    that the leading "%" is dropped.\n\n    See https://docs.python.org/library/stdtypes.html#printf-style-string-formatting\n    for documentation of Python string formatting.\n    '
    if isinstance(value, tuple):
        value = str(value)
    try:
        return ('%' + str(arg)) % value
    except (ValueError, TypeError):
        return ''

@register.filter(is_safe=True)
@stringfilter
def title(value):
    if False:
        return 10
    'Convert a string into titlecase.'
    t = re.sub("([a-z])'([A-Z])", lambda m: m[0].lower(), value.title())
    return re.sub('\\d([A-Z])', lambda m: m[0].lower(), t)

@register.filter(is_safe=True)
@stringfilter
def truncatechars(value, arg):
    if False:
        i = 10
        return i + 15
    'Truncate a string after `arg` number of characters.'
    try:
        length = int(arg)
    except ValueError:
        return value
    return Truncator(value).chars(length)

@register.filter(is_safe=True)
@stringfilter
def truncatechars_html(value, arg):
    if False:
        i = 10
        return i + 15
    '\n    Truncate HTML after `arg` number of chars.\n    Preserve newlines in the HTML.\n    '
    try:
        length = int(arg)
    except ValueError:
        return value
    return Truncator(value).chars(length, html=True)

@register.filter(is_safe=True)
@stringfilter
def truncatewords(value, arg):
    if False:
        while True:
            i = 10
    '\n    Truncate a string after `arg` number of words.\n    Remove newlines within the string.\n    '
    try:
        length = int(arg)
    except ValueError:
        return value
    return Truncator(value).words(length, truncate=' …')

@register.filter(is_safe=True)
@stringfilter
def truncatewords_html(value, arg):
    if False:
        return 10
    '\n    Truncate HTML after `arg` number of words.\n    Preserve newlines in the HTML.\n    '
    try:
        length = int(arg)
    except ValueError:
        return value
    return Truncator(value).words(length, html=True, truncate=' …')

@register.filter(is_safe=False)
@stringfilter
def upper(value):
    if False:
        for i in range(10):
            print('nop')
    'Convert a string into all uppercase.'
    return value.upper()

@register.filter(is_safe=False)
@stringfilter
def urlencode(value, safe=None):
    if False:
        return 10
    "\n    Escape a value for use in a URL.\n\n    The ``safe`` parameter determines the characters which should not be\n    escaped by Python's quote() function. If not provided, use the default safe\n    characters (but an empty string can be provided when *all* characters\n    should be escaped).\n    "
    kwargs = {}
    if safe is not None:
        kwargs['safe'] = safe
    return quote(value, **kwargs)

@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def urlize(value, autoescape=True):
    if False:
        i = 10
        return i + 15
    'Convert URLs in plain text into clickable links.'
    return mark_safe(_urlize(value, nofollow=True, autoescape=autoescape))

@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def urlizetrunc(value, limit, autoescape=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert URLs into clickable links, truncating URLs to the given character\n    limit, and adding 'rel=nofollow' attribute to discourage spamming.\n\n    Argument: Length to truncate URLs to.\n    "
    return mark_safe(_urlize(value, trim_url_limit=int(limit), nofollow=True, autoescape=autoescape))

@register.filter(is_safe=False)
@stringfilter
def wordcount(value):
    if False:
        i = 10
        return i + 15
    'Return the number of words.'
    return len(value.split())

@register.filter(is_safe=True)
@stringfilter
def wordwrap(value, arg):
    if False:
        for i in range(10):
            print('nop')
    'Wrap words at `arg` line length.'
    return wrap(value, int(arg))

@register.filter(is_safe=True)
@stringfilter
def ljust(value, arg):
    if False:
        i = 10
        return i + 15
    'Left-align the value in a field of a given width.'
    return value.ljust(int(arg))

@register.filter(is_safe=True)
@stringfilter
def rjust(value, arg):
    if False:
        i = 10
        return i + 15
    'Right-align the value in a field of a given width.'
    return value.rjust(int(arg))

@register.filter(is_safe=True)
@stringfilter
def center(value, arg):
    if False:
        i = 10
        return i + 15
    'Center the value in a field of a given width.'
    return value.center(int(arg))

@register.filter
@stringfilter
def cut(value, arg):
    if False:
        print('Hello World!')
    'Remove all values of arg from the given string.'
    safe = isinstance(value, SafeData)
    value = value.replace(arg, '')
    if safe and arg != ';':
        return mark_safe(value)
    return value

@register.filter('escape', is_safe=True)
@stringfilter
def escape_filter(value):
    if False:
        print('Hello World!')
    'Mark the value as a string that should be auto-escaped.'
    return conditional_escape(value)

@register.filter(is_safe=True)
def escapeseq(value):
    if False:
        i = 10
        return i + 15
    '\n    An "escape" filter for sequences. Mark each element in the sequence,\n    individually, as a string that should be auto-escaped. Return a list with\n    the results.\n    '
    return [conditional_escape(obj) for obj in value]

@register.filter(is_safe=True)
@stringfilter
def force_escape(value):
    if False:
        return 10
    '\n    Escape a string\'s HTML. Return a new string containing the escaped\n    characters (as opposed to "escape", which marks the content for later\n    possible escaping).\n    '
    return escape(value)

@register.filter('linebreaks', is_safe=True, needs_autoescape=True)
@stringfilter
def linebreaks_filter(value, autoescape=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replace line breaks in plain text with appropriate HTML; a single\n    newline becomes an HTML line break (``<br>``) and a new line\n    followed by a blank line becomes a paragraph break (``</p>``).\n    '
    autoescape = autoescape and (not isinstance(value, SafeData))
    return mark_safe(linebreaks(value, autoescape))

@register.filter(is_safe=True, needs_autoescape=True)
@stringfilter
def linebreaksbr(value, autoescape=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert all newlines in a piece of plain text to HTML line breaks\n    (``<br>``).\n    '
    autoescape = autoescape and (not isinstance(value, SafeData))
    value = normalize_newlines(value)
    if autoescape:
        value = escape(value)
    return mark_safe(value.replace('\n', '<br>'))

@register.filter(is_safe=True)
@stringfilter
def safe(value):
    if False:
        while True:
            i = 10
    'Mark the value as a string that should not be auto-escaped.'
    return mark_safe(value)

@register.filter(is_safe=True)
def safeseq(value):
    if False:
        i = 10
        return i + 15
    '\n    A "safe" filter for sequences. Mark each element in the sequence,\n    individually, as safe, after converting them to strings. Return a list\n    with the results.\n    '
    return [mark_safe(obj) for obj in value]

@register.filter(is_safe=True)
@stringfilter
def striptags(value):
    if False:
        i = 10
        return i + 15
    'Strip all [X]HTML tags.'
    return strip_tags(value)

def _property_resolver(arg):
    if False:
        return 10
    "\n    When arg is convertible to float, behave like operator.itemgetter(arg)\n    Otherwise, chain __getitem__() and getattr().\n\n    >>> _property_resolver(1)('abc')\n    'b'\n    >>> _property_resolver('1')('abc')\n    Traceback (most recent call last):\n    ...\n    TypeError: string indices must be integers\n    >>> class Foo:\n    ...     a = 42\n    ...     b = 3.14\n    ...     c = 'Hey!'\n    >>> _property_resolver('b')(Foo())\n    3.14\n    "
    try:
        float(arg)
    except ValueError:
        if VARIABLE_ATTRIBUTE_SEPARATOR + '_' in arg or arg[0] == '_':
            raise AttributeError('Access to private variables is forbidden.')
        parts = arg.split(VARIABLE_ATTRIBUTE_SEPARATOR)

        def resolve(value):
            if False:
                return 10
            for part in parts:
                try:
                    value = value[part]
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    value = getattr(value, part)
            return value
        return resolve
    else:
        return itemgetter(arg)

@register.filter(is_safe=False)
def dictsort(value, arg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a list of dicts, return that list sorted by the property given in\n    the argument.\n    '
    try:
        return sorted(value, key=_property_resolver(arg))
    except (AttributeError, TypeError):
        return ''

@register.filter(is_safe=False)
def dictsortreversed(value, arg):
    if False:
        return 10
    '\n    Given a list of dicts, return that list sorted in reverse order by the\n    property given in the argument.\n    '
    try:
        return sorted(value, key=_property_resolver(arg), reverse=True)
    except (AttributeError, TypeError):
        return ''

@register.filter(is_safe=False)
def first(value):
    if False:
        return 10
    'Return the first item in a list.'
    try:
        return value[0]
    except IndexError:
        return ''

@register.filter(is_safe=True, needs_autoescape=True)
def join(value, arg, autoescape=True):
    if False:
        print('Hello World!')
    "Join a list with a string, like Python's ``str.join(list)``."
    try:
        if autoescape:
            data = conditional_escape(arg).join([conditional_escape(v) for v in value])
        else:
            data = arg.join(value)
    except TypeError:
        return value
    return mark_safe(data)

@register.filter(is_safe=True)
def last(value):
    if False:
        for i in range(10):
            print('nop')
    'Return the last item in a list.'
    try:
        return value[-1]
    except IndexError:
        return ''

@register.filter(is_safe=False)
def length(value):
    if False:
        return 10
    'Return the length of the value - useful for lists.'
    try:
        return len(value)
    except (ValueError, TypeError):
        return 0

@register.filter(is_safe=True)
def random(value):
    if False:
        for i in range(10):
            print('nop')
    'Return a random item from the list.'
    try:
        return random_module.choice(value)
    except IndexError:
        return ''

@register.filter('slice', is_safe=True)
def slice_filter(value, arg):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a slice of the list using the same syntax as Python's list slicing.\n    "
    try:
        bits = []
        for x in str(arg).split(':'):
            if not x:
                bits.append(None)
            else:
                bits.append(int(x))
        return value[slice(*bits)]
    except (ValueError, TypeError):
        return value

@register.filter(is_safe=True, needs_autoescape=True)
def unordered_list(value, autoescape=True):
    if False:
        return 10
    "\n    Recursively take a self-nested list and return an HTML unordered list --\n    WITHOUT opening and closing <ul> tags.\n\n    Assume the list is in the proper format. For example, if ``var`` contains:\n    ``['States', ['Kansas', ['Lawrence', 'Topeka'], 'Illinois']]``, then\n    ``{{ var|unordered_list }}`` returns::\n\n        <li>States\n        <ul>\n                <li>Kansas\n                <ul>\n                        <li>Lawrence</li>\n                        <li>Topeka</li>\n                </ul>\n                </li>\n                <li>Illinois</li>\n        </ul>\n        </li>\n    "
    if autoescape:
        escaper = conditional_escape
    else:

        def escaper(x):
            if False:
                for i in range(10):
                    print('nop')
            return x

    def walk_items(item_list):
        if False:
            while True:
                i = 10
        item_iterator = iter(item_list)
        try:
            item = next(item_iterator)
            while True:
                try:
                    next_item = next(item_iterator)
                except StopIteration:
                    yield (item, None)
                    break
                if isinstance(next_item, (list, tuple, types.GeneratorType)):
                    try:
                        iter(next_item)
                    except TypeError:
                        pass
                    else:
                        yield (item, next_item)
                        item = next(item_iterator)
                        continue
                yield (item, None)
                item = next_item
        except StopIteration:
            pass

    def list_formatter(item_list, tabs=1):
        if False:
            while True:
                i = 10
        indent = '\t' * tabs
        output = []
        for (item, children) in walk_items(item_list):
            sublist = ''
            if children:
                sublist = '\n%s<ul>\n%s\n%s</ul>\n%s' % (indent, list_formatter(children, tabs + 1), indent, indent)
            output.append('%s<li>%s%s</li>' % (indent, escaper(item), sublist))
        return '\n'.join(output)
    return mark_safe(list_formatter(value))

@register.filter(is_safe=False)
def add(value, arg):
    if False:
        for i in range(10):
            print('nop')
    'Add the arg to the value.'
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        try:
            return value + arg
        except Exception:
            return ''

@register.filter(is_safe=False)
def get_digit(value, arg):
    if False:
        return 10
    '\n    Given a whole number, return the requested digit of it, where 1 is the\n    right-most digit, 2 is the second-right-most digit, etc. Return the\n    original value for invalid input (if input or argument is not an integer,\n    or if argument is less than 1). Otherwise, output is always an integer.\n    '
    try:
        arg = int(arg)
        value = int(value)
    except ValueError:
        return value
    if arg < 1:
        return value
    try:
        return int(str(value)[-arg])
    except IndexError:
        return 0

@register.filter(expects_localtime=True, is_safe=False)
def date(value, arg=None):
    if False:
        for i in range(10):
            print('nop')
    'Format a date according to the given format.'
    if value in (None, ''):
        return ''
    try:
        return formats.date_format(value, arg)
    except AttributeError:
        try:
            return format(value, arg)
        except AttributeError:
            return ''

@register.filter(expects_localtime=True, is_safe=False)
def time(value, arg=None):
    if False:
        return 10
    'Format a time according to the given format.'
    if value in (None, ''):
        return ''
    try:
        return formats.time_format(value, arg)
    except (AttributeError, TypeError):
        try:
            return time_format(value, arg)
        except (AttributeError, TypeError):
            return ''

@register.filter('timesince', is_safe=False)
def timesince_filter(value, arg=None):
    if False:
        return 10
    'Format a date as the time since that date (i.e. "4 days, 6 hours").'
    if not value:
        return ''
    try:
        if arg:
            return timesince(value, arg)
        return timesince(value)
    except (ValueError, TypeError):
        return ''

@register.filter('timeuntil', is_safe=False)
def timeuntil_filter(value, arg=None):
    if False:
        for i in range(10):
            print('nop')
    'Format a date as the time until that date (i.e. "4 days, 6 hours").'
    if not value:
        return ''
    try:
        return timeuntil(value, arg)
    except (ValueError, TypeError):
        return ''

@register.filter(is_safe=False)
def default(value, arg):
    if False:
        for i in range(10):
            print('nop')
    'If value is unavailable, use given default.'
    return value or arg

@register.filter(is_safe=False)
def default_if_none(value, arg):
    if False:
        while True:
            i = 10
    'If value is None, use given default.'
    if value is None:
        return arg
    return value

@register.filter(is_safe=False)
def divisibleby(value, arg):
    if False:
        return 10
    'Return True if the value is divisible by the argument.'
    return int(value) % int(arg) == 0

@register.filter(is_safe=False)
def yesno(value, arg=None):
    if False:
        while True:
            i = 10
    '\n    Given a string mapping values for true, false, and (optionally) None,\n    return one of those strings according to the value:\n\n    ==========  ======================  ==================================\n    Value       Argument                Outputs\n    ==========  ======================  ==================================\n    ``True``    ``"yeah,no,maybe"``     ``yeah``\n    ``False``   ``"yeah,no,maybe"``     ``no``\n    ``None``    ``"yeah,no,maybe"``     ``maybe``\n    ``None``    ``"yeah,no"``           ``"no"`` (converts None to False\n                                        if no mapping for None is given.\n    ==========  ======================  ==================================\n    '
    if arg is None:
        arg = gettext('yes,no,maybe')
    bits = arg.split(',')
    if len(bits) < 2:
        return value
    try:
        (yes, no, maybe) = bits
    except ValueError:
        (yes, no, maybe) = (bits[0], bits[1], bits[1])
    if value is None:
        return maybe
    if value:
        return yes
    return no

@register.filter(is_safe=True)
def filesizeformat(bytes_):
    if False:
        print('Hello World!')
    "\n    Format the value like a 'human-readable' file size (i.e. 13 KB, 4.1 MB,\n    102 bytes, etc.).\n    "
    try:
        bytes_ = int(bytes_)
    except (TypeError, ValueError, UnicodeDecodeError):
        value = ngettext('%(size)d byte', '%(size)d bytes', 0) % {'size': 0}
        return avoid_wrapping(value)

    def filesize_number_format(value):
        if False:
            while True:
                i = 10
        return formats.number_format(round(value, 1), 1)
    KB = 1 << 10
    MB = 1 << 20
    GB = 1 << 30
    TB = 1 << 40
    PB = 1 << 50
    negative = bytes_ < 0
    if negative:
        bytes_ = -bytes_
    if bytes_ < KB:
        value = ngettext('%(size)d byte', '%(size)d bytes', bytes_) % {'size': bytes_}
    elif bytes_ < MB:
        value = gettext('%s KB') % filesize_number_format(bytes_ / KB)
    elif bytes_ < GB:
        value = gettext('%s MB') % filesize_number_format(bytes_ / MB)
    elif bytes_ < TB:
        value = gettext('%s GB') % filesize_number_format(bytes_ / GB)
    elif bytes_ < PB:
        value = gettext('%s TB') % filesize_number_format(bytes_ / TB)
    else:
        value = gettext('%s PB') % filesize_number_format(bytes_ / PB)
    if negative:
        value = '-%s' % value
    return avoid_wrapping(value)

@register.filter(is_safe=False)
def pluralize(value, arg='s'):
    if False:
        print('Hello World!')
    '\n    Return a plural suffix if the value is not 1, \'1\', or an object of\n    length 1. By default, use \'s\' as the suffix:\n\n    * If value is 0, vote{{ value|pluralize }} display "votes".\n    * If value is 1, vote{{ value|pluralize }} display "vote".\n    * If value is 2, vote{{ value|pluralize }} display "votes".\n\n    If an argument is provided, use that string instead:\n\n    * If value is 0, class{{ value|pluralize:"es" }} display "classes".\n    * If value is 1, class{{ value|pluralize:"es" }} display "class".\n    * If value is 2, class{{ value|pluralize:"es" }} display "classes".\n\n    If the provided argument contains a comma, use the text before the comma\n    for the singular case and the text after the comma for the plural case:\n\n    * If value is 0, cand{{ value|pluralize:"y,ies" }} display "candies".\n    * If value is 1, cand{{ value|pluralize:"y,ies" }} display "candy".\n    * If value is 2, cand{{ value|pluralize:"y,ies" }} display "candies".\n    '
    if ',' not in arg:
        arg = ',' + arg
    bits = arg.split(',')
    if len(bits) > 2:
        return ''
    (singular_suffix, plural_suffix) = bits[:2]
    try:
        return singular_suffix if float(value) == 1 else plural_suffix
    except ValueError:
        pass
    except TypeError:
        try:
            return singular_suffix if len(value) == 1 else plural_suffix
        except TypeError:
            pass
    return ''

@register.filter('phone2numeric', is_safe=True)
def phone2numeric_filter(value):
    if False:
        for i in range(10):
            print('nop')
    'Take a phone number and converts it in to its numerical equivalent.'
    return phone2numeric(value)

@register.filter(is_safe=True)
def pprint(value):
    if False:
        print('Hello World!')
    'A wrapper around pprint.pprint -- for debugging, really.'
    try:
        return pformat(value)
    except Exception as e:
        return 'Error in formatting: %s: %s' % (e.__class__.__name__, e)