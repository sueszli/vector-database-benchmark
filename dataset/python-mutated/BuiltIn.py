import difflib
import re
import time
from collections import OrderedDict
from robot.api import logger, SkipExecution
from robot.api.deco import keyword
from robot.errors import BreakLoop, ContinueLoop, DataError, ExecutionFailed, ExecutionFailures, ExecutionPassed, PassExecution, ReturnFromKeyword, VariableError
from robot.running import Keyword, RUN_KW_REGISTER
from robot.running.context import EXECUTION_CONTEXTS
from robot.running.usererrorhandler import UserErrorHandler
from robot.utils import DotDict, escape, format_assign_message, get_error_message, get_time, html_escape, is_falsy, is_integer, is_list_like, is_string, is_truthy, Matcher, normalize, normalize_whitespace, parse_re_flags, parse_time, prepr, plural_or_not as s, RERAISED_EXCEPTIONS, safe_str, secs_to_timestr, seq2str, split_from_equals, timestr_to_secs
from robot.utils.asserts import assert_equal, assert_not_equal
from robot.variables import evaluate_expression, is_dict_variable, is_list_variable, search_variable, DictVariableResolver, VariableResolver
from robot.version import get_version

def run_keyword_variant(resolve, dry_run=False):
    if False:
        return 10

    def decorator(method):
        if False:
            print('Hello World!')
        RUN_KW_REGISTER.register_run_keyword('BuiltIn', method.__name__, resolve, deprecation_warning=False, dry_run=dry_run)
        return method
    return decorator

class _BuiltInBase:

    @property
    def robot_running(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True/False depending on is Robot Framework running or not.\n\n        Can be used by libraries and other extensions.\n\n        New in Robot Framework 6.1.\n        '
        return EXECUTION_CONTEXTS.current is not None

    @property
    def dry_run_active(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True/False depending on is dry-run active or not.\n\n        Can be used by libraries and other extensions. Notice that library\n        keywords are not run at all in dry-run, but library ``__init__``\n        can utilize this information.\n\n        New in Robot Framework 6.1.\n        '
        return self.robot_running and self._context.dry_run

    @property
    def _context(self):
        if False:
            i = 10
            return i + 15
        return self._get_context()

    def _get_context(self, top=False):
        if False:
            print('Hello World!')
        ctx = EXECUTION_CONTEXTS.current if not top else EXECUTION_CONTEXTS.top
        if ctx is None:
            raise RobotNotRunningError('Cannot access execution context')
        return ctx

    @property
    def _namespace(self):
        if False:
            return 10
        return self._get_context().namespace

    @property
    def _variables(self):
        if False:
            for i in range(10):
                print('nop')
        return self._namespace.variables

    def _matches(self, string, pattern, caseless=False):
        if False:
            return 10
        matcher = Matcher(pattern, caseless=caseless, spaceless=False)
        return matcher.match(string)

    def _is_true(self, condition):
        if False:
            return 10
        if is_string(condition):
            condition = self.evaluate(condition)
        return bool(condition)

    def _log_types(self, *args):
        if False:
            print('Hello World!')
        self._log_types_at_level('DEBUG', *args)

    def _log_types_at_level(self, level, *args):
        if False:
            for i in range(10):
                print('nop')
        msg = ['Argument types are:'] + [self._get_type(a) for a in args]
        self.log('\n'.join(msg), level)

    def _get_type(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return str(type(arg))

class _Converter(_BuiltInBase):

    def convert_to_integer(self, item, base=None):
        if False:
            while True:
                i = 10
        'Converts the given item to an integer number.\n\n        If the given item is a string, it is by default expected to be an\n        integer in base 10. There are two ways to convert from other bases:\n\n        - Give base explicitly to the keyword as ``base`` argument.\n\n        - Prefix the given string with the base so that ``0b`` means binary\n          (base 2), ``0o`` means octal (base 8), and ``0x`` means hex (base 16).\n          The prefix is considered only when ``base`` argument is not given and\n          may itself be prefixed with a plus or minus sign.\n\n        The syntax is case-insensitive and possible spaces are ignored.\n\n        Examples:\n        | ${result} = | Convert To Integer | 100    |    | # Result is 100   |\n        | ${result} = | Convert To Integer | FF AA  | 16 | # Result is 65450 |\n        | ${result} = | Convert To Integer | 100    | 8  | # Result is 64    |\n        | ${result} = | Convert To Integer | -100   | 2  | # Result is -4    |\n        | ${result} = | Convert To Integer | 0b100  |    | # Result is 4     |\n        | ${result} = | Convert To Integer | -0x100 |    | # Result is -256  |\n\n        See also `Convert To Number`, `Convert To Binary`, `Convert To Octal`,\n        `Convert To Hex`, and `Convert To Bytes`.\n        '
        self._log_types(item)
        return self._convert_to_integer(item, base)

    def _convert_to_integer(self, orig, base=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            (item, base) = self._get_base(orig, base)
            if base:
                return int(item, self._convert_to_integer(base))
            return int(item)
        except:
            raise RuntimeError(f"'{orig}' cannot be converted to an integer: {get_error_message()}")

    def _get_base(self, item, base):
        if False:
            return 10
        if not is_string(item):
            return (item, base)
        item = normalize(item)
        if item.startswith(('-', '+')):
            sign = item[0]
            item = item[1:]
        else:
            sign = ''
        bases = {'0b': 2, '0o': 8, '0x': 16}
        if base or not item.startswith(tuple(bases)):
            return (sign + item, base)
        return (sign + item[2:], bases[item[:2]])

    def convert_to_binary(self, item, base=None, prefix=None, length=None):
        if False:
            print('Hello World!')
        'Converts the given item to a binary string.\n\n        The ``item``, with an optional ``base``, is first converted to an\n        integer using `Convert To Integer` internally. After that it\n        is converted to a binary number (base 2) represented as a\n        string such as ``1011``.\n\n        The returned value can contain an optional ``prefix`` and can be\n        required to be of minimum ``length`` (excluding the prefix and a\n        possible minus sign). If the value is initially shorter than\n        the required length, it is padded with zeros.\n\n        Examples:\n        | ${result} = | Convert To Binary | 10 |         |           | # Result is 1010   |\n        | ${result} = | Convert To Binary | F  | base=16 | prefix=0b | # Result is 0b1111 |\n        | ${result} = | Convert To Binary | -2 | prefix=B | length=4 | # Result is -B0010 |\n\n        See also `Convert To Integer`, `Convert To Octal` and `Convert To Hex`.\n        '
        return self._convert_to_bin_oct_hex(item, base, prefix, length, 'b')

    def convert_to_octal(self, item, base=None, prefix=None, length=None):
        if False:
            print('Hello World!')
        'Converts the given item to an octal string.\n\n        The ``item``, with an optional ``base``, is first converted to an\n        integer using `Convert To Integer` internally. After that it\n        is converted to an octal number (base 8) represented as a\n        string such as ``775``.\n\n        The returned value can contain an optional ``prefix`` and can be\n        required to be of minimum ``length`` (excluding the prefix and a\n        possible minus sign). If the value is initially shorter than\n        the required length, it is padded with zeros.\n\n        Examples:\n        | ${result} = | Convert To Octal | 10 |            |          | # Result is 12      |\n        | ${result} = | Convert To Octal | -F | base=16    | prefix=0 | # Result is -017    |\n        | ${result} = | Convert To Octal | 16 | prefix=oct | length=4 | # Result is oct0020 |\n\n        See also `Convert To Integer`, `Convert To Binary` and `Convert To Hex`.\n        '
        return self._convert_to_bin_oct_hex(item, base, prefix, length, 'o')

    def convert_to_hex(self, item, base=None, prefix=None, length=None, lowercase=False):
        if False:
            i = 10
            return i + 15
        'Converts the given item to a hexadecimal string.\n\n        The ``item``, with an optional ``base``, is first converted to an\n        integer using `Convert To Integer` internally. After that it\n        is converted to a hexadecimal number (base 16) represented as\n        a string such as ``FF0A``.\n\n        The returned value can contain an optional ``prefix`` and can be\n        required to be of minimum ``length`` (excluding the prefix and a\n        possible minus sign). If the value is initially shorter than\n        the required length, it is padded with zeros.\n\n        By default the value is returned as an upper case string, but the\n        ``lowercase`` argument a true value (see `Boolean arguments`) turns\n        the value (but not the given prefix) to lower case.\n\n        Examples:\n        | ${result} = | Convert To Hex | 255 |           |              | # Result is FF    |\n        | ${result} = | Convert To Hex | -10 | prefix=0x | length=2     | # Result is -0x0A |\n        | ${result} = | Convert To Hex | 255 | prefix=X | lowercase=yes | # Result is Xff   |\n\n        See also `Convert To Integer`, `Convert To Binary` and `Convert To Octal`.\n        '
        spec = 'x' if lowercase else 'X'
        return self._convert_to_bin_oct_hex(item, base, prefix, length, spec)

    def _convert_to_bin_oct_hex(self, item, base, prefix, length, format_spec):
        if False:
            for i in range(10):
                print('nop')
        self._log_types(item)
        ret = format(self._convert_to_integer(item, base), format_spec)
        prefix = prefix or ''
        if ret[0] == '-':
            prefix = '-' + prefix
            ret = ret[1:]
        if length:
            ret = ret.rjust(self._convert_to_integer(length), '0')
        return prefix + ret

    def convert_to_number(self, item, precision=None):
        if False:
            while True:
                i = 10
        "Converts the given item to a floating point number.\n\n        If the optional ``precision`` is positive or zero, the returned number\n        is rounded to that number of decimal digits. Negative precision means\n        that the number is rounded to the closest multiple of 10 to the power\n        of the absolute precision. If a number is equally close to a certain\n        precision, it is always rounded away from zero.\n\n        Examples:\n        | ${result} = | Convert To Number | 42.512 |    | # Result is 42.512 |\n        | ${result} = | Convert To Number | 42.512 | 1  | # Result is 42.5   |\n        | ${result} = | Convert To Number | 42.512 | 0  | # Result is 43.0   |\n        | ${result} = | Convert To Number | 42.512 | -1 | # Result is 40.0   |\n\n        Notice that machines generally cannot store floating point numbers\n        accurately. This may cause surprises with these numbers in general\n        and also when they are rounded. For more information see, for example,\n        these resources:\n\n        - http://docs.python.org/tutorial/floatingpoint.html\n        - http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition\n\n        If you want to avoid possible problems with floating point numbers,\n        you can implement custom keywords using Python's\n        [http://docs.python.org/library/decimal.html|decimal] or\n        [http://docs.python.org/library/fractions.html|fractions] modules.\n\n        If you need an integer number, use `Convert To Integer` instead.\n        "
        self._log_types(item)
        return self._convert_to_number(item, precision)

    def _convert_to_number(self, item, precision=None):
        if False:
            print('Hello World!')
        number = self._convert_to_number_without_precision(item)
        if precision is not None:
            number = float(round(number, self._convert_to_integer(precision)))
        return number

    def _convert_to_number_without_precision(self, item):
        if False:
            i = 10
            return i + 15
        try:
            return float(item)
        except:
            error = get_error_message()
            try:
                return float(self._convert_to_integer(item))
            except RuntimeError:
                raise RuntimeError(f"'{item}' cannot be converted to a floating point number: {error}")

    def convert_to_string(self, item):
        if False:
            print('Hello World!')
        'Converts the given item to a Unicode string.\n\n        Strings are also [https://en.wikipedia.org/wiki/Unicode_equivalence|\n        NFC normalized].\n\n        Use `Encode String To Bytes` and `Decode Bytes To String` keywords\n        in ``String`` library if you need to convert between Unicode and byte\n        strings using different encodings. Use `Convert To Bytes` if you just\n        want to create byte strings.\n        '
        self._log_types(item)
        return safe_str(item)

    def convert_to_boolean(self, item):
        if False:
            return 10
        "Converts the given item to Boolean true or false.\n\n        Handles strings ``True`` and ``False`` (case-insensitive) as expected,\n        otherwise returns item's\n        [http://docs.python.org/library/stdtypes.html#truth|truth value]\n        using Python's ``bool()`` method.\n        "
        self._log_types(item)
        if is_string(item):
            if item.upper() == 'TRUE':
                return True
            if item.upper() == 'FALSE':
                return False
        return bool(item)

    def convert_to_bytes(self, input, input_type='text'):
        if False:
            for i in range(10):
                print('nop')
        'Converts the given ``input`` to bytes according to the ``input_type``.\n\n        Valid input types are listed below:\n\n        - ``text:`` Converts text to bytes character by character. All\n          characters with ordinal below 256 can be used and are converted to\n          bytes with same values. Many characters are easiest to represent\n          using escapes like ``\\x00`` or ``\\xff``. Supports both Unicode\n          strings and bytes.\n\n        - ``int:`` Converts integers separated by spaces to bytes. Similarly as\n          with `Convert To Integer`, it is possible to use binary, octal, or\n          hex values by prefixing the values with ``0b``, ``0o``, or ``0x``,\n          respectively.\n\n        - ``hex:`` Converts hexadecimal values to bytes. Single byte is always\n          two characters long (e.g. ``01`` or ``FF``). Spaces are ignored and\n          can be used freely as a visual separator.\n\n        - ``bin:`` Converts binary values to bytes. Single byte is always eight\n          characters long (e.g. ``00001010``). Spaces are ignored and can be\n          used freely as a visual separator.\n\n        In addition to giving the input as a string, it is possible to use\n        lists or other iterables containing individual characters or numbers.\n        In that case numbers do not need to be padded to certain length and\n        they cannot contain extra spaces.\n\n        Examples (last column shows returned bytes):\n        | ${bytes} = | Convert To Bytes | hyvä      |     | # hyv\\xe4      |\n        | ${bytes} = | Convert To Bytes | hyv\\xe4   |     | # hyv\\xe4      |\n        | ${bytes} = | Convert To Bytes | \\xff\\x07  |     | # \\xff\\x07     |\n        | ${bytes} = | Convert To Bytes | 82 70     | int | # RF           |\n        | ${bytes} = | Convert To Bytes | 0b10 0x10 | int | # \\x02\\x10     |\n        | ${bytes} = | Convert To Bytes | ff 00 07  | hex | # \\xff\\x00\\x07 |\n        | ${bytes} = | Convert To Bytes | 52462121  | hex | # RF!!         |\n        | ${bytes} = | Convert To Bytes | 0000 1000 | bin | # \\x08         |\n        | ${input} = | Create List      | 1         | 2   | 12             |\n        | ${bytes} = | Convert To Bytes | ${input}  | int | # \\x01\\x02\\x0c |\n        | ${bytes} = | Convert To Bytes | ${input}  | hex | # \\x01\\x02\\x12 |\n\n        Use `Encode String To Bytes` in ``String`` library if you need to\n        convert text to bytes using a certain encoding.\n        '
        try:
            try:
                ordinals = getattr(self, f'_get_ordinals_from_{input_type}')
            except AttributeError:
                raise RuntimeError(f"Invalid input type '{input_type}'.")
            return bytes(bytearray((o for o in ordinals(input))))
        except:
            raise RuntimeError('Creating bytes failed: ' + get_error_message())

    def _get_ordinals_from_text(self, input):
        if False:
            while True:
                i = 10
        for char in input:
            ordinal = char if is_integer(char) else ord(char)
            yield self._test_ordinal(ordinal, char, 'Character')

    def _test_ordinal(self, ordinal, original, type):
        if False:
            return 10
        if 0 <= ordinal <= 255:
            return ordinal
        raise RuntimeError(f"{type} '{original}' cannot be represented as a byte.")

    def _get_ordinals_from_int(self, input):
        if False:
            while True:
                i = 10
        if is_string(input):
            input = input.split()
        elif is_integer(input):
            input = [input]
        for integer in input:
            ordinal = self._convert_to_integer(integer)
            yield self._test_ordinal(ordinal, integer, 'Integer')

    def _get_ordinals_from_hex(self, input):
        if False:
            while True:
                i = 10
        for token in self._input_to_tokens(input, length=2):
            ordinal = self._convert_to_integer(token, base=16)
            yield self._test_ordinal(ordinal, token, 'Hex value')

    def _get_ordinals_from_bin(self, input):
        if False:
            while True:
                i = 10
        for token in self._input_to_tokens(input, length=8):
            ordinal = self._convert_to_integer(token, base=2)
            yield self._test_ordinal(ordinal, token, 'Binary value')

    def _input_to_tokens(self, input, length):
        if False:
            i = 10
            return i + 15
        if not is_string(input):
            return input
        input = ''.join(input.split())
        if len(input) % length != 0:
            raise RuntimeError(f'Expected input to be multiple of {length}.')
        return (input[i:i + length] for i in range(0, len(input), length))

    def create_list(self, *items):
        if False:
            while True:
                i = 10
        'Returns a list containing given items.\n\n        The returned list can be assigned both to ``${scalar}`` and ``@{list}``\n        variables.\n\n        Examples:\n        | @{list} =   | Create List | a    | b    | c    |\n        | ${scalar} = | Create List | a    | b    | c    |\n        | ${ints} =   | Create List | ${1} | ${2} | ${3} |\n        '
        return list(items)

    @run_keyword_variant(resolve=0)
    def create_dictionary(self, *items):
        if False:
            i = 10
            return i + 15
        "Creates and returns a dictionary based on the given ``items``.\n\n        Items are typically given using the ``key=value`` syntax same way as\n        ``&{dictionary}`` variables are created in the Variable table. Both\n        keys and values can contain variables, and possible equal sign in key\n        can be escaped with a backslash like ``escaped\\=key=value``. It is\n        also possible to get items from existing dictionaries by simply using\n        them like ``&{dict}``.\n\n        Alternatively items can be specified so that keys and values are given\n        separately. This and the ``key=value`` syntax can even be combined,\n        but separately given items must be first. If same key is used multiple\n        times, the last value has precedence.\n\n        The returned dictionary is ordered, and values with strings as keys\n        can also be accessed using a convenient dot-access syntax like\n        ``${dict.key}``. Technically the returned dictionary is Robot\n        Framework's own ``DotDict`` instance. If there is a need, it can be\n        converted into a regular Python ``dict`` instance by using the\n        `Convert To Dictionary` keyword from the Collections library.\n\n        Examples:\n        | &{dict} = | Create Dictionary | key=value | foo=bar | | | # key=value syntax |\n        | Should Be True | ${dict} == {'key': 'value', 'foo': 'bar'} |\n        | &{dict2} = | Create Dictionary | key | value | foo | bar | # separate key and value |\n        | Should Be Equal | ${dict} | ${dict2} |\n        | &{dict} = | Create Dictionary | ${1}=${2} | &{dict} | foo=new | | # using variables |\n        | Should Be True | ${dict} == {1: 2, 'key': 'value', 'foo': 'new'} |\n        | Should Be Equal | ${dict.key} | value | | | | # dot-access |\n        "
        (separate, combined) = self._split_dict_items(items)
        result = DotDict(self._format_separate_dict_items(separate))
        combined = DictVariableResolver(combined).resolve(self._variables)
        result.update(combined)
        return result

    def _split_dict_items(self, items):
        if False:
            i = 10
            return i + 15
        separate = []
        for item in items:
            (name, value) = split_from_equals(item)
            if value is not None or is_dict_variable(item):
                break
            separate.append(item)
        return (separate, items[len(separate):])

    def _format_separate_dict_items(self, separate):
        if False:
            for i in range(10):
                print('nop')
        separate = self._variables.replace_list(separate)
        if len(separate) % 2 != 0:
            raise DataError(f'Expected even number of keys and values, got {len(separate)}.')
        return [separate[i:i + 2] for i in range(0, len(separate), 2)]

class _Verify(_BuiltInBase):

    def _set_and_remove_tags(self, tags):
        if False:
            i = 10
            return i + 15
        set_tags = [tag for tag in tags if not tag.startswith('-')]
        remove_tags = [tag[1:] for tag in tags if tag.startswith('-')]
        if remove_tags:
            self.remove_tags(*remove_tags)
        if set_tags:
            self.set_tags(*set_tags)

    def fail(self, msg=None, *tags):
        if False:
            while True:
                i = 10
        "Fails the test with the given message and optionally alters its tags.\n\n        The error message is specified using the ``msg`` argument.\n        It is possible to use HTML in the given error message, similarly\n        as with any other keyword accepting an error message, by prefixing\n        the error with ``*HTML*``.\n\n        It is possible to modify tags of the current test case by passing tags\n        after the message. Tags starting with a hyphen (e.g. ``-regression``)\n        are removed and others added. Tags are modified using `Set Tags` and\n        `Remove Tags` internally, and the semantics setting and removing them\n        are the same as with these keywords.\n\n        Examples:\n        | Fail | Test not ready   |             | | # Fails with the given message.    |\n        | Fail | *HTML*<b>Test not ready</b> | | | # Fails using HTML in the message. |\n        | Fail | Test not ready   | not-ready   | | # Fails and adds 'not-ready' tag.  |\n        | Fail | OS not supported | -regression | | # Removes tag 'regression'.        |\n        | Fail | My message       | tag    | -t*  | # Removes all tags starting with 't' except the newly added 'tag'. |\n\n        See `Fatal Error` if you need to stop the whole test execution.\n        "
        self._set_and_remove_tags(tags)
        raise AssertionError(msg) if msg else AssertionError()

    def fatal_error(self, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Stops the whole test execution.\n\n        The test or suite where this keyword is used fails with the provided\n        message, and subsequent tests fail with a canned message.\n        Possible teardowns will nevertheless be executed.\n\n        See `Fail` if you only want to stop one test case unconditionally.\n        '
        error = AssertionError(msg) if msg else AssertionError()
        error.ROBOT_EXIT_ON_FAILURE = True
        raise error

    def should_not_be_true(self, condition, msg=None):
        if False:
            return 10
        'Fails if the given condition is true.\n\n        See `Should Be True` for details about how ``condition`` is evaluated\n        and how ``msg`` can be used to override the default error message.\n        '
        if self._is_true(condition):
            raise AssertionError(msg or f"'{condition}' should not be true.")

    def should_be_true(self, condition, msg=None):
        if False:
            return 10
        "Fails if the given condition is not true.\n\n        If ``condition`` is a string (e.g. ``${rc} < 10``), it is evaluated as\n        a Python expression as explained in `Evaluating expressions` and the\n        keyword status is decided based on the result. If a non-string item is\n        given, the status is got directly from its\n        [http://docs.python.org/library/stdtypes.html#truth|truth value].\n\n        The default error message (``<condition> should be true``) is not very\n        informative, but it can be overridden with the ``msg`` argument.\n\n        Examples:\n        | Should Be True | ${rc} < 10            |\n        | Should Be True | '${status}' == 'PASS' | # Strings must be quoted |\n        | Should Be True | ${number}   | # Passes if ${number} is not zero |\n        | Should Be True | ${list}     | # Passes if ${list} is not empty  |\n\n        Variables used like ``${variable}``, as in the examples above, are\n        replaced in the expression before evaluation. Variables are also\n        available in the evaluation namespace, and can be accessed using\n        special ``$variable`` syntax as explained in the `Evaluating\n        expressions` section.\n\n        Examples:\n        | Should Be True | $rc < 10          |\n        | Should Be True | $status == 'PASS' | # Expected string must be quoted |\n        "
        if not self._is_true(condition):
            raise AssertionError(msg or f"'{condition}' should be true.")

    def should_be_equal(self, first, second, msg=None, values=True, ignore_case=False, formatter='str', strip_spaces=False, collapse_spaces=False):
        if False:
            while True:
                i = 10
        'Fails if the given objects are unequal.\n\n        Optional ``msg``, ``values`` and ``formatter`` arguments specify how\n        to construct the error message if this keyword fails:\n\n        - If ``msg`` is not given, the error message is ``<first> != <second>``.\n        - If ``msg`` is given and ``values`` gets a true value (default),\n          the error message is ``<msg>: <first> != <second>``.\n        - If ``msg`` is given and ``values`` gets a false value (see\n          `Boolean arguments`), the error message is simply ``<msg>``.\n        - ``formatter`` controls how to format the values. Possible values are\n          ``str`` (default), ``repr`` and ``ascii``, and they work similarly\n          as Python built-in functions with same names. See `String\n          representations` for more details.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`) and\n        both arguments are strings, comparison is done case-insensitively.\n        If both arguments are multiline strings, this keyword uses\n        `multiline string comparison`.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        Examples:\n        | Should Be Equal | ${x} | expected |\n        | Should Be Equal | ${x} | expected | Custom error message |\n        | Should Be Equal | ${x} | expected | Custom message | values=False |\n        | Should Be Equal | ${x} | expected | ignore_case=True | formatter=repr |\n\n        ``strip_spaces`` is new in Robot Framework 4.0 and\n        ``collapse_spaces`` is new in Robot Framework 4.1.\n        '
        self._log_types_at_info_if_different(first, second)
        if is_string(first) and is_string(second):
            if ignore_case:
                first = first.casefold()
                second = second.casefold()
            if strip_spaces:
                first = self._strip_spaces(first, strip_spaces)
                second = self._strip_spaces(second, strip_spaces)
            if collapse_spaces:
                first = self._collapse_spaces(first)
                second = self._collapse_spaces(second)
        self._should_be_equal(first, second, msg, values, formatter)

    def _should_be_equal(self, first, second, msg, values, formatter='str'):
        if False:
            print('Hello World!')
        include_values = self._include_values(values)
        formatter = self._get_formatter(formatter)
        if first == second:
            return
        if include_values and is_string(first) and is_string(second):
            self._raise_multi_diff(first, second, msg, formatter)
        assert_equal(first, second, msg, include_values, formatter)

    def _log_types_at_info_if_different(self, first, second):
        if False:
            while True:
                i = 10
        level = 'DEBUG' if type(first) == type(second) else 'INFO'
        self._log_types_at_level(level, first, second)

    def _raise_multi_diff(self, first, second, msg, formatter):
        if False:
            while True:
                i = 10
        first_lines = first.splitlines(True)
        second_lines = second.splitlines(True)
        if len(first_lines) < 3 or len(second_lines) < 3:
            return
        self.log(f'{first.rstrip()}\n\n!=\n\n{second.rstrip()}')
        diffs = list(difflib.unified_diff(first_lines, second_lines, fromfile='first', tofile='second', lineterm=''))
        diffs[3:] = [item[0] + formatter(item[1:]).rstrip() for item in diffs[3:]]
        prefix = 'Multiline strings are different:'
        if msg:
            prefix = f'{msg}: {prefix}'
        raise AssertionError('\n'.join([prefix] + diffs))

    def _include_values(self, values):
        if False:
            print('Hello World!')
        return is_truthy(values) and str(values).upper() != 'NO VALUES'

    def _strip_spaces(self, value, strip_spaces):
        if False:
            return 10
        if not is_string(value):
            return value
        if not is_string(strip_spaces):
            return value.strip() if strip_spaces else value
        if strip_spaces.upper() == 'LEADING':
            return value.lstrip()
        if strip_spaces.upper() == 'TRAILING':
            return value.rstrip()
        return value.strip() if is_truthy(strip_spaces) else value

    def _collapse_spaces(self, value):
        if False:
            for i in range(10):
                print('nop')
        return re.sub('\\s+', ' ', value) if is_string(value) else value

    def should_not_be_equal(self, first, second, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the given objects are equal.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`) and\n        both arguments are strings, comparison is done case-insensitively.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        ``strip_spaces`` is new in Robot Framework 4.0 and ``collapse_spaces`` is new\n        in Robot Framework 4.1.\n        '
        self._log_types_at_info_if_different(first, second)
        if is_string(first) and is_string(second):
            if ignore_case:
                first = first.casefold()
                second = second.casefold()
            if strip_spaces:
                first = self._strip_spaces(first, strip_spaces)
                second = self._strip_spaces(second, strip_spaces)
            if collapse_spaces:
                first = self._collapse_spaces(first)
                second = self._collapse_spaces(second)
        self._should_not_be_equal(first, second, msg, values)

    def _should_not_be_equal(self, first, second, msg, values):
        if False:
            for i in range(10):
                print('nop')
        assert_not_equal(first, second, msg, self._include_values(values))

    def should_not_be_equal_as_integers(self, first, second, msg=None, values=True, base=None):
        if False:
            i = 10
            return i + 15
        'Fails if objects are equal after converting them to integers.\n\n        See `Convert To Integer` for information how to convert integers from\n        other bases than 10 using ``base`` argument or ``0b/0o/0x`` prefixes.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``.\n\n        See `Should Be Equal As Integers` for some usage examples.\n        '
        self._log_types_at_info_if_different(first, second)
        self._should_not_be_equal(self._convert_to_integer(first, base), self._convert_to_integer(second, base), msg, values)

    def should_be_equal_as_integers(self, first, second, msg=None, values=True, base=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if objects are unequal after converting them to integers.\n\n        See `Convert To Integer` for information how to convert integers from\n        other bases than 10 using ``base`` argument or ``0b/0o/0x`` prefixes.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``.\n\n        Examples:\n        | Should Be Equal As Integers | 42   | ${42} | Error message |\n        | Should Be Equal As Integers | ABCD | abcd  | base=16 |\n        | Should Be Equal As Integers | 0b1011 | 11  |\n        '
        self._log_types_at_info_if_different(first, second)
        self._should_be_equal(self._convert_to_integer(first, base), self._convert_to_integer(second, base), msg, values)

    def should_not_be_equal_as_numbers(self, first, second, msg=None, values=True, precision=6):
        if False:
            print('Hello World!')
        'Fails if objects are equal after converting them to real numbers.\n\n        The conversion is done with `Convert To Number` keyword using the\n        given ``precision``.\n\n        See `Should Be Equal As Numbers` for examples on how to use\n        ``precision`` and why it does not always work as expected. See also\n        `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``.\n        '
        self._log_types_at_info_if_different(first, second)
        first = self._convert_to_number(first, precision)
        second = self._convert_to_number(second, precision)
        self._should_not_be_equal(first, second, msg, values)

    def should_be_equal_as_numbers(self, first, second, msg=None, values=True, precision=6):
        if False:
            i = 10
            return i + 15
        "Fails if objects are unequal after converting them to real numbers.\n\n        The conversion is done with `Convert To Number` keyword using the\n        given ``precision``.\n\n        Examples:\n        | Should Be Equal As Numbers | ${x} | 1.1 | | # Passes if ${x} is 1.1 |\n        | Should Be Equal As Numbers | 1.123 | 1.1 | precision=1  | # Passes |\n        | Should Be Equal As Numbers | 1.123 | 1.4 | precision=0  | # Passes |\n        | Should Be Equal As Numbers | 112.3 | 75  | precision=-2 | # Passes |\n\n        As discussed in the documentation of `Convert To Number`, machines\n        generally cannot store floating point numbers accurately. Because of\n        this limitation, comparing floats for equality is problematic and\n        a correct approach to use depends on the context. This keyword uses\n        a very naive approach of rounding the numbers before comparing them,\n        which is both prone to rounding errors and does not work very well if\n        numbers are really big or small. For more information about comparing\n        floats, and ideas on how to implement your own context specific\n        comparison algorithm, see\n        http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/.\n\n        If you want to avoid possible problems with floating point numbers,\n        you can implement custom keywords using Python's\n        [http://docs.python.org/library/decimal.html|decimal] or\n        [http://docs.python.org/library/fractions.html|fractions] modules.\n\n        See `Should Not Be Equal As Numbers` for a negative version of this\n        keyword and `Should Be Equal` for an explanation on how to override\n        the default error message with ``msg`` and ``values``.\n        "
        self._log_types_at_info_if_different(first, second)
        first = self._convert_to_number(first, precision)
        second = self._convert_to_number(second, precision)
        self._should_be_equal(first, second, msg, values)

    def should_not_be_equal_as_strings(self, first, second, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            return 10
        'Fails if objects are equal after converting them to strings.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`),\n        comparison is done case-insensitively.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        Strings are always [https://en.wikipedia.org/wiki/Unicode_equivalence|\n        NFC normalized].\n\n        ``strip_spaces`` is new in Robot Framework 4.0 and ``collapse_spaces`` is new\n        in Robot Framework 4.1.\n        '
        self._log_types_at_info_if_different(first, second)
        first = safe_str(first)
        second = safe_str(second)
        if ignore_case:
            first = first.casefold()
            second = second.casefold()
        if strip_spaces:
            first = self._strip_spaces(first, strip_spaces)
            second = self._strip_spaces(second, strip_spaces)
        if collapse_spaces:
            first = self._collapse_spaces(first)
            second = self._collapse_spaces(second)
        self._should_not_be_equal(first, second, msg, values)

    def should_be_equal_as_strings(self, first, second, msg=None, values=True, ignore_case=False, strip_spaces=False, formatter='str', collapse_spaces=False):
        if False:
            while True:
                i = 10
        'Fails if objects are unequal after converting them to strings.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg``, ``values`` and ``formatter``.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`),\n        comparison is done case-insensitively. If both arguments are\n        multiline strings, this keyword uses `multiline string comparison`.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        Strings are always [https://en.wikipedia.org/wiki/Unicode_equivalence|NFC normalized].\n\n        ``strip_spaces`` is new in Robot Framework 4.0\n        and ``collapse_spaces`` is new in Robot Framework 4.1.\n        '
        self._log_types_at_info_if_different(first, second)
        first = safe_str(first)
        second = safe_str(second)
        if ignore_case:
            first = first.casefold()
            second = second.casefold()
        if strip_spaces:
            first = self._strip_spaces(first, strip_spaces)
            second = self._strip_spaces(second, strip_spaces)
        if collapse_spaces:
            first = self._collapse_spaces(first)
            second = self._collapse_spaces(second)
        self._should_be_equal(first, second, msg, values, formatter)

    def should_not_start_with(self, str1, str2, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            return 10
        'Fails if the string ``str1`` starts with the string ``str2``.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``, as well as for semantics\n        of the ``ignore_case``, ``strip_spaces``, and ``collapse_spaces`` options.\n        '
        if ignore_case:
            str1 = str1.casefold()
            str2 = str2.casefold()
        if strip_spaces:
            str1 = self._strip_spaces(str1, strip_spaces)
            str2 = self._strip_spaces(str2, strip_spaces)
        if collapse_spaces:
            str1 = self._collapse_spaces(str1)
            str2 = self._collapse_spaces(str2)
        if str1.startswith(str2):
            raise AssertionError(self._get_string_msg(str1, str2, msg, values, 'starts with'))

    def should_start_with(self, str1, str2, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the string ``str1`` does not start with the string ``str2``.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``, as well as for semantics\n        of the ``ignore_case``, ``strip_spaces``, and ``collapse_spaces`` options.\n        '
        if ignore_case:
            str1 = str1.casefold()
            str2 = str2.casefold()
        if strip_spaces:
            str1 = self._strip_spaces(str1, strip_spaces)
            str2 = self._strip_spaces(str2, strip_spaces)
        if collapse_spaces:
            str1 = self._collapse_spaces(str1)
            str2 = self._collapse_spaces(str2)
        if not str1.startswith(str2):
            raise AssertionError(self._get_string_msg(str1, str2, msg, values, 'does not start with'))

    def should_not_end_with(self, str1, str2, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            return 10
        'Fails if the string ``str1`` ends with the string ``str2``.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``, as well as for semantics\n        of the ``ignore_case``, ``strip_spaces``, and ``collapse_spaces`` options.\n        '
        if ignore_case:
            str1 = str1.casefold()
            str2 = str2.casefold()
        if strip_spaces:
            str1 = self._strip_spaces(str1, strip_spaces)
            str2 = self._strip_spaces(str2, strip_spaces)
        if collapse_spaces:
            str1 = self._collapse_spaces(str1)
            str2 = self._collapse_spaces(str2)
        if str1.endswith(str2):
            raise AssertionError(self._get_string_msg(str1, str2, msg, values, 'ends with'))

    def should_end_with(self, str1, str2, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            return 10
        'Fails if the string ``str1`` does not end with the string ``str2``.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``, as well as for semantics\n        of the ``ignore_case``, ``strip_spaces``, and ``collapse_spaces`` options.\n        '
        if ignore_case:
            str1 = str1.casefold()
            str2 = str2.casefold()
        if strip_spaces:
            str1 = self._strip_spaces(str1, strip_spaces)
            str2 = self._strip_spaces(str2, strip_spaces)
        if collapse_spaces:
            str1 = self._collapse_spaces(str1)
            str2 = self._collapse_spaces(str2)
        if not str1.endswith(str2):
            raise AssertionError(self._get_string_msg(str1, str2, msg, values, 'does not end with'))

    def should_not_contain(self, container, item, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            i = 10
            return i + 15
        "Fails if ``container`` contains ``item`` one or more times.\n\n        Works with strings, lists, and anything that supports Python's ``in``\n        operator.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with arguments ``msg`` and ``values``. ``ignore_case``\n        has exactly the same semantics as with `Should Contain`.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        Examples:\n        | Should Not Contain | ${some list} | value  |\n        | Should Not Contain | ${output}    | FAILED | ignore_case=True |\n\n        ``strip_spaces`` is new in Robot Framework 4.0 and ``collapse_spaces`` is new\n        in Robot Framework 4.1.\n        "
        orig_container = container
        if ignore_case and is_string(item):
            item = item.casefold()
            if is_string(container):
                container = container.casefold()
            elif is_list_like(container):
                container = set((x.casefold() if is_string(x) else x for x in container))
        if strip_spaces and is_string(item):
            item = self._strip_spaces(item, strip_spaces)
            if is_string(container):
                container = self._strip_spaces(container, strip_spaces)
            elif is_list_like(container):
                container = set((self._strip_spaces(x, strip_spaces) for x in container))
        if collapse_spaces and is_string(item):
            item = self._collapse_spaces(item)
            if is_string(container):
                container = self._collapse_spaces(container)
            elif is_list_like(container):
                container = set((self._collapse_spaces(x) for x in container))
        if item in container:
            raise AssertionError(self._get_string_msg(orig_container, item, msg, values, 'contains'))

    def should_contain(self, container, item, msg=None, values=True, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            while True:
                i = 10
        "Fails if ``container`` does not contain ``item`` one or more times.\n\n        Works with strings, lists, and anything that supports Python's ``in``\n        operator.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with arguments ``msg`` and ``values``.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`) and\n        compared items are strings, it indicates that comparison should be\n        case-insensitive. If the ``container`` is a list-like object, string\n        items in it are compared case-insensitively.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        Examples:\n        | Should Contain | ${output}    | PASS  |\n        | Should Contain | ${some list} | value | msg=Failure! | values=False |\n        | Should Contain | ${some list} | value | ignore_case=True |\n\n        ``strip_spaces`` is new in Robot Framework 4.0 and ``collapse_spaces`` is new\n        in Robot Framework 4.1.\n        "
        orig_container = container
        if ignore_case and is_string(item):
            item = item.casefold()
            if is_string(container):
                container = container.casefold()
            elif is_list_like(container):
                container = set((x.casefold() if is_string(x) else x for x in container))
        if strip_spaces and is_string(item):
            item = self._strip_spaces(item, strip_spaces)
            if is_string(container):
                container = self._strip_spaces(container, strip_spaces)
            elif is_list_like(container):
                container = set((self._strip_spaces(x, strip_spaces) for x in container))
        if collapse_spaces and is_string(item):
            item = self._collapse_spaces(item)
            if is_string(container):
                container = self._collapse_spaces(container)
            elif is_list_like(container):
                container = set((self._collapse_spaces(x) for x in container))
        if item not in container:
            raise AssertionError(self._get_string_msg(orig_container, item, msg, values, 'does not contain'))

    def should_contain_any(self, container, *items, **configuration):
        if False:
            print('Hello World!')
        "Fails if ``container`` does not contain any of the ``*items``.\n\n        Works with strings, lists, and anything that supports Python's ``in``\n        operator.\n\n        Supports additional configuration parameters ``msg``, ``values``,\n        ``ignore_case`` and ``strip_spaces``, and ``collapse_spaces``\n        which have exactly the same semantics as arguments with same\n        names have with `Should Contain`. These arguments must always\n        be given using ``name=value`` syntax after all ``items``.\n\n        Note that possible equal signs in ``items`` must be escaped with\n        a backslash (e.g. ``foo\\=bar``) to avoid them to be passed in\n        as ``**configuration``.\n\n        Examples:\n        | Should Contain Any | ${string} | substring 1 | substring 2 |\n        | Should Contain Any | ${list}   | item 1 | item 2 | item 3 |\n        | Should Contain Any | ${list}   | item 1 | item 2 | item 3 | ignore_case=True |\n        | Should Contain Any | ${list}   | @{items} | msg=Custom message | values=False |\n        "
        msg = configuration.pop('msg', None)
        values = configuration.pop('values', True)
        ignore_case = is_truthy(configuration.pop('ignore_case', False))
        strip_spaces = configuration.pop('strip_spaces', False)
        collapse_spaces = is_truthy(configuration.pop('collapse_spaces', False))
        if configuration:
            raise RuntimeError(f'Unsupported configuration parameter{s(configuration)}: {seq2str(sorted(configuration))}.')
        if not items:
            raise RuntimeError('One or more items required.')
        orig_container = container
        if ignore_case:
            items = [x.casefold() if is_string(x) else x for x in items]
            if is_string(container):
                container = container.casefold()
            elif is_list_like(container):
                container = set((x.casefold() if is_string(x) else x for x in container))
        if strip_spaces:
            items = [self._strip_spaces(x, strip_spaces) for x in items]
            if is_string(container):
                container = self._strip_spaces(container, strip_spaces)
            elif is_list_like(container):
                container = set((self._strip_spaces(x, strip_spaces) for x in container))
        if collapse_spaces:
            items = [self._collapse_spaces(x) for x in items]
            if is_string(container):
                container = self._collapse_spaces(container)
            elif is_list_like(container):
                container = set((self._collapse_spaces(x) for x in container))
        if not any((item in container for item in items)):
            msg = self._get_string_msg(orig_container, seq2str(items, lastsep=' or '), msg, values, 'does not contain any of', quote_item2=False)
            raise AssertionError(msg)

    def should_not_contain_any(self, container, *items, **configuration):
        if False:
            return 10
        "Fails if ``container`` contains one or more of the ``*items``.\n\n        Works with strings, lists, and anything that supports Python's ``in``\n        operator.\n\n        Supports additional configuration parameters ``msg``, ``values``,\n        ``ignore_case`` and ``strip_spaces``, and ``collapse_spaces`` which have exactly\n        the same semantics as arguments with same names have with `Should Contain`.\n        These arguments must always be given using ``name=value`` syntax after all ``items``.\n\n        Note that possible equal signs in ``items`` must be escaped with\n        a backslash (e.g. ``foo\\=bar``) to avoid them to be passed in\n        as ``**configuration``.\n\n        Examples:\n        | Should Not Contain Any | ${string} | substring 1 | substring 2 |\n        | Should Not Contain Any | ${list}   | item 1 | item 2 | item 3 |\n        | Should Not Contain Any | ${list}   | item 1 | item 2 | item 3 | ignore_case=True |\n        | Should Not Contain Any | ${list}   | @{items} | msg=Custom message | values=False |\n        "
        msg = configuration.pop('msg', None)
        values = configuration.pop('values', True)
        ignore_case = is_truthy(configuration.pop('ignore_case', False))
        strip_spaces = configuration.pop('strip_spaces', False)
        collapse_spaces = is_truthy(configuration.pop('collapse_spaces', False))
        if configuration:
            raise RuntimeError(f'Unsupported configuration parameter{s(configuration)}: {seq2str(sorted(configuration))}.')
        if not items:
            raise RuntimeError('One or more items required.')
        orig_container = container
        if ignore_case:
            items = [x.casefold() if is_string(x) else x for x in items]
            if is_string(container):
                container = container.casefold()
            elif is_list_like(container):
                container = set((x.casefold() if is_string(x) else x for x in container))
        if strip_spaces:
            items = [self._strip_spaces(x, strip_spaces) for x in items]
            if is_string(container):
                container = self._strip_spaces(container, strip_spaces)
            elif is_list_like(container):
                container = set((self._strip_spaces(x, strip_spaces) for x in container))
        if collapse_spaces:
            items = [self._collapse_spaces(x) for x in items]
            if is_string(container):
                container = self._collapse_spaces(container)
            elif is_list_like(container):
                container = set((self._collapse_spaces(x) for x in container))
        if any((item in container for item in items)):
            msg = self._get_string_msg(orig_container, seq2str(items, lastsep=' or '), msg, values, 'contains one or more of', quote_item2=False)
            raise AssertionError(msg)

    def should_contain_x_times(self, container, item, count, msg=None, ignore_case=False, strip_spaces=False, collapse_spaces=False):
        if False:
            i = 10
            return i + 15
        'Fails if ``container`` does not contain ``item`` ``count`` times.\n\n        Works with strings, lists and all objects that `Get Count` works\n        with. The default error message can be overridden with ``msg`` and\n        the actual count is always logged.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`) and\n        compared items are strings, it indicates that comparison should be\n        case-insensitive. If the ``container`` is a list-like object, string\n        items in it are compared case-insensitively.\n\n        If ``strip_spaces`` is given a true value (see `Boolean arguments`)\n        and both arguments are strings, the comparison is done without leading\n        and trailing spaces. If ``strip_spaces`` is given a string value\n        ``LEADING`` or ``TRAILING`` (case-insensitive), the comparison is done\n        without leading or trailing spaces, respectively.\n\n        If ``collapse_spaces`` is given a true value (see `Boolean arguments`) and both\n        arguments are strings, the comparison is done with all white spaces replaced by\n        a single space character.\n\n        Examples:\n        | Should Contain X Times | ${output}    | hello | 2 |\n        | Should Contain X Times | ${some list} | value | 3 | ignore_case=True |\n\n        ``strip_spaces`` is new in Robot Framework 4.0 and ``collapse_spaces`` is new\n        in Robot Framework 4.1.\n        '
        count = self._convert_to_integer(count)
        orig_container = container
        if is_string(item):
            if ignore_case:
                item = item.casefold()
                if is_string(container):
                    container = container.casefold()
                elif is_list_like(container):
                    container = [x.casefold() if is_string(x) else x for x in container]
            if strip_spaces:
                item = self._strip_spaces(item, strip_spaces)
                if is_string(container):
                    container = self._strip_spaces(container, strip_spaces)
                elif is_list_like(container):
                    container = [self._strip_spaces(x, strip_spaces) for x in container]
            if collapse_spaces:
                item = self._collapse_spaces(item)
                if is_string(container):
                    container = self._collapse_spaces(container)
                elif is_list_like(container):
                    container = [self._collapse_spaces(x) for x in container]
        x = self.get_count(container, item)
        if not msg:
            msg = f"{orig_container!r} contains '{item}' {x} time{s(x)}, not {count} time{s(count)}."
        self.should_be_equal_as_integers(x, count, msg, values=False)

    def get_count(self, container, item):
        if False:
            for i in range(10):
                print('nop')
        'Returns and logs how many times ``item`` is found from ``container``.\n\n        This keyword works with Python strings and lists and all objects\n        that either have ``count`` method or can be converted to Python lists.\n\n        Example:\n        | ${count} = | Get Count | ${some item} | interesting value |\n        | Should Be True | 5 < ${count} < 10 |\n        '
        if not hasattr(container, 'count'):
            try:
                container = list(container)
            except:
                raise RuntimeError(f"Converting '{container}' to list failed: {get_error_message()}")
        count = container.count(item)
        self.log(f'Item found from container {count} time{s(count)}.')
        return count

    def should_not_match(self, string, pattern, msg=None, values=True, ignore_case=False):
        if False:
            return 10
        'Fails if the given ``string`` matches the given ``pattern``.\n\n        Pattern matching is similar as matching files in a shell with\n        ``*``, ``?`` and ``[chars]`` acting as wildcards. See the\n        `Glob patterns` section for more information.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`),\n        the comparison is case-insensitive.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values`.\n        '
        if self._matches(string, pattern, caseless=ignore_case):
            raise AssertionError(self._get_string_msg(string, pattern, msg, values, 'matches'))

    def should_match(self, string, pattern, msg=None, values=True, ignore_case=False):
        if False:
            return 10
        'Fails if the given ``string`` does not match the given ``pattern``.\n\n        Pattern matching is similar as matching files in a shell with\n        ``*``, ``?`` and ``[chars]`` acting as wildcards. See the\n        `Glob patterns` section for more information.\n\n        If ``ignore_case`` is given a true value (see `Boolean arguments`) and\n        compared items are strings, it indicates that comparison should be\n        case-insensitive.\n\n        See `Should Be Equal` for an explanation on how to override the default\n        error message with ``msg`` and ``values``.\n        '
        if not self._matches(string, pattern, caseless=ignore_case):
            raise AssertionError(self._get_string_msg(string, pattern, msg, values, 'does not match'))

    def should_match_regexp(self, string, pattern, msg=None, values=True, flags=None):
        if False:
            return 10
        "Fails if ``string`` does not match ``pattern`` as a regular expression.\n\n        See the `Regular expressions` section for more information about\n        regular expressions and how to use then in Robot Framework test data.\n\n        Notice that the given pattern does not need to match the whole string.\n        For example, the pattern ``ello`` matches the string ``Hello world!``.\n        If a full match is needed, the ``^`` and ``$`` characters can be used\n        to denote the beginning and end of the string, respectively.\n        For example, ``^ello$`` only matches the exact string ``ello``.\n\n        Possible flags altering how the expression is parsed (e.g. ``re.IGNORECASE``,\n        ``re.MULTILINE``) can be given using the ``flags`` argument (e.g.\n        ``flags=IGNORECASE | MULTILINE``) or embedded to the pattern (e.g.\n        ``(?im)pattern``).\n\n        If this keyword passes, it returns the portion of the string that\n        matched the pattern. Additionally, the possible captured groups are\n        returned.\n\n        See the `Should Be Equal` keyword for an explanation on how to override\n        the default error message with the ``msg`` and ``values`` arguments.\n\n        Examples:\n        | Should Match Regexp | ${output} | \\\\d{6}   | # Output contains six numbers  |\n        | Should Match Regexp | ${output} | ^\\\\d{6}$ | # Six numbers and nothing more |\n        | ${ret} = | Should Match Regexp | Foo: 42 | foo: \\\\d+ | flags=IGNORECASE |\n        | ${ret} = | Should Match Regexp | Foo: 42 | (?i)foo: \\\\d+ |\n        | ${match} | ${group1} | ${group2} = |\n        | ...      | Should Match Regexp | Bar: 43 | (Foo|Bar): (\\\\d+) |\n        =>\n        | ${ret} = 'Foo: 42'\n        | ${match} = 'Bar: 43'\n        | ${group1} = 'Bar'\n        | ${group2} = '43'\n\n        The ``flags`` argument is new in Robot Framework 6.0.\n        "
        res = re.search(pattern, string, flags=parse_re_flags(flags))
        if res is None:
            raise AssertionError(self._get_string_msg(string, pattern, msg, values, 'does not match'))
        match = res.group(0)
        groups = res.groups()
        if groups:
            return [match] + list(groups)
        return match

    def should_not_match_regexp(self, string, pattern, msg=None, values=True, flags=None):
        if False:
            print('Hello World!')
        'Fails if ``string`` matches ``pattern`` as a regular expression.\n\n        See `Should Match Regexp` for more information about arguments.\n        '
        if re.search(pattern, string, flags=parse_re_flags(flags)) is not None:
            raise AssertionError(self._get_string_msg(string, pattern, msg, values, 'matches'))

    def get_length(self, item):
        if False:
            i = 10
            return i + 15
        "Returns and logs the length of the given item as an integer.\n\n        The item can be anything that has a length, for example, a string,\n        a list, or a mapping. The keyword first tries to get the length with\n        the Python function ``len``, which calls the  item's ``__len__`` method\n        internally. If that fails, the keyword tries to call the item's\n        possible ``length`` and ``size`` methods directly. The final attempt is\n        trying to get the value of the item's ``length`` attribute. If all\n        these attempts are unsuccessful, the keyword fails.\n\n        Examples:\n        | ${length} = | Get Length    | Hello, world! |        |\n        | Should Be Equal As Integers | ${length}     | 13     |\n        | @{list} =   | Create List   | Hello,        | world! |\n        | ${length} = | Get Length    | ${list}       |        |\n        | Should Be Equal As Integers | ${length}     | 2      |\n\n        See also `Length Should Be`, `Should Be Empty` and `Should Not Be\n        Empty`.\n        "
        length = self._get_length(item)
        self.log(f'Length is {length}.')
        return length

    def _get_length(self, item):
        if False:
            for i in range(10):
                print('nop')
        try:
            return len(item)
        except RERAISED_EXCEPTIONS:
            raise
        except:
            try:
                return item.length()
            except RERAISED_EXCEPTIONS:
                raise
            except:
                try:
                    return item.size()
                except RERAISED_EXCEPTIONS:
                    raise
                except:
                    try:
                        return item.length
                    except RERAISED_EXCEPTIONS:
                        raise
                    except:
                        raise RuntimeError(f"Could not get length of '{item}'.")

    def length_should_be(self, item, length, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Verifies that the length of the given item is correct.\n\n        The length of the item is got using the `Get Length` keyword. The\n        default error message can be overridden with the ``msg`` argument.\n        '
        length = self._convert_to_integer(length)
        actual = self.get_length(item)
        if actual != length:
            raise AssertionError(msg or f"Length of '{item}' should be {length} but is {actual}.")

    def should_be_empty(self, item, msg=None):
        if False:
            print('Hello World!')
        'Verifies that the given item is empty.\n\n        The length of the item is got using the `Get Length` keyword. The\n        default error message can be overridden with the ``msg`` argument.\n        '
        if self.get_length(item) > 0:
            raise AssertionError(msg or f"'{item}' should be empty.")

    def should_not_be_empty(self, item, msg=None):
        if False:
            while True:
                i = 10
        'Verifies that the given item is not empty.\n\n        The length of the item is got using the `Get Length` keyword. The\n        default error message can be overridden with the ``msg`` argument.\n        '
        if self.get_length(item) == 0:
            raise AssertionError(msg or f"'{item}' should not be empty.")

    def _get_string_msg(self, item1, item2, custom_message, include_values, delimiter, quote_item1=True, quote_item2=True):
        if False:
            i = 10
            return i + 15
        if custom_message and (not self._include_values(include_values)):
            return custom_message
        item1 = f"'{safe_str(item1)}'" if quote_item1 else safe_str(item1)
        item2 = f"'{safe_str(item2)}'" if quote_item2 else safe_str(item2)
        default_message = f'{item1} {delimiter} {item2}'
        if not custom_message:
            return default_message
        return f'{custom_message}: {default_message}'

class _Variables(_BuiltInBase):

    def get_variables(self, no_decoration=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns a dictionary containing all variables in the current scope.\n\n        Variables are returned as a special dictionary that allows accessing\n        variables in space, case, and underscore insensitive manner similarly\n        as accessing variables in the test data. This dictionary supports all\n        same operations as normal Python dictionaries and, for example,\n        Collections library can be used to access or modify it. Modifying the\n        returned dictionary has no effect on the variables available in the\n        current scope.\n\n        By default variables are returned with ``${}``, ``@{}`` or ``&{}``\n        decoration based on variable types. Giving a true value (see `Boolean\n        arguments`) to the optional argument ``no_decoration`` will return\n        the variables without the decoration.\n\n        Example:\n        | ${example_variable} =         | Set Variable | example value         |\n        | ${variables} =                | Get Variables |                      |\n        | Dictionary Should Contain Key | ${variables} | \\${example_variable} |\n        | Dictionary Should Contain Key | ${variables} | \\${ExampleVariable}  |\n        | Set To Dictionary             | ${variables} | \\${name} | value     |\n        | Variable Should Not Exist     | \\${name}    |           |           |\n        | ${no decoration} =            | Get Variables | no_decoration=Yes |\n        | Dictionary Should Contain Key | ${no decoration} | example_variable |\n        '
        return self._variables.as_dict(decoration=is_falsy(no_decoration))

    @keyword(types=None)
    @run_keyword_variant(resolve=0)
    def get_variable_value(self, name, default=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns variable value or ``default`` if the variable does not exist.\n\n        The name of the variable can be given either as a normal variable name\n        like ``${name}`` or in escaped format like ``$name`` or ``\\${name}``.\n        For the reasons explained in the `Using variables with keywords creating\n        or accessing variables` section, using the escaped format is recommended.\n\n        Examples:\n        | ${x} =    `Get Variable Value`    $a    default\n        | ${y} =    `Get Variable Value`    $a    ${b}\n        | ${z} =    `Get Variable Value`    $z\n        =>\n        - ``${x}`` gets value of ``${a}`` if ``${a}`` exists and string ``default`` otherwise\n        - ``${y}`` gets value of ``${a}`` if ``${a}`` exists and value of ``${b}`` otherwise\n        - ``${z}`` is set to Python ``None`` if it does not exist previously\n        '
        try:
            name = self._get_var_name(name, require_assign=False)
            return self._variables.replace_scalar(name)
        except VariableError:
            return self._variables.replace_scalar(default)

    def log_variables(self, level='INFO'):
        if False:
            i = 10
            return i + 15
        'Logs all variables in the current scope with given log level.'
        variables = self.get_variables()
        for name in sorted(variables, key=lambda s: s[2:-1].casefold()):
            (name, value) = self._get_logged_variable(name, variables)
            msg = format_assign_message(name, value, cut_long=False)
            self.log(msg, level)

    def _get_logged_variable(self, name, variables):
        if False:
            i = 10
            return i + 15
        value = variables[name]
        try:
            if name[0] == '@':
                value = list(value)
            if name[0] == '&':
                value = OrderedDict(value)
        except RERAISED_EXCEPTIONS:
            raise
        except:
            name = '$' + name[1:]
        return (name, value)

    @run_keyword_variant(resolve=0)
    def variable_should_exist(self, name, msg=None):
        if False:
            return 10
        'Fails unless the given variable exists within the current scope.\n\n        The name of the variable can be given either as a normal variable name\n        like ``${name}`` or in escaped format like ``$name`` or ``\\${name}``.\n        For the reasons explained in the `Using variables with keywords creating\n        or accessing variables` section, using the escaped format is recommended.\n\n        The default error message can be overridden with the ``msg`` argument.\n\n        See also `Variable Should Not Exist` and `Keyword Should Exist`.\n        '
        name = self._get_var_name(name)
        try:
            self._variables.replace_scalar(name)
        except VariableError:
            raise AssertionError(self._variables.replace_string(msg) if msg else f"Variable '{name}' does not exist.")

    @run_keyword_variant(resolve=0)
    def variable_should_not_exist(self, name, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails if the given variable exists within the current scope.\n\n        The name of the variable can be given either as a normal variable name\n        like ``${name}`` or in escaped format like ``$name`` or ``\\${name}``.\n        For the reasons explained in the `Using variables with keywords creating\n        or accessing variables` section, using the escaped format is recommended.\n\n        The default error message can be overridden with the ``msg`` argument.\n\n        See also `Variable Should Exist` and `Keyword Should Exist`.\n        '
        name = self._get_var_name(name)
        try:
            self._variables.replace_scalar(name)
        except VariableError:
            pass
        else:
            raise AssertionError(self._variables.replace_string(msg) if msg else f"Variable '{name}' exists.")

    def replace_variables(self, text):
        if False:
            i = 10
            return i + 15
        'Replaces variables in the given text with their current values.\n\n        If the text contains undefined variables, this keyword fails.\n        If the given ``text`` contains only a single variable, its value is\n        returned as-is and it can be any object. Otherwise, this keyword\n        always returns a string.\n\n        Example:\n\n        The file ``template.txt`` contains ``Hello ${NAME}!`` and variable\n        ``${NAME}`` has the value ``Robot``.\n\n        | ${template} =   | Get File          | ${CURDIR}/template.txt |\n        | ${message} =    | Replace Variables | ${template}            |\n        | Should Be Equal | ${message}        | Hello Robot!           |\n        '
        return self._variables.replace_scalar(text)

    def set_variable(self, *values):
        if False:
            i = 10
            return i + 15
        'Returns the given values which can then be assigned to a variables.\n\n        This keyword is mainly used for setting scalar variables.\n        Additionally it can be used for converting a scalar variable\n        containing a list to a list variable or to multiple scalar variables.\n        It is recommended to use `Create List` when creating new lists.\n\n        Examples:\n        | ${hi} =    Set Variable    Hello, world!\n        | ${hi2} =    Set Variable    I said: ${hi}\n        | ${var1}    ${var2} =    Set Variable    Hello    world\n        | @{list} =    Set Variable    ${list with some items}\n        | ${item1}    ${item2} =    Set Variable    ${list with 2 items}\n\n        Variables created with this keyword are available only in the\n        scope where they are created. See `Set Global Variable`,\n        `Set Test Variable` and `Set Suite Variable` for information on how to\n        set variables so that they are available also in a larger scope.\n\n        *NOTE:* The ``VAR`` syntax introduced in Robot Framework 7.0 is generally\n        recommended over this keyword. The basic usage is shown below and the Robot\n        Framework User Guide explains the syntax in detail.\n\n        | VAR    ${hi}     Hello, world!\n        | VAR    ${hi2}    I said: ${hi}\n        '
        if len(values) == 0:
            return ''
        elif len(values) == 1:
            return values[0]
        else:
            return list(values)

    @run_keyword_variant(resolve=0)
    def set_local_variable(self, name, *values):
        if False:
            while True:
                i = 10
        'Makes a variable available everywhere within the local scope.\n\n        Variables set with this keyword are available within the\n        local scope of the currently executed test case or in the local scope\n        of the keyword in which they are defined. For example, if you set a\n        variable in a user keyword, it is available only in that keyword. Other\n        test cases or keywords will not see variables set with this keyword.\n\n        This keyword is equivalent to a normal variable assignment based on a\n        keyword return value. For example,\n\n        | ${var} =    `Set Variable`    value\n        | @{list} =    `Create List`    item1    item2    item3\n\n        are equivalent with\n\n        | `Set Local Variable`    @var    value\n        | `Set Local Variable`    @list    item1    item2    item3\n\n        The main use case for this keyword is creating local variables in\n        libraries.\n\n        See `Set Suite Variable` for more information and usage examples. See\n        also the `Using variables with keywords creating or accessing variables`\n        section for information why it is recommended to give the variable name\n        in escaped format like ``$name`` or ``\\${name}`` instead of the normal\n        ``${name}``.\n\n        See also `Set Global Variable` and `Set Test Variable`.\n\n        *NOTE:* The ``VAR`` syntax introduced in Robot Framework 7.0 is recommended\n        over this keyword.\n        '
        name = self._get_var_name(name)
        value = self._get_var_value(name, values)
        self._variables.set_local(name, value)
        self._log_set_variable(name, value)

    @run_keyword_variant(resolve=0)
    def set_test_variable(self, name, *values):
        if False:
            print('Hello World!')
        'Makes a variable available everywhere within the scope of the current test.\n\n        Variables set with this keyword are available everywhere within the\n        scope of the currently executed test case. For example, if you set a\n        variable in a user keyword, it is available both in the test case level\n        and also in all other user keywords used in the current test. Other\n        test cases will not see variables set with this keyword.\n        It is an error to call `Set Test Variable` outside the\n        scope of a test (e.g. in a Suite Setup or Teardown).\n\n        See `Set Suite Variable` for more information and usage examples. See\n        also the `Using variables with keywords creating or accessing variables`\n        section for information why it is recommended to give the variable name\n        in escaped format like ``$name`` or ``\\${name}`` instead of the normal\n        ``${name}``.\n\n        When creating automated tasks, not tests, it is possible to use `Set\n        Task Variable`. See also `Set Global Variable` and `Set Local Variable`.\n\n        *NOTE:* The ``VAR`` syntax introduced in Robot Framework 7.0 is recommended\n        over this keyword.\n        '
        name = self._get_var_name(name)
        value = self._get_var_value(name, values)
        self._variables.set_test(name, value)
        self._log_set_variable(name, value)

    @run_keyword_variant(resolve=0)
    def set_task_variable(self, name, *values):
        if False:
            print('Hello World!')
        'Makes a variable available everywhere within the scope of the current task.\n\n        This is an alias for `Set Test Variable` that is more applicable when\n        creating tasks, not tests.\n\n        *NOTE:* The ``VAR`` syntax introduced in Robot Framework 7.0 is recommended\n        over this keyword.\n        '
        self.set_test_variable(name, *values)

    @run_keyword_variant(resolve=0)
    def set_suite_variable(self, name, *values):
        if False:
            return 10
        'Makes a variable available everywhere within the scope of the current suite.\n\n        Variables set with this keyword are available everywhere within the\n        scope of the currently executed test suite. Setting variables with this\n        keyword thus has the same effect as creating them using the Variables\n        section in the data file or importing them from variable files.\n\n        Possible child test suites do not see variables set with this keyword\n        by default, but that can be controlled by using ``children=<option>``\n        as the last argument. If the specified ``<option>`` is given a true value\n        (see `Boolean arguments`), the variable is set also to the child\n        suites. Parent and sibling suites will never see variables set with\n        this keyword.\n\n        The name of the variable can be given either as a normal variable name\n        like ``${NAME}`` or in escaped format as ``\\${NAME}`` or ``$NAME``.\n        For the reasons explained in the `Using variables with keywords creating\n        or accessing variables` section, *using the escaped format is highly\n        recommended*.\n\n        Variable value can be specified using the same syntax as when variables\n        are created in the Variables section. Same way as in that section,\n        it is possible to create scalar values, lists and dictionaries.\n        The type is got from the variable name prefix ``$``, ``@`` and ``&``,\n        respectively.\n\n        If a variable already exists within the new scope, its value will be\n        overwritten. If a variable already exists within the current scope,\n        the value can be left empty and the variable within the new scope gets\n        the value within the current scope.\n\n        Examples:\n        | Set Suite Variable    $SCALAR    Hello, world!\n        | Set Suite Variable    $SCALAR    Hello, world!    children=True\n        | Set Suite Variable    @LIST      First item       Second item\n        | Set Suite Variable    &DICT      key=value        foo=bar\n        | ${ID} =    Get ID\n        | Set Suite Variable    $ID\n\n        To override an existing value with an empty value, use built-in\n        variables ``${EMPTY}``, ``@{EMPTY}`` or ``&{EMPTY}``:\n\n        | Set Suite Variable    $SCALAR    ${EMPTY}\n        | Set Suite Variable    @LIST      @{EMPTY}\n        | Set Suite Variable    &DICT      &{EMPTY}\n\n        See also `Set Global Variable`, `Set Test Variable` and `Set Local Variable`.\n\n        *NOTE:* The ``VAR`` syntax introduced in Robot Framework 7.0 is recommended\n        over this keyword. The basic usage is shown below and the Robot Framework\n        User Guide explains the syntax in detail.\n\n        | VAR    ${SCALAR}    Hello, world!                scope=SUITE\n        | VAR    @{LIST}      First item    Second item    scope=SUITE\n        | VAR    &{DICT}      key=value     foo=bar        scope=SUITE\n        '
        name = self._get_var_name(name)
        if values and is_string(values[-1]) and values[-1].startswith('children='):
            children = self._variables.replace_scalar(values[-1][9:])
            children = is_truthy(children)
            values = values[:-1]
        else:
            children = False
        value = self._get_var_value(name, values)
        self._variables.set_suite(name, value, children=children)
        self._log_set_variable(name, value)

    @run_keyword_variant(resolve=0)
    def set_global_variable(self, name, *values):
        if False:
            i = 10
            return i + 15
        'Makes a variable available globally in all tests and suites.\n\n        Variables set with this keyword are globally available in all\n        subsequent test suites, test cases and user keywords. Also variables\n        created Variables sections are overridden. Variables assigned locally\n        based on keyword return values or by using `Set Suite Variable`,\n        `Set Test Variable` or `Set Local Variable` override these variables\n        in that scope, but the global value is not changed in those cases.\n\n        In practice setting variables with this keyword has the same effect\n        as using command line options ``--variable`` and ``--variablefile``.\n        Because this keyword can change variables everywhere, it should be\n        used with care.\n\n        See `Set Suite Variable` for more information and usage examples. See\n        also the `Using variables with keywords creating or accessing variables`\n        section for information why it is recommended to give the variable name\n        in escaped format like ``$name`` or ``\\${name}`` instead of the normal\n        ``${name}``.\n\n        *NOTE:* The ``VAR`` syntax introduced in Robot Framework 7.0 is recommended\n        over this keyword.\n        '
        name = self._get_var_name(name)
        value = self._get_var_value(name, values)
        self._variables.set_global(name, value)
        self._log_set_variable(name, value)

    def _get_var_name(self, original, require_assign=True):
        if False:
            for i in range(10):
                print('nop')
        try:
            replaced = self._variables.replace_string(original)
        except VariableError:
            replaced = original
        try:
            name = self._resolve_var_name(replaced)
        except ValueError:
            name = original
        match = search_variable(name, identifiers='$@&')
        match.resolve_base(self._variables)
        valid = match.is_assign() if require_assign else match.is_variable()
        if not valid:
            raise DataError(f"Invalid variable name '{name}'.")
        return str(match)

    def _resolve_var_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name.startswith('\\'):
            name = name[1:]
        if len(name) < 2 or name[0] not in '$@&':
            raise ValueError
        if name[1] != '{':
            name = f'{name[0]}{{{name[1:]}}}'
        match = search_variable(name, identifiers='$@&', ignore_errors=True)
        match.resolve_base(self._variables)
        if not match.is_assign():
            raise ValueError
        return str(match)

    def _get_var_value(self, name, values):
        if False:
            while True:
                i = 10
        if not values:
            return self._variables[name]
        if name[0] == '$':
            if len(values) != 1 or is_list_variable(values[0]):
                raise DataError(f"Setting list value to scalar variable '{name}' is not supported anymore. Create list variable '@{name[1:]}' instead.")
            return self._variables.replace_scalar(values[0])
        resolver = VariableResolver.from_name_and_value(name, values)
        return resolver.resolve(self._variables)

    def _log_set_variable(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        self.log(format_assign_message(name, value))

class _RunKeyword(_BuiltInBase):

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword(self, name, *args):
        if False:
            print('Hello World!')
        'Executes the given keyword with the given arguments.\n\n        Because the name of the keyword to execute is given as an argument, it\n        can be a variable and thus set dynamically, e.g. from a return value of\n        another keyword or from the command line.\n        '
        if not is_string(name):
            raise RuntimeError('Keyword name must be a string.')
        ctx = self._context
        if not (ctx.dry_run or self._accepts_embedded_arguments(name, ctx)):
            (name, args) = self._replace_variables_in_name([name] + list(args))
        parent = ctx.steps[-1][0] if ctx.steps else ctx.test or ctx.suite
        kw = Keyword(name, args=args, parent=parent, lineno=getattr(parent, 'lineno', None))
        return kw.run(ctx)

    def _accepts_embedded_arguments(self, name, ctx):
        if False:
            print('Hello World!')
        if '{' in name:
            runner = ctx.get_runner(name, recommend_on_failure=False)
            return runner and hasattr(runner, 'embedded_args')
        return False

    def _replace_variables_in_name(self, name_and_args):
        if False:
            for i in range(10):
                print('nop')
        resolved = self._variables.replace_list(name_and_args, replace_until=1, ignore_errors=self._context.in_teardown)
        if not resolved:
            raise DataError(f'Keyword name missing: Given arguments {name_and_args} resolved to an empty list.')
        if not is_string(resolved[0]):
            raise RuntimeError('Keyword name must be a string.')
        return (resolved[0], resolved[1:])

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keywords(self, *keywords):
        if False:
            print('Hello World!')
        'Executes all the given keywords in a sequence.\n\n        This keyword is mainly useful in setups and teardowns when they need\n        to take care of multiple actions and creating a new higher level user\n        keyword would be an overkill.\n\n        By default all arguments are expected to be keywords to be executed.\n\n        Examples:\n        | `Run Keywords` | `Initialize database` | `Start servers` | `Clear logs` |\n        | `Run Keywords` | ${KW 1} | ${KW 2} |\n        | `Run Keywords` | @{KEYWORDS} |\n\n        Keywords can also be run with arguments using upper case ``AND`` as\n        a separator between keywords. The keywords are executed so that the\n        first argument is the first keyword and proceeding arguments until\n        the first ``AND`` are arguments to it. First argument after the first\n        ``AND`` is the second keyword and proceeding arguments until the next\n        ``AND`` are its arguments. And so on.\n\n        Examples:\n        | `Run Keywords` | `Initialize database` | db1 | AND | `Start servers` | server1 | server2 |\n        | `Run Keywords` | `Initialize database` | ${DB NAME} | AND | `Start servers` | @{SERVERS} | AND | `Clear logs` |\n        | `Run Keywords` | ${KW} | AND | @{KW WITH ARGS} |\n\n        Notice that the ``AND`` control argument must be used explicitly and\n        cannot itself come from a variable. If you need to use literal ``AND``\n        string as argument, you can either use variables or escape it with\n        a backslash like ``\\AND``.\n        '
        self._run_keywords(self._split_run_keywords(list(keywords)))

    def _run_keywords(self, iterable):
        if False:
            print('Hello World!')
        errors = []
        for (kw, args) in iterable:
            try:
                self.run_keyword(kw, *args)
            except ExecutionPassed as err:
                err.set_earlier_failures(errors)
                raise err
            except ExecutionFailed as err:
                errors.extend(err.get_errors())
                if not err.can_continue(self._context):
                    break
        if errors:
            raise ExecutionFailures(errors)

    def _split_run_keywords(self, keywords):
        if False:
            while True:
                i = 10
        if 'AND' not in keywords:
            for name in self._split_run_keywords_without_and(keywords):
                yield (name, ())
        else:
            for kw_call in self._split_run_keywords_with_and(keywords):
                if not kw_call:
                    raise DataError('AND must have keyword before and after.')
                yield (kw_call[0], kw_call[1:])

    def _split_run_keywords_without_and(self, keywords):
        if False:
            while True:
                i = 10
        replace_list = self._variables.replace_list
        ignore_errors = self._context.in_teardown
        for name in keywords:
            if is_list_variable(name):
                for n in replace_list([name], ignore_errors=ignore_errors):
                    yield escape(n)
            else:
                yield name

    def _split_run_keywords_with_and(self, keywords):
        if False:
            print('Hello World!')
        while 'AND' in keywords:
            index = keywords.index('AND')
            yield keywords[:index]
            keywords = keywords[index + 1:]
        yield keywords

    @run_keyword_variant(resolve=1, dry_run=True)
    def run_keyword_if(self, condition, name, *args):
        if False:
            for i in range(10):
                print('nop')
        "Runs the given keyword with the given arguments, if ``condition`` is true.\n\n        *NOTE:* Robot Framework 4.0 introduced built-in IF/ELSE support and using\n        that is generally recommended over using this keyword.\n\n        The given ``condition`` is evaluated in Python as explained in the\n        `Evaluating expressions` section, and ``name`` and ``*args`` have same\n        semantics as with `Run Keyword`.\n\n        Example, a simple if/else construct:\n        | `Run Keyword If` | '${status}' == 'OK' | Some Action    | arg |\n        | `Run Keyword If` | '${status}' != 'OK' | Another Action |\n\n        In this example, only either ``Some Action`` or ``Another Action`` is\n        executed, based on the value of the ``${status}`` variable.\n\n        Variables used like ``${variable}``, as in the examples above, are\n        replaced in the expression before evaluation. Variables are also\n        available in the evaluation namespace and can be accessed using special\n        ``$variable`` syntax as explained in the `Evaluating expressions` section.\n\n        Example:\n        | `Run Keyword If` | $result is None or $result == 'FAIL' | Keyword |\n\n        This keyword supports also optional ELSE and ELSE IF branches. Both\n        of them are defined in ``*args`` and must use exactly format ``ELSE``\n        or ``ELSE IF``, respectively. ELSE branches must contain first the\n        name of the keyword to execute and then its possible arguments. ELSE\n        IF branches must first contain a condition, like the first argument\n        to this keyword, and then the keyword to execute and its possible\n        arguments. It is possible to have ELSE branch after ELSE IF and to\n        have multiple ELSE IF branches. Nested `Run Keyword If` usage is not\n        supported when using ELSE and/or ELSE IF branches.\n\n        Given previous example, if/else construct can also be created like this:\n        | `Run Keyword If` | '${status}' == 'PASS' | Some Action | arg | ELSE | Another Action |\n\n        The return value of this keyword is the return value of the actually\n        executed keyword or Python ``None`` if no keyword was executed (i.e.\n        if ``condition`` was false). Hence, it is recommended to use ELSE\n        and/or ELSE IF branches to conditionally assign return values from\n        keyword to variables (see `Set Variable If` you need to set fixed\n        values conditionally). This is illustrated by the example below:\n\n        | ${var1} =   | `Run Keyword If` | ${rc} == 0     | Some keyword returning a value |\n        | ...         | ELSE IF          | 0 < ${rc} < 42 | Another keyword |\n        | ...         | ELSE IF          | ${rc} < 0      | Another keyword with args | ${rc} | arg2 |\n        | ...         | ELSE             | Final keyword to handle abnormal cases | ${rc} |\n        | ${var2} =   | `Run Keyword If` | ${condition}  | Some keyword |\n\n        In this example, ${var2} will be set to ``None`` if ${condition} is\n        false.\n\n        Notice that ``ELSE`` and ``ELSE IF`` control words must be used\n        explicitly and thus cannot come from variables. If you need to use\n        literal ``ELSE`` and ``ELSE IF`` strings as arguments, you can escape\n        them with a backslash like ``\\ELSE`` and ``\\ELSE IF``.\n        "
        (args, branch) = self._split_elif_or_else_branch(args)
        if self._is_true(condition):
            return self.run_keyword(name, *args)
        return branch()

    def _split_elif_or_else_branch(self, args):
        if False:
            i = 10
            return i + 15
        if 'ELSE IF' in args:
            (args, branch) = self._split_branch(args, 'ELSE IF', 2, 'condition and keyword')
            return (args, lambda : self.run_keyword_if(*branch))
        if 'ELSE' in args:
            (args, branch) = self._split_branch(args, 'ELSE', 1, 'keyword')
            return (args, lambda : self.run_keyword(*branch))
        return (args, lambda : None)

    def _split_branch(self, args, control_word, required, required_error):
        if False:
            while True:
                i = 10
        index = list(args).index(control_word)
        branch = self._variables.replace_list(args[index + 1:], required)
        if len(branch) < required:
            raise DataError(f'{control_word} requires {required_error}.')
        return (args[:index], branch)

    @run_keyword_variant(resolve=1, dry_run=True)
    def run_keyword_unless(self, condition, name, *args):
        if False:
            i = 10
            return i + 15
        '*DEPRECATED since RF 5.0. Use Native IF/ELSE or `Run Keyword If` instead.*\n\n        Runs the given keyword with the given arguments if ``condition`` is false.\n\n        See `Run Keyword If` for more information and an example. Notice that this\n        keyword does not support ELSE or ELSE IF branches like `Run Keyword If` does.\n        '
        if not self._is_true(condition):
            return self.run_keyword(name, *args)

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_and_ignore_error(self, name, *args):
        if False:
            print('Hello World!')
        'Runs the given keyword with the given arguments and ignores possible error.\n\n        This keyword returns two values, so that the first is either string\n        ``PASS`` or ``FAIL``, depending on the status of the executed keyword.\n        The second value is either the return value of the keyword or the\n        received error message. See `Run Keyword And Return Status` If you are\n        only interested in the execution status.\n\n        The keyword name and arguments work as in `Run Keyword`. See\n        `Run Keyword If` for a usage example.\n\n        Errors caused by invalid syntax, timeouts, or fatal exceptions are not\n        caught by this keyword. Otherwise this keyword itself never fails.\n\n        *NOTE:* Robot Framework 5.0 introduced native TRY/EXCEPT functionality\n        that is generally recommended for error handling.\n        '
        try:
            return ('PASS', self.run_keyword(name, *args))
        except ExecutionFailed as err:
            if err.dont_continue or err.skip:
                raise
            return ('FAIL', str(err))

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_and_warn_on_failure(self, name, *args):
        if False:
            return 10
        'Runs the specified keyword logs a warning if the keyword fails.\n\n        This keyword is similar to `Run Keyword And Ignore Error` but if the executed\n        keyword fails, the error message is logged as a warning to make it more\n        visible. Returns status and possible return value or error message exactly\n        like `Run Keyword And Ignore Error` does.\n\n        Errors caused by invalid syntax, timeouts, or fatal exceptions are not\n        caught by this keyword. Otherwise this keyword itself never fails.\n\n        New in Robot Framework 4.0.\n        '
        (status, message) = self.run_keyword_and_ignore_error(name, *args)
        if status == 'FAIL':
            logger.warn(f"Executing keyword '{name}' failed:\n{message}")
        return (status, message)

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_and_return_status(self, name, *args):
        if False:
            i = 10
            return i + 15
        'Runs the given keyword with given arguments and returns the status as a Boolean value.\n\n        This keyword returns Boolean ``True`` if the keyword that is executed\n        succeeds and ``False`` if it fails. This is useful, for example, in\n        combination with `Run Keyword If`. If you are interested in the error\n        message or return value, use `Run Keyword And Ignore Error` instead.\n\n        The keyword name and arguments work as in `Run Keyword`.\n\n        Example:\n        | ${passed} = | `Run Keyword And Return Status` | Keyword | args |\n        | `Run Keyword If` | ${passed} | Another keyword |\n\n        Errors caused by invalid syntax, timeouts, or fatal exceptions are not\n        caught by this keyword. Otherwise this keyword itself never fails.\n        '
        (status, _) = self.run_keyword_and_ignore_error(name, *args)
        return status == 'PASS'

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_and_continue_on_failure(self, name, *args):
        if False:
            i = 10
            return i + 15
        'Runs the keyword and continues execution even if a failure occurs.\n\n        The keyword name and arguments work as with `Run Keyword`.\n\n        Example:\n        | Run Keyword And Continue On Failure | Fail | This is a stupid example |\n        | Log | This keyword is executed |\n\n        The execution is not continued if the failure is caused by invalid syntax,\n        timeout, or fatal exception.\n        '
        try:
            return self.run_keyword(name, *args)
        except ExecutionFailed as err:
            if not err.dont_continue:
                err.continue_on_failure = True
            raise err

    @run_keyword_variant(resolve=1, dry_run=True)
    def run_keyword_and_expect_error(self, expected_error, name, *args):
        if False:
            for i in range(10):
                print('nop')
        'Runs the keyword and checks that the expected error occurred.\n\n        The keyword to execute and its arguments are specified using ``name``\n        and ``*args`` exactly like with `Run Keyword`.\n\n        The expected error must be given in the same format as in Robot Framework\n        reports. By default it is interpreted as a glob pattern with ``*``, ``?``\n        and ``[chars]`` as wildcards, but that can be changed by using various\n        prefixes explained in the table below. Prefixes are case-sensitive and\n        they must be separated from the actual message with a colon and an\n        optional space like ``PREFIX: Message`` or ``PREFIX:Message``.\n\n        | = Prefix = | = Explanation = |\n        | ``EQUALS`` | Exact match. Especially useful if the error contains glob wildcards. |\n        | ``STARTS`` | Error must start with the specified error. |\n        | ``REGEXP`` | Regular expression match. |\n        | ``GLOB``   | Same as the default behavior. |\n\n        See the `Pattern matching` section for more information about glob\n        patterns and regular expressions.\n\n        If the expected error occurs, the error message is returned and it can\n        be further processed or tested if needed. If there is no error, or the\n        error does not match the expected error, this keyword fails.\n\n        Examples:\n        | Run Keyword And Expect Error | My error            | Keyword | arg |\n        | Run Keyword And Expect Error | ValueError: *       | Some Keyword  |\n        | Run Keyword And Expect Error | STARTS: ValueError: | Some Keyword  |\n        | Run Keyword And Expect Error | EQUALS:No match for \'//input[@type="text"]\' |\n        | ...                          | Find Element | //input[@type="text"] |\n        | ${msg} =                     | Run Keyword And Expect Error | * |\n        | ...                          | Keyword | arg1 | arg2 |\n        | Log To Console | ${msg} |\n\n        Errors caused by invalid syntax, timeouts, or fatal exceptions are not\n        caught by this keyword.\n\n        *NOTE:* Regular expression matching used to require only the beginning\n        of the error to match the given pattern. That was changed in Robot\n        Framework 5.0 and nowadays the pattern must match the error fully.\n        To match only the beginning, add ``.*`` at the end of the pattern like\n        ``REGEXP: Start.*``.\n\n        *NOTE:* Robot Framework 5.0 introduced native TRY/EXCEPT functionality\n        that is generally recommended for error handling. It supports same\n        pattern matching syntax as this keyword.\n        '
        try:
            self.run_keyword(name, *args)
        except ExecutionFailed as err:
            if err.dont_continue or err.skip:
                raise
            error = err.message
        else:
            raise AssertionError(f"Expected error '{expected_error}' did not occur.")
        if not self._error_is_expected(error, expected_error):
            raise AssertionError(f"Expected error '{expected_error}' but got '{error}'.")
        return error

    def _error_is_expected(self, error, expected_error):
        if False:
            return 10
        glob = self._matches
        matchers = {'GLOB': glob, 'EQUALS': lambda s, p: s == p, 'STARTS': lambda s, p: s.startswith(p), 'REGEXP': lambda s, p: re.fullmatch(p, s) is not None}
        prefixes = tuple((prefix + ':' for prefix in matchers))
        if not expected_error.startswith(prefixes):
            return glob(error, expected_error)
        (prefix, expected_error) = expected_error.split(':', 1)
        return matchers[prefix](error, expected_error.lstrip())

    @run_keyword_variant(resolve=1, dry_run=True)
    def repeat_keyword(self, repeat, name, *args):
        if False:
            return 10
        "Executes the specified keyword multiple times.\n\n        ``name`` and ``args`` define the keyword that is executed similarly as\n        with `Run Keyword`. ``repeat`` specifies how many times (as a count) or\n        how long time (as a timeout) the keyword should be executed.\n\n        If ``repeat`` is given as count, it specifies how many times the\n        keyword should be executed. ``repeat`` can be given as an integer or\n        as a string that can be converted to an integer. If it is a string,\n        it can have postfix ``times`` or ``x`` (case and space insensitive)\n        to make the expression more explicit.\n\n        If ``repeat`` is given as timeout, it must be in Robot Framework's\n        time format (e.g. ``1 minute``, ``2 min 3 s``). Using a number alone\n        (e.g. ``1`` or ``1.5``) does not work in this context.\n\n        If ``repeat`` is zero or negative, the keyword is not executed at\n        all. This keyword fails immediately if any of the execution\n        rounds fails.\n\n        Examples:\n        | Repeat Keyword | 5 times   | Go to Previous Page |\n        | Repeat Keyword | ${var}    | Some Keyword | arg1 | arg2 |\n        | Repeat Keyword | 2 minutes | Some Keyword | arg1 | arg2 |\n        "
        try:
            count = self._get_repeat_count(repeat)
        except RuntimeError as err:
            timeout = self._get_repeat_timeout(repeat)
            if timeout is None:
                raise err
            keywords = self._keywords_repeated_by_timeout(timeout, name, args)
        else:
            keywords = self._keywords_repeated_by_count(count, name, args)
        self._run_keywords(keywords)

    def _get_repeat_count(self, times, require_postfix=False):
        if False:
            i = 10
            return i + 15
        times = normalize(str(times))
        if times.endswith('times'):
            times = times[:-5]
        elif times.endswith('x'):
            times = times[:-1]
        elif require_postfix:
            raise ValueError
        return self._convert_to_integer(times)

    def _get_repeat_timeout(self, timestr):
        if False:
            for i in range(10):
                print('nop')
        try:
            float(timestr)
        except ValueError:
            pass
        else:
            return None
        try:
            return timestr_to_secs(timestr)
        except ValueError:
            return None

    def _keywords_repeated_by_count(self, count, name, args):
        if False:
            for i in range(10):
                print('nop')
        if count <= 0:
            self.log(f"Keyword '{name}' repeated zero times.")
        for i in range(count):
            self.log(f'Repeating keyword, round {i + 1}/{count}.')
            yield (name, args)

    def _keywords_repeated_by_timeout(self, timeout, name, args):
        if False:
            for i in range(10):
                print('nop')
        if timeout <= 0:
            self.log(f"Keyword '{name}' repeated zero times.")
        round = 0
        maxtime = time.time() + timeout
        while time.time() < maxtime:
            round += 1
            remaining = secs_to_timestr(maxtime - time.time(), compact=True)
            self.log(f'Repeating keyword, round {round}, {remaining} remaining.')
            yield (name, args)

    @run_keyword_variant(resolve=2, dry_run=True)
    def wait_until_keyword_succeeds(self, retry, retry_interval, name, *args):
        if False:
            return 10
        'Runs the specified keyword and retries if it fails.\n\n        ``name`` and ``args`` define the keyword that is executed similarly\n        as with `Run Keyword`. How long to retry running the keyword is\n        defined using ``retry`` argument either as timeout or count.\n        ``retry_interval`` is the time to wait between execution attempts.\n\n        If ``retry`` is given as timeout, it must be in Robot Framework\'s\n        time format (e.g. ``1 minute``, ``2 min 3 s``, ``4.5``) that is\n        explained in an appendix of Robot Framework User Guide. If it is\n        given as count, it must have ``times`` or ``x`` postfix (e.g.\n        ``5 times``, ``10 x``). ``retry_interval`` must always be given in\n        Robot Framework\'s time format.\n\n        By default ``retry_interval`` is the time to wait _after_ a keyword has\n        failed. For example, if the first run takes 2 seconds and the retry\n        interval is 3 seconds, the second run starts 5 seconds after the first\n        run started. If ``retry_interval`` start with prefix ``strict:``, the\n        execution time of the previous keyword is subtracted from the retry time.\n        With the earlier example the second run would thus start 3 seconds after\n        the first run started. A warning is logged if keyword execution time is\n        longer than a strict interval.\n\n        If the keyword does not succeed regardless of retries, this keyword\n        fails. If the executed keyword passes, its return value is returned.\n\n        Examples:\n        | Wait Until Keyword Succeeds | 2 min | 5 sec | My keyword | argument |\n        | ${result} = | Wait Until Keyword Succeeds | 3x | 200ms | My keyword |\n        | ${result} = | Wait Until Keyword Succeeds | 3x | strict: 200ms | My keyword |\n\n        All normal failures are caught by this keyword. Errors caused by\n        invalid syntax, test or keyword timeouts, or fatal exceptions (caused\n        e.g. by `Fatal Error`) are not caught.\n\n        Running the same keyword multiple times inside this keyword can create\n        lots of output and considerably increase the size of the generated\n        output files. It is possible to remove unnecessary keywords from\n        the outputs using ``--RemoveKeywords WUKS`` command line option.\n\n        Support for "strict" retry interval is new in Robot Framework 4.1.\n        '
        maxtime = count = -1
        try:
            count = self._get_repeat_count(retry, require_postfix=True)
        except ValueError:
            timeout = timestr_to_secs(retry)
            maxtime = time.time() + timeout
            message = f'for {secs_to_timestr(timeout)}'
        else:
            if count <= 0:
                raise ValueError(f'Retry count {count} is not positive.')
            message = f'{count} time{s(count)}'
        if is_string(retry_interval) and normalize(retry_interval).startswith('strict:'):
            retry_interval = retry_interval.split(':', 1)[1].strip()
            strict_interval = True
        else:
            strict_interval = False
        retry_interval = sleep_time = timestr_to_secs(retry_interval)
        while True:
            start_time = time.time()
            try:
                return self.run_keyword(name, *args)
            except ExecutionFailed as err:
                if err.dont_continue or err.skip:
                    raise
                count -= 1
                if time.time() > maxtime > 0 or count == 0:
                    raise AssertionError(f"Keyword '{name}' failed after retrying {message}. The last error was: {err}")
            finally:
                if strict_interval:
                    execution_time = time.time() - start_time
                    sleep_time = retry_interval - execution_time
                    if sleep_time < 0:
                        logger.warn(f'Keyword execution time {secs_to_timestr(execution_time)} is longer than retry interval {secs_to_timestr(retry_interval)}.')
            self._sleep_in_parts(sleep_time)

    @run_keyword_variant(resolve=1)
    def set_variable_if(self, condition, *values):
        if False:
            return 10
        "Sets variable based on the given condition.\n\n        The basic usage is giving a condition and two values. The\n        given condition is first evaluated the same way as with the\n        `Should Be True` keyword. If the condition is true, then the\n        first value is returned, and otherwise the second value is\n        returned. The second value can also be omitted, in which case\n        it has a default value None. This usage is illustrated in the\n        examples below, where ``${rc}`` is assumed to be zero.\n\n        | ${var1} = | Set Variable If | ${rc} == 0 | zero     | nonzero |\n        | ${var2} = | Set Variable If | ${rc} > 0  | value1   | value2  |\n        | ${var3} = | Set Variable If | ${rc} > 0  | whatever |         |\n        =>\n        | ${var1} = 'zero'\n        | ${var2} = 'value2'\n        | ${var3} = None\n\n        It is also possible to have 'else if' support by replacing the\n        second value with another condition, and having two new values\n        after it. If the first condition is not true, the second is\n        evaluated and one of the values after it is returned based on\n        its truth value. This can be continued by adding more\n        conditions without a limit.\n\n        | ${var} = | Set Variable If | ${rc} == 0        | zero           |\n        | ...      | ${rc} > 0       | greater than zero | less then zero |\n        |          |\n        | ${var} = | Set Variable If |\n        | ...      | ${rc} == 0      | zero              |\n        | ...      | ${rc} == 1      | one               |\n        | ...      | ${rc} == 2      | two               |\n        | ...      | ${rc} > 2       | greater than two  |\n        | ...      | ${rc} < 0       | less than zero    |\n\n        Use `Get Variable Value` if you need to set variables\n        dynamically based on whether a variable exist or not.\n        "
        values = list(values)
        while True:
            values = self._verify_values_for_set_variable_if(values)
            if self._is_true(condition):
                return self._variables.replace_scalar(values[0])
            if len(values) == 1:
                return None
            if len(values) == 2:
                return self._variables.replace_scalar(values[1])
            (condition, *values) = values[1:]
            condition = self._variables.replace_scalar(condition)

    def _verify_values_for_set_variable_if(self, values):
        if False:
            return 10
        if not values:
            raise RuntimeError('At least one value is required.')
        if is_list_variable(values[0]):
            values[:1] = [escape(item) for item in self._variables[values[0]]]
            return self._verify_values_for_set_variable_if(values)
        return values

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_if_test_failed(self, name, *args):
        if False:
            while True:
                i = 10
        'Runs the given keyword with the given arguments, if the test failed.\n\n        This keyword can only be used in a test teardown. Trying to use it\n        anywhere else results in an error.\n\n        Otherwise, this keyword works exactly like `Run Keyword`, see its\n        documentation for more details.\n        '
        test = self._get_test_in_teardown('Run Keyword If Test Failed')
        if test.failed:
            return self.run_keyword(name, *args)

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_if_test_passed(self, name, *args):
        if False:
            while True:
                i = 10
        'Runs the given keyword with the given arguments, if the test passed.\n\n        This keyword can only be used in a test teardown. Trying to use it\n        anywhere else results in an error.\n\n        Otherwise, this keyword works exactly like `Run Keyword`, see its\n        documentation for more details.\n        '
        test = self._get_test_in_teardown('Run Keyword If Test Passed')
        if test.passed:
            return self.run_keyword(name, *args)

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_if_timeout_occurred(self, name, *args):
        if False:
            i = 10
            return i + 15
        'Runs the given keyword if either a test or a keyword timeout has occurred.\n\n        This keyword can only be used in a test teardown. Trying to use it\n        anywhere else results in an error.\n\n        Otherwise, this keyword works exactly like `Run Keyword`, see its\n        documentation for more details.\n        '
        self._get_test_in_teardown('Run Keyword If Timeout Occurred')
        if self._context.timeout_occurred:
            return self.run_keyword(name, *args)

    def _get_test_in_teardown(self, kwname):
        if False:
            i = 10
            return i + 15
        ctx = self._context
        if ctx.test and ctx.in_test_teardown:
            return ctx.test
        raise RuntimeError(f"Keyword '{kwname}' can only be used in test teardown.")

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_if_all_tests_passed(self, name, *args):
        if False:
            for i in range(10):
                print('nop')
        'Runs the given keyword with the given arguments, if all tests passed.\n\n        This keyword can only be used in a suite teardown. Trying to use it\n        anywhere else results in an error.\n\n        Otherwise, this keyword works exactly like `Run Keyword`, see its\n        documentation for more details.\n        '
        suite = self._get_suite_in_teardown('Run Keyword If All Tests Passed')
        if suite.statistics.failed == 0:
            return self.run_keyword(name, *args)

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_if_any_tests_failed(self, name, *args):
        if False:
            for i in range(10):
                print('nop')
        'Runs the given keyword with the given arguments, if one or more tests failed.\n\n        This keyword can only be used in a suite teardown. Trying to use it\n        anywhere else results in an error.\n\n        Otherwise, this keyword works exactly like `Run Keyword`, see its\n        documentation for more details.\n        '
        suite = self._get_suite_in_teardown('Run Keyword If Any Tests Failed')
        if suite.statistics.failed > 0:
            return self.run_keyword(name, *args)

    def _get_suite_in_teardown(self, kw):
        if False:
            print('Hello World!')
        if not self._context.in_suite_teardown:
            raise RuntimeError(f"Keyword '{kw}' can only be used in suite teardown.")
        return self._context.suite

class _Control(_BuiltInBase):

    def skip(self, msg='Skipped with Skip keyword.'):
        if False:
            while True:
                i = 10
        'Skips the rest of the current test.\n\n        Skips the remaining keywords in the current test and sets the given\n        message to the test. If the test has teardown, it will be executed.\n        '
        raise SkipExecution(msg)

    def skip_if(self, condition, msg=None):
        if False:
            i = 10
            return i + 15
        'Skips the rest of the current test if the ``condition`` is True.\n\n        Skips the remaining keywords in the current test and sets the given\n        message to the test. If ``msg`` is not given, the ``condition`` will\n        be used as the message. If the test has teardown, it will be executed.\n\n        If the ``condition`` evaluates to False, does nothing.\n        '
        if self._is_true(condition):
            raise SkipExecution(msg or condition)

    def continue_for_loop(self):
        if False:
            return 10
        "Skips the current FOR loop iteration and continues from the next.\n\n        ---\n\n        *NOTE:* Robot Framework 5.0 added support for native ``CONTINUE`` statement that\n        is recommended over this keyword. In the examples below, ``Continue For Loop``\n        can simply be replaced with ``CONTINUE``. In addition to that, native ``IF``\n        syntax (new in RF 4.0) or inline ``IF`` syntax (new in RF 5.0) can be used\n        instead of ``Run Keyword If``. For example, the first example below could be\n        written like this instead:\n\n        | IF    '${var}' == 'CONTINUE'    CONTINUE\n\n        This keyword will eventually be deprecated and removed.\n\n        ---\n\n        Skips the remaining keywords in the current FOR loop iteration and\n        continues from the next one. Starting from Robot Framework 5.0, this\n        keyword can only be used inside a loop, not in a keyword used in a loop.\n\n        Example:\n        | FOR | ${var}         | IN                     | @{VALUES}         |\n        |     | Run Keyword If | '${var}' == 'CONTINUE' | Continue For Loop |\n        |     | Do Something   | ${var}                 |\n        | END |\n\n        See `Continue For Loop If` to conditionally continue a FOR loop without\n        using `Run Keyword If` or other wrapper keywords.\n        "
        if not self._context.allow_loop_control:
            raise DataError("'Continue For Loop' can only be used inside a loop.")
        self.log('Continuing for loop from the next iteration.')
        raise ContinueLoop()

    def continue_for_loop_if(self, condition):
        if False:
            for i in range(10):
                print('nop')
        "Skips the current FOR loop iteration if the ``condition`` is true.\n\n        ---\n\n        *NOTE:* Robot Framework 5.0 added support for native ``CONTINUE`` statement\n        and for inline ``IF``, and that combination should be used instead of this\n        keyword. For example, ``Continue For Loop If`` usage in the example below\n        could be replaced with\n\n        | IF    '${var}' == 'CONTINUE'    CONTINUE\n\n        This keyword will eventually be deprecated and removed.\n\n        ---\n\n        A wrapper for `Continue For Loop` to continue a FOR loop based on\n        the given condition. The condition is evaluated using the same\n        semantics as with `Should Be True` keyword.\n\n        Example:\n        | FOR | ${var}               | IN                     | @{VALUES} |\n        |     | Continue For Loop If | '${var}' == 'CONTINUE' |\n        |     | Do Something         | ${var}                 |\n        | END |\n        "
        if not self._context.allow_loop_control:
            raise DataError("'Continue For Loop If' can only be used inside a loop.")
        if self._is_true(condition):
            self.continue_for_loop()

    def exit_for_loop(self):
        if False:
            while True:
                i = 10
        "Stops executing the enclosing FOR loop.\n\n        ---\n\n        *NOTE:* Robot Framework 5.0 added support for native ``BREAK`` statement that\n        is recommended over this keyword. In the examples below, ``Exit For Loop``\n        can simply be replaced with ``BREAK``. In addition to that, native ``IF``\n        syntax (new in RF 4.0) or inline ``IF`` syntax (new in RF 5.0) can be used\n        instead of ``Run Keyword If``. For example, the first example below could be\n        written like this instead:\n\n        | IF    '${var}' == 'EXIT'    BREAK\n\n        This keyword will eventually be deprecated and removed.\n\n        ---\n\n        Exits the enclosing FOR loop and continues execution after it. Starting\n        from Robot Framework 5.0, this keyword can only be used inside a loop,\n        not in a keyword used in a loop.\n\n        Example:\n        | FOR | ${var}         | IN                 | @{VALUES}     |\n        |     | Run Keyword If | '${var}' == 'EXIT' | Exit For Loop |\n        |     | Do Something   | ${var} |\n        | END |\n\n        See `Exit For Loop If` to conditionally exit a FOR loop without\n        using `Run Keyword If` or other wrapper keywords.\n        "
        if not self._context.allow_loop_control:
            raise DataError("'Exit For Loop' can only be used inside a loop.")
        self.log('Exiting for loop altogether.')
        raise BreakLoop()

    def exit_for_loop_if(self, condition):
        if False:
            print('Hello World!')
        "Stops executing the enclosing FOR loop if the ``condition`` is true.\n\n        ---\n\n        *NOTE:* Robot Framework 5.0 added support for native ``BREAK`` statement\n        and for inline ``IF``, and that combination should be used instead of this\n        keyword. For example, ``Exit For Loop If`` usage in the example below\n        could be replaced with\n\n        | IF    '${var}' == 'EXIT'    BREAK\n\n        This keyword will eventually be deprecated and removed.\n\n        ---\n\n        A wrapper for `Exit For Loop` to exit a FOR loop based on\n        the given condition. The condition is evaluated using the same\n        semantics as with `Should Be True` keyword.\n\n        Example:\n        | FOR | ${var}           | IN                 | @{VALUES} |\n        |     | Exit For Loop If | '${var}' == 'EXIT' |\n        |     | Do Something     | ${var}             |\n        | END |\n        "
        if not self._context.allow_loop_control:
            raise DataError("'Exit For Loop If' can only be used inside a loop.")
        if self._is_true(condition):
            self.exit_for_loop()

    @run_keyword_variant(resolve=0)
    def return_from_keyword(self, *return_values):
        if False:
            return 10
        "Returns from the enclosing user keyword.\n\n        ---\n\n        *NOTE:* Robot Framework 5.0 added support for native ``RETURN`` statement that\n        is recommended over this keyword. In the examples below, ``Return From Keyword``\n        can simply be replaced with ``RETURN``. In addition to that, native ``IF``\n        syntax (new in RF 4.0) or inline ``IF`` syntax (new in RF 5.0) can be used\n        instead of ``Run Keyword If``. For example, the first example below could be\n        written like this instead:\n\n        | IF    ${rc} < 0    RETURN\n\n        This keyword will eventually be deprecated and removed.\n\n        ---\n\n        This keyword can be used to return from a user keyword with PASS status\n        without executing it fully. It is also possible to return values\n        similarly as with the ``[Return]`` setting. For more detailed information\n        about working with the return values, see the User Guide.\n\n        This keyword is typically wrapped to some other keyword, such as\n        `Run Keyword If`, to return based on a condition:\n\n        | Run Keyword If    ${rc} < 0    Return From Keyword\n\n        It is possible to use this keyword to return from a keyword also inside\n        a for loop. That, as well as returning values, is demonstrated by the\n        `Find Index` keyword in the following somewhat advanced example.\n        Notice that it is often a good idea to move this kind of complicated\n        logic into a library.\n\n        | ***** Variables *****\n        | @{LIST} =    foo    baz\n        |\n        | ***** Test Cases *****\n        | Example\n        |     ${index} =    Find Index    baz    @{LIST}\n        |     Should Be Equal    ${index}    ${1}\n        |     ${index} =    Find Index    non existing    @{LIST}\n        |     Should Be Equal    ${index}    ${-1}\n        |\n        | ***** Keywords *****\n        | Find Index\n        |    [Arguments]    ${element}    @{items}\n        |    ${index} =    Set Variable    ${0}\n        |    FOR    ${item}    IN    @{items}\n        |        Run Keyword If    '${item}' == '${element}'    Return From Keyword    ${index}\n        |        ${index} =    Set Variable    ${index + 1}\n        |    END\n        |    Return From Keyword    ${-1}\n\n        The most common use case, returning based on an expression, can be\n        accomplished directly with `Return From Keyword If`. See also\n        `Run Keyword And Return` and `Run Keyword And Return If`.\n        "
        self._return_from_keyword(return_values)

    def _return_from_keyword(self, return_values=None, failures=None):
        if False:
            while True:
                i = 10
        self.log('Returning from the enclosing user keyword.')
        raise ReturnFromKeyword(return_values, failures)

    @run_keyword_variant(resolve=1)
    def return_from_keyword_if(self, condition, *return_values):
        if False:
            print('Hello World!')
        "Returns from the enclosing user keyword if ``condition`` is true.\n\n        ---\n\n        *NOTE:* Robot Framework 5.0 added support for native ``RETURN`` statement\n        and for inline ``IF``, and that combination should be used instead of this\n        keyword. For example, ``Return From Keyword`` usage in the example below\n        could be replaced with\n\n        | IF    '${item}' == '${element}'    RETURN    ${index}\n\n        This keyword will eventually be deprecated and removed.\n\n        ---\n\n        A wrapper for `Return From Keyword` to return based on the given\n        condition. The condition is evaluated using the same semantics as\n        with `Should Be True` keyword.\n\n        Given the same example as in `Return From Keyword`, we can rewrite the\n        `Find Index` keyword as follows:\n\n        | ***** Keywords *****\n        | Find Index\n        |    [Arguments]    ${element}    @{items}\n        |    ${index} =    Set Variable    ${0}\n        |    FOR    ${item}    IN    @{items}\n        |        Return From Keyword If    '${item}' == '${element}'    ${index}\n        |        ${index} =    Set Variable    ${index + 1}\n        |    END\n        |    Return From Keyword    ${-1}\n\n        See also `Run Keyword And Return` and `Run Keyword And Return If`.\n        "
        if self._is_true(condition):
            self._return_from_keyword(return_values)

    @run_keyword_variant(resolve=0, dry_run=True)
    def run_keyword_and_return(self, name, *args):
        if False:
            i = 10
            return i + 15
        'Runs the specified keyword and returns from the enclosing user keyword.\n\n        The keyword to execute is defined with ``name`` and ``*args`` exactly\n        like with `Run Keyword`. After running the keyword, returns from the\n        enclosing user keyword and passes possible return value from the\n        executed keyword further. Returning from a keyword has exactly same\n        semantics as with `Return From Keyword`.\n\n        Example:\n        | `Run Keyword And Return`  | `My Keyword` | arg1 | arg2 |\n        | # Above is equivalent to: |\n        | ${result} =               | `My Keyword` | arg1 | arg2 |\n        | `Return From Keyword`     | ${result}    |      |      |\n\n        Use `Run Keyword And Return If` if you want to run keyword and return\n        based on a condition.\n        '
        try:
            ret = self.run_keyword(name, *args)
        except ExecutionFailed as err:
            self._return_from_keyword(failures=[err])
        else:
            self._return_from_keyword(return_values=[escape(ret)])

    @run_keyword_variant(resolve=1, dry_run=True)
    def run_keyword_and_return_if(self, condition, name, *args):
        if False:
            while True:
                i = 10
        'Runs the specified keyword and returns from the enclosing user keyword.\n\n        A wrapper for `Run Keyword And Return` to run and return based on\n        the given ``condition``. The condition is evaluated using the same\n        semantics as with `Should Be True` keyword.\n\n        Example:\n        | `Run Keyword And Return If` | ${rc} > 0 | `My Keyword` | arg1 | arg2 |\n        | # Above is equivalent to:   |\n        | `Run Keyword If`            | ${rc} > 0 | `Run Keyword And Return` | `My Keyword ` | arg1 | arg2 |\n\n        Use `Return From Keyword If` if you want to return a certain value\n        based on a condition.\n        '
        if self._is_true(condition):
            self.run_keyword_and_return(name, *args)

    def pass_execution(self, message, *tags):
        if False:
            print('Hello World!')
        'Skips rest of the current test, setup, or teardown with PASS status.\n\n        This keyword can be used anywhere in the test data, but the place where\n        used affects the behavior:\n\n        - When used in any setup or teardown (suite, test or keyword), passes\n          that setup or teardown. Possible keyword teardowns of the started\n          keywords are executed. Does not affect execution or statuses\n          otherwise.\n        - When used in a test outside setup or teardown, passes that particular\n          test case. Possible test and keyword teardowns are executed.\n\n        Possible continuable failures before this keyword is used, as well as\n        failures in executed teardowns, will fail the execution.\n\n        It is mandatory to give a message explaining why execution was passed.\n        By default the message is considered plain text, but starting it with\n        ``*HTML*`` allows using HTML formatting.\n\n        It is also possible to modify test tags passing tags after the message\n        similarly as with `Fail` keyword. Tags starting with a hyphen\n        (e.g. ``-regression``) are removed and others added. Tags are modified\n        using `Set Tags` and `Remove Tags` internally, and the semantics\n        setting and removing them are the same as with these keywords.\n\n        Examples:\n        | Pass Execution | All features available in this version tested. |\n        | Pass Execution | Deprecated test. | deprecated | -regression    |\n\n        This keyword is typically wrapped to some other keyword, such as\n        `Run Keyword If`, to pass based on a condition. The most common case\n        can be handled also with `Pass Execution If`:\n\n        | Run Keyword If    | ${rc} < 0 | Pass Execution | Negative values are cool. |\n        | Pass Execution If | ${rc} < 0 | Negative values are cool. |\n\n        Passing execution in the middle of a test, setup or teardown should be\n        used with care. In the worst case it leads to tests that skip all the\n        parts that could actually uncover problems in the tested application.\n        In cases where execution cannot continue due to external factors,\n        it is often safer to fail the test case and make it non-critical.\n        '
        message = message.strip()
        if not message:
            raise RuntimeError('Message cannot be empty.')
        self._set_and_remove_tags(tags)
        (log_message, level) = self._get_logged_test_message_and_level(message)
        self.log(f'Execution passed with message:\n{log_message}', level)
        raise PassExecution(message)

    @run_keyword_variant(resolve=1)
    def pass_execution_if(self, condition, message, *tags):
        if False:
            i = 10
            return i + 15
        "Conditionally skips rest of the current test, setup, or teardown with PASS status.\n\n        A wrapper for `Pass Execution` to skip rest of the current test,\n        setup or teardown based the given ``condition``. The condition is\n        evaluated similarly as with `Should Be True` keyword, and ``message``\n        and ``*tags`` have same semantics as with `Pass Execution`.\n\n        Example:\n        | FOR | ${var}            | IN                     | @{VALUES}               |\n        |     | Pass Execution If | '${var}' == 'EXPECTED' | Correct value was found |\n        |     | Do Something      | ${var}                 |\n        | END |\n        "
        if self._is_true(condition):
            message = self._variables.replace_string(message)
            tags = self._variables.replace_list(tags)
            self.pass_execution(message, *tags)

class _Misc(_BuiltInBase):

    def no_operation(self):
        if False:
            return 10
        'Does absolutely nothing.'

    def sleep(self, time_, reason=None):
        if False:
            for i in range(10):
                print('nop')
        'Pauses the test executed for the given time.\n\n        ``time`` may be either a number or a time string. Time strings are in\n        a format such as ``1 day 2 hours 3 minutes 4 seconds 5milliseconds`` or\n        ``1d 2h 3m 4s 5ms``, and they are fully explained in an appendix of\n        Robot Framework User Guide. Providing a value without specifying minutes\n        or seconds, defaults to seconds.\n        Optional `reason` can be used to explain why\n        sleeping is necessary. Both the time slept and the reason are logged.\n\n        Examples:\n        | Sleep | 42                   |\n        | Sleep | 1.5                  |\n        | Sleep | 2 minutes 10 seconds |\n        | Sleep | 10s                  | Wait for a reply |\n        '
        seconds = timestr_to_secs(time_)
        if seconds < 0:
            seconds = 0
        self._sleep_in_parts(seconds)
        self.log(f'Slept {secs_to_timestr(seconds)}.')
        if reason:
            self.log(reason)

    def _sleep_in_parts(self, seconds):
        if False:
            for i in range(10):
                print('nop')
        endtime = time.time() + float(seconds)
        while True:
            remaining = endtime - time.time()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.01))

    def catenate(self, *items):
        if False:
            i = 10
            return i + 15
        "Catenates the given items together and returns the resulted string.\n\n        By default, items are catenated with spaces, but if the first item\n        contains the string ``SEPARATOR=<sep>``, the separator ``<sep>`` is\n        used instead. Items are converted into strings when necessary.\n\n        Examples:\n        | ${str1} = | Catenate | Hello         | world |       |\n        | ${str2} = | Catenate | SEPARATOR=--- | Hello | world |\n        | ${str3} = | Catenate | SEPARATOR=    | Hello | world |\n        =>\n        | ${str1} = 'Hello world'\n        | ${str2} = 'Hello---world'\n        | ${str3} = 'Helloworld'\n        "
        if not items:
            return ''
        items = [str(item) for item in items]
        if items[0].startswith('SEPARATOR='):
            sep = items[0][len('SEPARATOR='):]
            items = items[1:]
        else:
            sep = ' '
        return sep.join(items)

    def log(self, message, level='INFO', html=False, console=False, repr='DEPRECATED', formatter='str'):
        if False:
            while True:
                i = 10
        'Logs the given message with the given level.\n\n        Valid levels are TRACE, DEBUG, INFO (default), WARN and ERROR.\n        In addition to that, there are pseudo log levels HTML and CONSOLE that\n        both log messages using INFO.\n\n        Messages below the current active log\n        level are ignored. See `Set Log Level` keyword and ``--loglevel``\n        command line option for more details about setting the level.\n\n        Messages logged with the WARN or ERROR levels are automatically\n        visible also in the console and in the Test Execution Errors section\n        in the log file.\n\n        If the ``html`` argument is given a true value (see `Boolean\n        arguments`) or the HTML pseudo log level is used, the message is\n        considered to be HTML and special characters\n        such as ``<`` are not escaped. For example, logging\n        ``<img src="image.png">`` creates an image in this case, but\n        otherwise the message is that exact string. When using the HTML pseudo\n        level, the messages is logged using the INFO level.\n\n        If the ``console`` argument is true or the CONSOLE pseudo level is\n        used, the message is written both to the console and to the log file.\n        When using the CONSOLE pseudo level, the message is logged using the\n        INFO level. If the message should not be logged to the log file or there\n        are special formatting needs, use the `Log To Console` keyword instead.\n\n        The ``formatter`` argument controls how to format the string\n        representation of the message. Possible values are ``str`` (default),\n        ``repr``, ``ascii``, ``len``, and ``type``. They work similarly to\n        Python built-in functions with same names. When using ``repr``, bigger\n        lists, dictionaries and other containers are also pretty-printed so\n        that there is one item per row. For more details see `String\n        representations`.\n\n        The old way to control string representation was using the ``repr``\n        argument. This argument has been deprecated and ``formatter=repr``\n        should be used instead.\n\n        Examples:\n        | Log | Hello, world!        |          |   | # Normal INFO message.   |\n        | Log | Warning, world!      | WARN     |   | # Warning.               |\n        | Log | <b>Hello</b>, world! | html=yes |   | # INFO message as HTML.  |\n        | Log | <b>Hello</b>, world! | HTML     |   | # Same as above.         |\n        | Log | <b>Hello</b>, world! | DEBUG    | html=true | # DEBUG as HTML. |\n        | Log | Hello, console!   | console=yes | | # Log also to the console. |\n        | Log | Hello, console!   | CONSOLE     | | # Log also to the console. |\n        | Log | Null is \\x00    | formatter=repr | | # Log ``\'Null is \\x00\'``. |\n\n        See `Log Many` if you want to log multiple messages in one go, and\n        `Log To Console` if you only want to write to the console.\n\n        Formatter options ``type`` and ``len`` are new in Robot Framework 5.0.\n        The CONSOLE level is new in Robot Framework 6.1.\n        '
        if repr == 'DEPRECATED':
            formatter = self._get_formatter(formatter)
        else:
            logger.warn("The 'repr' argument of 'BuiltIn.Log' is deprecated. Use 'formatter=repr' instead.")
            formatter = prepr if is_truthy(repr) else self._get_formatter(formatter)
        message = formatter(message)
        logger.write(message, level, html)
        if console:
            logger.console(message)

    def _get_formatter(self, formatter):
        if False:
            return 10
        try:
            return {'str': safe_str, 'repr': prepr, 'ascii': ascii, 'len': len, 'type': lambda x: type(x).__name__}[formatter.lower()]
        except KeyError:
            raise ValueError(f"Invalid formatter '{formatter}'. Available 'str', 'repr', 'ascii', 'len', and 'type'.")

    @run_keyword_variant(resolve=0)
    def log_many(self, *messages):
        if False:
            i = 10
            return i + 15
        'Logs the given messages as separate entries using the INFO level.\n\n        Supports also logging list and dictionary variable items individually.\n\n        Examples:\n        | Log Many | Hello   | ${var}  |\n        | Log Many | @{list} | &{dict} |\n\n        See `Log` and `Log To Console` keywords if you want to use alternative\n        log levels, use HTML, or log to the console.\n        '
        for msg in self._yield_logged_messages(messages):
            self.log(msg)

    def _yield_logged_messages(self, messages):
        if False:
            i = 10
            return i + 15
        for msg in messages:
            match = search_variable(msg)
            value = self._variables.replace_scalar(msg)
            if match.is_list_variable():
                for item in value:
                    yield item
            elif match.is_dict_variable():
                for (name, value) in value.items():
                    yield f'{name}={value}'
            else:
                yield value

    def log_to_console(self, message, stream='STDOUT', no_newline=False, format=''):
        if False:
            print('Hello World!')
        'Logs the given message to the console.\n\n        By default uses the standard output stream. Using the standard error\n        stream is possible by giving the ``stream`` argument value ``STDERR``\n        (case-insensitive).\n\n        By default appends a newline to the logged message. This can be\n        disabled by giving the ``no_newline`` argument a true value (see\n        `Boolean arguments`).\n\n        By default adds no alignment formatting. The ``format`` argument allows,\n        for example, alignment and customized padding of the log message. Please see the\n        [https://docs.python.org/3/library/string.html#formatspec|format specification] for\n        detailed alignment possibilities. This argument is new in Robot\n        Framework 5.0.\n\n        Examples:\n        | Log To Console | Hello, console!             |                 |\n        | Log To Console | Hello, stderr!              | STDERR          |\n        | Log To Console | Message starts here and is  | no_newline=true |\n        | Log To Console | continued without newline.  |                 |\n        | Log To Console | center message with * pad   | format=*^60     |\n        | Log To Console | 30 spaces before msg starts | format=>30      |\n\n        This keyword does not log the message to the normal log file. Use\n        `Log` keyword, possibly with argument ``console``, if that is desired.\n        '
        if format:
            format = '{:' + format + '}'
            message = format.format(message)
        logger.console(message, newline=is_falsy(no_newline), stream=stream)

    @run_keyword_variant(resolve=0)
    def comment(self, *messages):
        if False:
            i = 10
            return i + 15
        'Displays the given messages in the log file as keyword arguments.\n\n        This keyword does nothing with the arguments it receives, but as they\n        are visible in the log, this keyword can be used to display simple\n        messages. Given arguments are ignored so thoroughly that they can even\n        contain non-existing variables. If you are interested about variable\n        values, you can use the `Log` or `Log Many` keywords.\n        '
        pass

    def set_log_level(self, level):
        if False:
            i = 10
            return i + 15
        'Sets the log threshold to the specified level and returns the old level.\n\n        Messages below the level will not logged. The default logging level is\n        INFO, but it can be overridden with the command line option ``--loglevel``.\n\n        The available levels: TRACE, DEBUG, INFO (default), WARN, ERROR and NONE\n        (no logging).\n        '
        old = self._context.output.set_log_level(level)
        self._namespace.variables.set_global('${LOG_LEVEL}', level.upper())
        self.log(f'Log level changed from {old} to {level.upper()}.', level='DEBUG')
        return old

    def reload_library(self, name_or_instance):
        if False:
            print('Hello World!')
        'Rechecks what keywords the specified library provides.\n\n        Can be called explicitly in the test data or by a library itself\n        when keywords it provides have changed.\n\n        The library can be specified by its name or as the active instance of\n        the library. The latter is especially useful if the library itself\n        calls this keyword as a method.\n        '
        library = self._namespace.reload_library(name_or_instance)
        self.log(f'Reloaded library {library.name} with {len(library)} keywords.')

    @run_keyword_variant(resolve=0)
    def import_library(self, name, *args):
        if False:
            while True:
                i = 10
        'Imports a library with the given name and optional arguments.\n\n        This functionality allows dynamic importing of libraries while tests\n        are running. That may be necessary, if the library itself is dynamic\n        and not yet available when test data is processed. In a normal case,\n        libraries should be imported using the Library setting in the Setting\n        section.\n\n        This keyword supports importing libraries both using library\n        names and physical paths. When paths are used, they must be\n        given in absolute format or found from\n        [http://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#module-search-path|\n        search path]. Forward slashes can be used as path separators in all\n        operating systems.\n\n        It is possible to pass arguments to the imported library and also\n        named argument syntax works if the library supports it. ``WITH NAME``\n        syntax can be used to give a custom name to the imported library.\n\n        Examples:\n        | Import Library | MyLibrary |\n        | Import Library | ${CURDIR}/Lib.py | arg1 | named=arg2 | WITH NAME | Custom |\n        '
        (args, alias) = self._split_alias(args)
        try:
            self._namespace.import_library(name, args, alias)
        except DataError as err:
            raise RuntimeError(str(err))

    def _split_alias(self, args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) > 1 and normalize_whitespace(args[-2]) in ('WITH NAME', 'AS'):
            return (args[:-2], args[-1])
        return (args, None)

    @run_keyword_variant(resolve=0)
    def import_variables(self, path, *args):
        if False:
            for i in range(10):
                print('nop')
        'Imports a variable file with the given path and optional arguments.\n\n        Variables imported with this keyword are set into the test suite scope\n        similarly when importing them in the Setting table using the Variables\n        setting. These variables override possible existing variables with\n        the same names. This functionality can thus be used to import new\n        variables, for example, for each test in a test suite.\n\n        The given path must be absolute or found from\n        [http://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html##module-search-path|search path].\n        Forward slashes can be used as path separator regardless\n        the operating system.\n\n        Examples:\n        | Import Variables | ${CURDIR}/variables.py   |      |      |\n        | Import Variables | ${CURDIR}/../vars/env.py | arg1 | arg2 |\n        | Import Variables | file_from_pythonpath.py  |      |      |\n        '
        try:
            self._namespace.import_variables(path, list(args), overwrite=True)
        except DataError as err:
            raise RuntimeError(str(err))

    @run_keyword_variant(resolve=0)
    def import_resource(self, path):
        if False:
            i = 10
            return i + 15
        'Imports a resource file with the given path.\n\n        Resources imported with this keyword are set into the test suite scope\n        similarly when importing them in the Setting table using the Resource\n        setting.\n\n        The given path must be absolute or found from\n        [http://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#module-search-path|search path].\n        Forward slashes can be used as path separator regardless\n        the operating system.\n\n        Examples:\n        | Import Resource | ${CURDIR}/resource.txt |\n        | Import Resource | ${CURDIR}/../resources/resource.html |\n        | Import Resource | found_from_pythonpath.robot |\n        '
        try:
            self._namespace.import_resource(path)
        except DataError as err:
            raise RuntimeError(str(err))

    def set_library_search_order(self, *search_order):
        if False:
            for i in range(10):
                print('nop')
        'Sets the resolution order to use when a name matches multiple keywords.\n\n        The library search order is used to resolve conflicts when a keyword name\n        that is used matches multiple keyword implementations. The first library\n        (or resource, see below) containing the keyword is selected and that\n        keyword implementation used. If the keyword is not found from any library\n        (or resource), execution fails the same way as when the search order is\n        not set.\n\n        When this keyword is used, there is no need to use the long\n        ``LibraryName.Keyword Name`` notation.  For example, instead of\n        having\n\n        | MyLibrary.Keyword | arg |\n        | MyLibrary.Another Keyword |\n        | MyLibrary.Keyword | xxx |\n\n        you can have\n\n        | Set Library Search Order | MyLibrary |\n        | Keyword | arg |\n        | Another Keyword |\n        | Keyword | xxx |\n\n        This keyword can be used also to set the order of keywords in different\n        resource files. In this case resource names must be given without paths\n        or extensions like:\n\n        | Set Library Search Order | resource | another_resource |\n\n        *NOTE:*\n        - The search order is valid only in the suite where this keyword is used.\n        - Keywords in resources always have higher priority than\n          keywords in libraries regardless the search order.\n        - The old order is returned and can be used to reset the search order later.\n        - Calling this keyword without arguments removes possible search order.\n        - Library and resource names in the search order are both case and space\n          insensitive.\n        '
        return self._namespace.set_search_order(search_order)

    def keyword_should_exist(self, name, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails unless the given keyword exists in the current scope.\n\n        Fails also if there is more than one keyword with the same name.\n        Works both with the short name (e.g. ``Log``) and the full name\n        (e.g. ``BuiltIn.Log``).\n\n        The default error message can be overridden with the ``msg`` argument.\n\n        See also `Variable Should Exist`.\n        '
        try:
            runner = self._namespace.get_runner(name, recommend_on_failure=False)
        except DataError as error:
            raise AssertionError(msg or error.message)
        if isinstance(runner, UserErrorHandler):
            raise AssertionError(msg or runner.error.message)

    def get_time(self, format='timestamp', time_='NOW'):
        if False:
            while True:
                i = 10
        "Returns the given time in the requested format.\n\n        *NOTE:* DateTime library contains much more flexible keywords for\n        getting the current date and time and for date and time handling in\n        general.\n\n        How time is returned is determined based on the given ``format``\n        string as follows. Note that all checks are case-insensitive.\n\n        1) If ``format`` contains the word ``epoch``, the time is returned\n           in seconds after the UNIX epoch (1970-01-01 00:00:00 UTC).\n           The return value is always an integer.\n\n        2) If ``format`` contains any of the words ``year``, ``month``,\n           ``day``, ``hour``, ``min``, or ``sec``, only the selected parts are\n           returned. The order of the returned parts is always the one\n           in the previous sentence and the order of words in ``format``\n           is not significant. The parts are returned as zero-padded\n           strings (e.g. May -> ``05``).\n\n        3) Otherwise (and by default) the time is returned as a\n           timestamp string in the format ``2006-02-24 15:08:31``.\n\n        By default this keyword returns the current local time, but\n        that can be altered using ``time`` argument as explained below.\n        Note that all checks involving strings are case-insensitive.\n\n        1) If ``time`` is a number, or a string that can be converted to\n           a number, it is interpreted as seconds since the UNIX epoch.\n           This documentation was originally written about 1177654467\n           seconds after the epoch.\n\n        2) If ``time`` is a timestamp, that time will be used. Valid\n           timestamp formats are ``YYYY-MM-DD hh:mm:ss`` and\n           ``YYYYMMDD hhmmss``.\n\n        3) If ``time`` is equal to ``NOW`` (default), the current local\n           time is used.\n\n        4) If ``time`` is equal to ``UTC``, the current time in\n           [http://en.wikipedia.org/wiki/Coordinated_Universal_Time|UTC]\n           is used.\n\n        5) If ``time`` is in the format like ``NOW - 1 day`` or ``UTC + 1 hour\n           30 min``, the current local/UTC time plus/minus the time\n           specified with the time string is used. The time string format\n           is described in an appendix of Robot Framework User Guide.\n\n        Examples (expecting the current local time is 2006-03-29 15:06:21):\n        | ${time} = | Get Time |             |  |  |\n        | ${secs} = | Get Time | epoch       |  |  |\n        | ${year} = | Get Time | return year |  |  |\n        | ${yyyy}   | ${mm}    | ${dd} =     | Get Time | year,month,day |\n        | @{time} = | Get Time | year month day hour min sec |  |  |\n        | ${y}      | ${s} =   | Get Time    | seconds and year |  |\n        =>\n        | ${time} = '2006-03-29 15:06:21'\n        | ${secs} = 1143637581\n        | ${year} = '2006'\n        | ${yyyy} = '2006', ${mm} = '03', ${dd} = '29'\n        | @{time} = ['2006', '03', '29', '15', '06', '21']\n        | ${y} = '2006'\n        | ${s} = '21'\n\n        Examples (expecting the current local time is 2006-03-29 15:06:21 and\n        UTC time is 2006-03-29 12:06:21):\n        | ${time} = | Get Time |              | 1177654467          | # Time given as epoch seconds        |\n        | ${secs} = | Get Time | sec          | 2007-04-27 09:14:27 | # Time given as a timestamp          |\n        | ${year} = | Get Time | year         | NOW                 | # The local time of execution        |\n        | @{time} = | Get Time | hour min sec | NOW + 1h 2min 3s    | # 1h 2min 3s added to the local time |\n        | @{utc} =  | Get Time | hour min sec | UTC                 | # The UTC time of execution          |\n        | ${hour} = | Get Time | hour         | UTC - 1 hour        | # 1h subtracted from the UTC  time   |\n        =>\n        | ${time} = '2007-04-27 09:14:27'\n        | ${secs} = 27\n        | ${year} = '2006'\n        | @{time} = ['16', '08', '24']\n        | @{utc} = ['12', '06', '21']\n        | ${hour} = '11'\n        "
        return get_time(format, parse_time(time_))

    def evaluate(self, expression, modules=None, namespace=None):
        if False:
            i = 10
            return i + 15
        'Evaluates the given expression in Python and returns the result.\n\n        ``expression`` is evaluated in Python as explained in the\n        `Evaluating expressions` section.\n\n        ``modules`` argument can be used to specify a comma separated\n        list of Python modules to be imported and added to the evaluation\n        namespace.\n\n        ``namespace`` argument can be used to pass a custom evaluation\n        namespace as a dictionary. Possible ``modules`` are added to this\n        namespace.\n\n        Variables used like ``${variable}`` are replaced in the expression\n        before evaluation. Variables are also available in the evaluation\n        namespace and can be accessed using the special ``$variable`` syntax\n        as explained in the `Evaluating expressions` section.\n\n        Starting from Robot Framework 3.2, modules used in the expression are\n        imported automatically. There are, however, two cases where they need to\n        be explicitly specified using the ``modules`` argument:\n\n        - When nested modules like ``rootmod.submod`` are implemented so that\n          the root module does not automatically import sub modules. This is\n          illustrated by the ``selenium.webdriver`` example below.\n\n        - When using a module in the expression part of a list comprehension.\n          This is illustrated by the ``json`` example below.\n\n        Examples (expecting ``${result}`` is number 3.14):\n        | ${status} =  | Evaluate | 0 < ${result} < 10 | # Would also work with string \'3.14\' |\n        | ${status} =  | Evaluate | 0 < $result < 10   | # Using variable itself, not string representation |\n        | ${random} =  | Evaluate | random.randint(0, sys.maxsize) |\n        | ${options} = | Evaluate | selenium.webdriver.ChromeOptions() | modules=selenium.webdriver |\n        | ${items} =   | Evaluate | [json.loads(item) for item in (\'1\', \'"b"\')] | modules=json |\n        | ${ns} =      | Create Dictionary | x=${4}    | y=${2}              |\n        | ${result} =  | Evaluate | x*10 + y           | namespace=${ns}     |\n        =>\n        | ${status} = True\n        | ${random} = <random integer>\n        | ${options} = ChromeOptions instance\n        | ${items} = [1, \'b\']\n        | ${result} = 42\n\n        *NOTE*: Prior to Robot Framework 3.2 using ``modules=rootmod.submod``\n        was not enough to make the root module itself available in the\n        evaluation namespace. It needed to be taken into use explicitly like\n        ``modules=rootmod, rootmod.submod``.\n        '
        try:
            return evaluate_expression(expression, self._variables.current, modules, namespace)
        except DataError as err:
            raise RuntimeError(err.message)

    def call_method(self, object, method_name, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Calls the named method of the given object with the provided arguments.\n\n        The possible return value from the method is returned and can be\n        assigned to a variable. Keyword fails both if the object does not have\n        a method with the given name or if executing the method raises an\n        exception.\n\n        Possible equal signs in arguments must be escaped with a backslash\n        like ``\\=``.\n\n        Examples:\n        | Call Method      | ${hashtable} | put          | myname  | myvalue |\n        | ${isempty} =     | Call Method  | ${hashtable} | isEmpty |         |\n        | Should Not Be True | ${isempty} |              |         |         |\n        | ${value} =       | Call Method  | ${hashtable} | get     | myname  |\n        | Should Be Equal  | ${value}     | myvalue      |         |         |\n        | Call Method      | ${object}    | kwargs    | name=value | foo=bar |\n        | Call Method      | ${object}    | positional   | escaped\\=equals  |\n        '
        try:
            method = getattr(object, method_name)
        except AttributeError:
            raise RuntimeError(f"{type(object).__name__} object does not have method '{method_name}'.")
        try:
            return method(*args, **kwargs)
        except Exception as err:
            msg = get_error_message()
            raise RuntimeError(f"Calling method '{method_name}' failed: {msg}") from err

    def regexp_escape(self, *patterns):
        if False:
            for i in range(10):
                print('nop')
        "Returns each argument string escaped for use as a regular expression.\n\n        This keyword can be used to escape strings to be used with\n        `Should Match Regexp` and `Should Not Match Regexp` keywords.\n\n        Escaping is done with Python's ``re.escape()`` function.\n\n        Examples:\n        | ${escaped} = | Regexp Escape | ${original} |\n        | @{strings} = | Regexp Escape | @{strings}  |\n        "
        if len(patterns) == 0:
            return ''
        if len(patterns) == 1:
            return re.escape(patterns[0])
        return [re.escape(p) for p in patterns]

    def set_test_message(self, message, append=False):
        if False:
            i = 10
            return i + 15
        'Sets message for the current test case.\n\n        If the optional ``append`` argument is given a true value (see `Boolean\n        arguments`), the given ``message`` is added after the possible earlier\n        message by joining the messages with a space.\n\n        In test teardown this keyword can alter the possible failure message,\n        but otherwise failures override messages set by this keyword. Notice\n        that in teardown the message is available as a built-in variable\n        ``${TEST MESSAGE}``.\n\n        It is possible to use HTML format in the message by starting the message\n        with ``*HTML*``.\n\n        Examples:\n        | Set Test Message | My message           |                          |\n        | Set Test Message | is continued.        | append=yes               |\n        | Should Be Equal  | ${TEST MESSAGE}      | My message is continued. |\n        | Set Test Message | `*`HTML`*` <b>Hello!</b> |                      |\n\n        This keyword can not be used in suite setup or suite teardown.\n        '
        test = self._context.test
        if not test:
            raise RuntimeError("'Set Test Message' keyword cannot be used in suite setup or teardown.")
        test.message = self._get_new_text(test.message, message, append, handle_html=True)
        if self._context.in_test_teardown:
            self._variables.set_test('${TEST_MESSAGE}', test.message)
        (message, level) = self._get_logged_test_message_and_level(test.message)
        self.log(f'Set test message to:\n{message}', level)

    def _get_new_text(self, old, new, append, handle_html=False):
        if False:
            print('Hello World!')
        if not is_string(new):
            new = str(new)
        if not (is_truthy(append) and old):
            return new
        if handle_html:
            if new.startswith('*HTML*'):
                new = new[6:].lstrip()
                if not old.startswith('*HTML*'):
                    old = f'*HTML* {html_escape(old)}'
            elif old.startswith('*HTML*'):
                new = html_escape(new)
        return f'{old} {new}'

    def _get_logged_test_message_and_level(self, message):
        if False:
            print('Hello World!')
        if message.startswith('*HTML*'):
            return (message[6:].lstrip(), 'HTML')
        return (message, 'INFO')

    def set_test_documentation(self, doc, append=False):
        if False:
            return 10
        'Sets documentation for the current test case.\n\n        By default the possible existing documentation is overwritten, but\n        this can be changed using the optional ``append`` argument similarly\n        as with `Set Test Message` keyword.\n\n        The current test documentation is available as a built-in variable\n        ``${TEST DOCUMENTATION}``. This keyword can not be used in suite\n        setup or suite teardown.\n        '
        test = self._context.test
        if not test:
            raise RuntimeError("'Set Test Documentation' keyword cannot be used in suite setup or teardown.")
        test.doc = self._get_new_text(test.doc, doc, append)
        self._variables.set_test('${TEST_DOCUMENTATION}', test.doc)
        self.log(f'Set test documentation to:\n{test.doc}')

    def set_suite_documentation(self, doc, append=False, top=False):
        if False:
            print('Hello World!')
        'Sets documentation for the current test suite.\n\n        By default, the possible existing documentation is overwritten, but\n        this can be changed using the optional ``append`` argument similarly\n        as with `Set Test Message` keyword.\n\n        This keyword sets the documentation of the current suite by default.\n        If the optional ``top`` argument is given a true value (see `Boolean\n        arguments`), the documentation of the top level suite is altered\n        instead.\n\n        The documentation of the current suite is available as a built-in\n        variable ``${SUITE DOCUMENTATION}``.\n        '
        suite = self._get_context(top).suite
        suite.doc = self._get_new_text(suite.doc, doc, append)
        self._variables.set_suite('${SUITE_DOCUMENTATION}', suite.doc, top)
        self.log(f'Set suite documentation to:\n{suite.doc}')

    def set_suite_metadata(self, name, value, append=False, top=False):
        if False:
            i = 10
            return i + 15
        'Sets metadata for the current test suite.\n\n        By default, possible existing metadata values are overwritten, but\n        this can be changed using the optional ``append`` argument similarly\n        as with `Set Test Message` keyword.\n\n        This keyword sets the metadata of the current suite by default.\n        If the optional ``top`` argument is given a true value (see `Boolean\n        arguments`), the metadata of the top level suite is altered instead.\n\n        The metadata of the current suite is available as a built-in variable\n        ``${SUITE METADATA}`` in a Python dictionary. Notice that modifying this\n        variable directly has no effect on the actual metadata the suite has.\n        '
        if not is_string(name):
            name = str(name)
        metadata = self._get_context(top).suite.metadata
        original = metadata.get(name, '')
        metadata[name] = self._get_new_text(original, value, append)
        self._variables.set_suite('${SUITE_METADATA}', metadata.copy(), top)
        self.log(f"Set suite metadata '{name}' to value '{metadata[name]}'.")

    def set_tags(self, *tags):
        if False:
            while True:
                i = 10
        'Adds given ``tags`` for the current test or all tests in a suite.\n\n        When this keyword is used inside a test case, that test gets\n        the specified tags and other tests are not affected.\n\n        If this keyword is used in a suite setup, all test cases in\n        that suite, recursively, gets the given tags. It is a failure\n        to use this keyword in a suite teardown.\n\n        The current tags are available as a built-in variable ``@{TEST TAGS}``.\n\n        See `Remove Tags` if you want to remove certain tags and `Fail` if\n        you want to fail the test case after setting and/or removing tags.\n        '
        ctx = self._context
        if ctx.test:
            ctx.test.tags.add(tags)
            ctx.variables.set_test('@{TEST_TAGS}', list(ctx.test.tags))
        elif not ctx.in_suite_teardown:
            ctx.suite.set_tags(tags, persist=True)
        else:
            raise RuntimeError("'Set Tags' cannot be used in suite teardown.")
        self.log(f'Set tag{s(tags)} {seq2str(tags)}.')

    def remove_tags(self, *tags):
        if False:
            for i in range(10):
                print('nop')
        'Removes given ``tags`` from the current test or all tests in a suite.\n\n        Tags can be given exactly or using a pattern with ``*``, ``?`` and\n        ``[chars]`` acting as wildcards. See the `Glob patterns` section\n        for more information.\n\n        This keyword can affect either one test case or all test cases in a\n        test suite similarly as `Set Tags` keyword.\n\n        The current tags are available as a built-in variable ``@{TEST TAGS}``.\n\n        Example:\n        | Remove Tags | mytag | something-* | ?ython |\n\n        See `Set Tags` if you want to add certain tags and `Fail` if you want\n        to fail the test case after setting and/or removing tags.\n        '
        ctx = self._context
        if ctx.test:
            ctx.test.tags.remove(tags)
            ctx.variables.set_test('@{TEST_TAGS}', list(ctx.test.tags))
        elif not ctx.in_suite_teardown:
            ctx.suite.set_tags(remove=tags, persist=True)
        else:
            raise RuntimeError("'Remove Tags' cannot be used in suite teardown.")
        self.log(f'Removed tag{s(tags)} {seq2str(tags)}.')

    def get_library_instance(self, name=None, all=False):
        if False:
            return 10
        'Returns the currently active instance of the specified library.\n\n        This keyword makes it easy for libraries to interact with\n        other libraries that have state. This is illustrated by\n        the Python example below:\n\n        | from robot.libraries.BuiltIn import BuiltIn\n        |\n        | def title_should_start_with(expected):\n        |     seleniumlib = BuiltIn().get_library_instance(\'SeleniumLibrary\')\n        |     title = seleniumlib.get_title()\n        |     if not title.startswith(expected):\n        |         raise AssertionError(f"Title \'{title}\' did not start with \'{expected}\'.")\n\n        It is also possible to use this keyword in the test data and\n        pass the returned library instance to another keyword. If a\n        library is imported with a custom name, the ``name`` used to get\n        the instance must be that name and not the original library name.\n\n        If the optional argument ``all`` is given a true value, then a\n        dictionary mapping all library names to instances will be returned.\n\n        Example:\n        | &{all libs} = | Get library instance | all=True |\n        '
        if all:
            return self._namespace.get_library_instances()
        try:
            return self._namespace.get_library_instance(name)
        except DataError as err:
            raise RuntimeError(str(err))

class BuiltIn(_Verify, _Converter, _Variables, _RunKeyword, _Control, _Misc):
    """An always available standard library with often needed keywords.

    ``BuiltIn`` is Robot Framework's standard library that provides a set
    of generic keywords needed often. It is imported automatically and
    thus always available. The provided keywords can be used, for example,
    for verifications (e.g. `Should Be Equal`, `Should Contain`),
    conversions (e.g. `Convert To Integer`) and for various other purposes
    (e.g. `Log`, `Sleep`, `Run Keyword If`, `Set Global Variable`).

    == Table of contents ==

    %TOC%

    = HTML error messages =

    Many of the keywords accept an optional error message to use if the keyword
    fails, and it is possible to use HTML in these messages by prefixing them
    with ``*HTML*``. See `Fail` keyword for a usage example. Notice that using
    HTML in messages is not limited to BuiltIn library but works with any
    error message.

    = Evaluating expressions =

    Many keywords, such as `Evaluate`, `Run Keyword If` and `Should Be True`,
    accept an expression that is evaluated in Python.

    == Evaluation namespace ==

    Expressions are evaluated using Python's
    [http://docs.python.org/library/functions.html#eval|eval] function so
    that all Python built-ins like ``len()`` and ``int()`` are available.
    In addition to that, all unrecognized variables are considered to be
    modules that are automatically imported. It is possible to use all
    available Python modules, including the standard modules and the installed
    third party modules.

    Examples:
    | `Should Be True`    len('${result}') > 3
    | `Run Keyword If`    os.sep == '/'    Non-Windows Keyword
    | ${version} =    `Evaluate`    robot.__version__

    `Evaluate` also allows configuring the execution namespace with a custom
    namespace and with custom modules to be imported. The latter functionality
    is useful in special cases where the automatic module import does not work
    such as when using nested modules like ``rootmod.submod`` or list
    comprehensions. See the documentation of the `Evaluate` keyword for mode
    details.

    == Variables in expressions ==

    When a variable is used in the expressing using the normal ``${variable}``
    syntax, its value is replaced before the expression is evaluated. This
    means that the value used in the expression will be the string
    representation of the variable value, not the variable value itself.
    This is not a problem with numbers and other objects that have a string
    representation that can be evaluated directly, but with other objects
    the behavior depends on the string representation. Most importantly,
    strings must always be quoted, and if they can contain newlines, they must
    be triple quoted.

    Examples:
    | `Should Be True`    ${rc} < 10                   Return code greater than 10
    | `Run Keyword If`    '${status}' == 'PASS'        Log    Passed
    | `Run Keyword If`    'FAIL' in '''${output}'''    Log    Output contains FAIL

    Actual variables values are also available in the evaluation namespace.
    They can be accessed using special variable syntax without the curly
    braces like ``$variable``. These variables should never be quoted.

    Examples:
    | `Should Be True`    $rc < 10             Return code greater than 10
    | `Run Keyword If`    $status == 'PASS'    `Log`    Passed
    | `Run Keyword If`    'FAIL' in $output    `Log`    Output contains FAIL
    | `Should Be True`    len($result) > 1 and $result[1] == 'OK'
    | `Should Be True`    $result is not None

    Using the ``$variable`` syntax slows down expression evaluation a little.
    This should not typically matter, but should be taken into account if
    complex expressions are evaluated often and there are strict time
    constrains.

    Notice that instead of creating complicated expressions, it is often better
    to move the logic into a library. That eases maintenance and can also
    enhance execution speed.

    = Using variables with keywords creating or accessing variables =

    This library has special keywords `Set Global Variable`, `Set Suite Variable`,
    `Set Test Variable` and `Set Local Variable` for creating variables in
    different scopes. These keywords take the variable name and its value as
    arguments. The name can be given using the normal ``${variable}`` syntax or
    in escaped format either like ``$variable`` or ``\\${variable}``. For example,
    these are typically equivalent and create new suite level variable
    ``${name}`` with value ``value``:

    | Set Suite Variable    ${name}     value
    | Set Suite Variable    $name       value
    | Set Suite Variable    \\${name}    value

    A problem with using the normal ``${variable}`` syntax is that these
    keywords cannot easily know is the idea to create a variable with exactly
    that name or does that variable actually contain the name of the variable
    to create. If the variable does not initially exist, it will always be
    created. If it exists and its value is a variable name either in the normal
    or in the escaped syntax, variable with _that_ name is created instead.
    For example, if ``${name}`` variable would exist and contain value
    ``$example``, these examples would create different variables:

    | Set Suite Variable    ${name}     value    # Creates ${example}.
    | Set Suite Variable    $name       value    # Creates ${name}.
    | Set Suite Variable    \\${name}    value    # Creates ${name}.

    Because the behavior when using the normal ``${variable}`` syntax depends
    on the possible existing value of the variable, it is *highly recommended
    to use the escaped ``$variable`` or ``\\${variable}`` format instead*.

    This same problem occurs also with special keywords for accessing variables
    `Get Variable Value`, `Variable Should Exist` and `Variable Should Not Exist`.

    *NOTE:* It is recommended to use the ``VAR`` syntax introduced in Robot
    Framework 7.0 for creating variables in different scopes instead of the
    `Set Global/Suite/Test/Local Variable` keywords. It makes creating variables
    uniform and avoids all the problems discussed above.

    = Boolean arguments =

    Some keywords accept arguments that are handled as Boolean values true or
    false. If such an argument is given as a string, it is considered false if
    it is an empty string or equal to ``FALSE``, ``NONE``, ``NO``, ``OFF`` or
    ``0``, case-insensitively. Keywords verifying something that allow dropping
    actual and expected values from the possible error message also consider
    string ``no values`` to be false. Other strings are considered true unless
    the keyword documentation explicitly states otherwise, and other argument
    types are tested using the same
    [http://docs.python.org/library/stdtypes.html#truth|rules as in Python].

    True examples:
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=True         # Strings are generally true.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=yes          # Same as the above.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=${TRUE}      # Python ``True`` is true.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=${42}        # Numbers other than 0 are true.

    False examples:
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=False        # String ``false`` is false.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=no           # Also string ``no`` is false.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=${EMPTY}     # Empty string is false.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=${FALSE}     # Python ``False`` is false.
    | `Should Be Equal`    ${x}    ${y}    Custom error    values=no values    # ``no values`` works with ``values`` argument

    = Pattern matching =

    Many keywords accept arguments as either glob or regular expression patterns.

    == Glob patterns ==

    Some keywords, for example `Should Match`, support so called
    [http://en.wikipedia.org/wiki/Glob_(programming)|glob patterns] where:

    | ``*``        | matches any string, even an empty string                |
    | ``?``        | matches any single character                            |
    | ``[chars]``  | matches one character in the bracket                    |
    | ``[!chars]`` | matches one character not in the bracket                |
    | ``[a-z]``    | matches one character from the range in the bracket     |
    | ``[!a-z]``   | matches one character not from the range in the bracket |

    Unlike with glob patterns normally, path separator characters ``/`` and
    ``\\`` and the newline character ``\\n`` are matches by the above
    wildcards.

    == Regular expressions ==

    Some keywords, for example `Should Match Regexp`, support
    [http://en.wikipedia.org/wiki/Regular_expression|regular expressions]
    that are more powerful but also more complicated that glob patterns.
    The regular expression support is implemented using Python's
    [http://docs.python.org/library/re.html|re module] and its documentation
    should be consulted for more information about the syntax.

    Because the backslash character (``\\``) is an escape character in
    Robot Framework test data, possible backslash characters in regular
    expressions need to be escaped with another backslash like ``\\\\d\\\\w+``.
    Strings that may contain special characters but should be handled
    as literal strings, can be escaped with the `Regexp Escape` keyword.

    = Multiline string comparison =

    `Should Be Equal` and `Should Be Equal As Strings` report the failures using
    [http://en.wikipedia.org/wiki/Diff_utility#Unified_format|unified diff
    format] if both strings have more than two lines.

    Example:
    | ${first} =     `Catenate`    SEPARATOR=\\n    Not in second    Same    Differs    Same
    | ${second} =    `Catenate`    SEPARATOR=\\n    Same    Differs2    Same    Not in first
    | `Should Be Equal`    ${first}    ${second}

    Results in the following error message:

    | Multiline strings are different:
    | --- first
    | +++ second
    | @@ -1,4 +1,4 @@
    | -Not in second
    |  Same
    | -Differs
    | +Differs2
    |  Same
    | +Not in first

    = String representations =

    Several keywords log values explicitly (e.g. `Log`) or implicitly (e.g.
    `Should Be Equal` when there are failures). By default, keywords log values
    using human-readable string representation, which means that strings
    like ``Hello`` and numbers like ``42`` are logged as-is. Most of the time
    this is the desired behavior, but there are some problems as well:

    - It is not possible to see difference between different objects that
      have the same string representation like string ``42`` and integer ``42``.
      `Should Be Equal` and some other keywords add the type information to
      the error message in these cases, though.

    - Non-printable characters such as the null byte are not visible.

    - Trailing whitespace is not visible.

    - Different newlines (``\\r\\n`` on Windows, ``\\n`` elsewhere) cannot
      be separated from each others.

    - There are several Unicode characters that are different but look the
      same. One example is the Latin ``a`` (``\\u0061``) and the Cyrillic
      ``а`` (``\\u0430``). Error messages like ``a != а`` are not very helpful.

    - Some Unicode characters can be represented using
      [https://en.wikipedia.org/wiki/Unicode_equivalence|different forms].
      For example, ``ä`` can be represented either as a single code point
      ``\\u00e4`` or using two combined code points ``\\u0061`` and ``\\u0308``.
      Such forms are considered canonically equivalent, but strings
      containing them are not considered equal when compared in Python. Error
      messages like ``ä != ä`` are not that helpful either.

    - Containers such as lists and dictionaries are formatted into a single
      line making it hard to see individual items they contain.

    To overcome the above problems, some keywords such as `Log` and
    `Should Be Equal` have an optional ``formatter`` argument that can be
    used to configure the string representation. The supported values are
    ``str`` (default), ``repr``, and ``ascii`` that work similarly as
    [https://docs.python.org/library/functions.html|Python built-in functions]
    with same names. More detailed semantics are explained below.

    == str ==

    Use the human-readable string representation. Equivalent to using ``str()``
    in Python. This is the default.

    == repr ==

    Use the machine-readable string representation. Similar to using ``repr()``
    in Python, which means that strings like ``Hello`` are logged like
    ``'Hello'``, newlines and non-printable characters are escaped like ``\\n``
    and ``\\x00``, and so on. Non-ASCII characters are shown as-is like ``ä``.

    In this mode bigger lists, dictionaries and other containers are
    pretty-printed so that there is one item per row.

    == ascii ==

    Same as using ``ascii()`` in Python. Similar to using ``repr`` explained above
    but with the following differences:

    - Non-ASCII characters are escaped like ``\\xe4`` instead of
      showing them as-is like ``ä``. This makes it easier to see differences
      between Unicode characters that look the same but are not equal.
    - Containers are not pretty-printed.
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = get_version()

class RobotNotRunningError(AttributeError):
    """Used when something cannot be done because Robot is not running.

    Based on AttributeError to be backwards compatible with RF < 2.8.5.
    May later be based directly on Exception, so new code should except
    this exception explicitly.
    """
    pass

def register_run_keyword(library, keyword, args_to_process=0, deprecation_warning=True):
    if False:
        i = 10
        return i + 15
    'Tell Robot Framework that this keyword runs other keywords internally.\n\n    *NOTE:* This API will change in the future. For more information see\n    https://github.com/robotframework/robotframework/issues/2190.\n\n    :param library: Name of the library the keyword belongs to.\n    :param keyword: Name of the keyword itself.\n    :param args_to_process: How many arguments to process normally before\n        passing them to the keyword. Other arguments are not touched at all.\n    :param deprecation_warning: Set to ``False```to avoid the warning.\n\n    Registered keywords are handled specially by Robot so that:\n\n    - Their arguments are not resolved normally (use ``args_to_process``\n      to control that). This basically means not replacing variables or\n      handling escapes.\n    - They are not stopped by timeouts.\n    - If there are conflicts with keyword names, these keywords have\n      *lower* precedence than other keywords.\n\n    Main use cases are:\n\n    - Library keyword is using `BuiltIn.run_keyword` internally to execute other\n      keywords. Registering the caller as a "run keyword variant" avoids variables\n      and escapes in arguments being resolved multiple times. All arguments passed\n      to `run_keyword` can and should be left unresolved.\n    - Keyword has some need to not resolve variables in arguments. This way\n      variable values are not logged anywhere by Robot automatically.\n\n    As mentioned above, this API will likely be reimplemented in the future\n    or there could be new API for library keywords to execute other keywords.\n    External libraries can nevertheless use this API if they really need it and\n    are aware of the possible breaking changes in the future.\n\n    Examples::\n\n        from robot.libraries.BuiltIn import BuiltIn, register_run_keyword\n\n        def my_run_keyword(name, *args):\n            # do something\n            return BuiltIn().run_keyword(name, *args)\n\n        register_run_keyword(__name__, \'My Run Keyword\')\n\n        -------------\n\n        from robot.libraries.BuiltIn import BuiltIn, register_run_keyword\n\n        class MyLibrary:\n            def my_run_keyword_if(self, expression, name, *args):\n                # Do something\n                if self._is_true(expression):\n                    return BuiltIn().run_keyword(name, *args)\n\n        # Process one argument normally to get `expression` resolved.\n        register_run_keyword(\'MyLibrary\', \'my_run_keyword_if\', args_to_process=1)\n    '
    RUN_KW_REGISTER.register_run_keyword(library, keyword, args_to_process, deprecation_warning)