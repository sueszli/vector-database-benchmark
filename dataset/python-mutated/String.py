import os
import re
from fnmatch import fnmatchcase
from random import randint
from string import ascii_lowercase, ascii_uppercase, digits
from robot.api import logger
from robot.api.deco import keyword
from robot.utils import FileReader, is_bytes, is_string, is_truthy, parse_re_flags, safe_str, type_name
from robot.version import get_version

class String:
    """A library for string manipulation and verification.

    ``String`` is Robot Framework's standard library for manipulating
    strings (e.g. `Replace String Using Regexp`, `Split To Lines`) and
    verifying their contents (e.g. `Should Be String`).

    Following keywords from ``BuiltIn`` library can also be used with strings:

    - `Catenate`
    - `Get Length`
    - `Length Should Be`
    - `Should (Not) Be Empty`
    - `Should (Not) Be Equal (As Strings/Integers/Numbers)`
    - `Should (Not) Match (Regexp)`
    - `Should (Not) Contain`
    - `Should (Not) Start With`
    - `Should (Not) End With`
    - `Convert To String`
    - `Convert To Bytes`
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = get_version()

    def convert_to_lower_case(self, string):
        if False:
            return 10
        "Converts string to lower case.\n\n        Uses Python's standard\n        [https://docs.python.org/library/stdtypes.html#str.lower|lower()]\n        method.\n\n        Examples:\n        | ${str1} = | Convert To Lower Case | ABC |\n        | ${str2} = | Convert To Lower Case | 1A2c3D |\n        | Should Be Equal | ${str1} | abc |\n        | Should Be Equal | ${str2} | 1a2c3d |\n        "
        return string.lower()

    def convert_to_upper_case(self, string):
        if False:
            for i in range(10):
                print('nop')
        "Converts string to upper case.\n\n        Uses Python's standard\n        [https://docs.python.org/library/stdtypes.html#str.upper|upper()]\n        method.\n\n        Examples:\n        | ${str1} = | Convert To Upper Case | abc |\n        | ${str2} = | Convert To Upper Case | 1a2C3d |\n        | Should Be Equal | ${str1} | ABC |\n        | Should Be Equal | ${str2} | 1A2C3D |\n        "
        return string.upper()

    @keyword(types=None)
    def convert_to_title_case(self, string, exclude=None):
        if False:
            print('Hello World!')
        'Converts string to title case.\n\n        Uses the following algorithm:\n\n        - Split the string to words from whitespace characters (spaces,\n          newlines, etc.).\n        - Exclude words that are not all lower case. This preserves,\n          for example, "OK" and "iPhone".\n        - Exclude also words listed in the optional ``exclude`` argument.\n        - Title case the first alphabetical character of each word that has\n          not been excluded.\n        - Join all words together so that original whitespace is preserved.\n\n        Explicitly excluded words can be given as a list or as a string with\n        words separated by a comma and an optional space. Excluded words are\n        actually considered to be regular expression patterns, so it is\n        possible to use something like "example[.!?]?" to match the word\n        "example" on it own and also if followed by ".", "!" or "?".\n        See `BuiltIn.Should Match Regexp` for more information about Python\n        regular expression syntax in general and how to use it in Robot\n        Framework data in particular.\n\n        Examples:\n        | ${str1} = | Convert To Title Case | hello, world!     |\n        | ${str2} = | Convert To Title Case | it\'s an OK iPhone | exclude=a, an, the |\n        | ${str3} = | Convert To Title Case | distance is 1 km. | exclude=is, km.? |\n        | Should Be Equal | ${str1} | Hello, World! |\n        | Should Be Equal | ${str2} | It\'s an OK iPhone |\n        | Should Be Equal | ${str3} | Distance is 1 km. |\n\n        The reason this keyword does not use Python\'s standard\n        [https://docs.python.org/library/stdtypes.html#str.title|title()]\n        method is that it can yield undesired results, for example, if\n        strings contain upper case letters or special characters like\n        apostrophes. It would, for example, convert "it\'s an OK iPhone"\n        to "It\'S An Ok Iphone".\n\n        New in Robot Framework 3.2.\n        '
        if not is_string(string):
            raise TypeError('This keyword works only with Unicode strings.')
        if is_string(exclude):
            exclude = [e.strip() for e in exclude.split(',')]
        elif not exclude:
            exclude = []
        exclude = [re.compile('^%s$' % e) for e in exclude]

        def title(word):
            if False:
                while True:
                    i = 10
            if any((e.match(word) for e in exclude)) or not word.islower():
                return word
            for (index, char) in enumerate(word):
                if char.isalpha():
                    return word[:index] + word[index].title() + word[index + 1:]
            return word
        tokens = re.split('(\\s+)', string, flags=re.UNICODE)
        return ''.join((title(token) for token in tokens))

    def encode_string_to_bytes(self, string, encoding, errors='strict'):
        if False:
            return 10
        'Encodes the given Unicode ``string`` to bytes using the given ``encoding``.\n\n        ``errors`` argument controls what to do if encoding some characters fails.\n        All values accepted by ``encode`` method in Python are valid, but in\n        practice the following values are most useful:\n\n        - ``strict``: fail if characters cannot be encoded (default)\n        - ``ignore``: ignore characters that cannot be encoded\n        - ``replace``: replace characters that cannot be encoded with\n          a replacement character\n\n        Examples:\n        | ${bytes} = | Encode String To Bytes | ${string} | UTF-8 |\n        | ${bytes} = | Encode String To Bytes | ${string} | ASCII | errors=ignore |\n\n        Use `Convert To Bytes` in ``BuiltIn`` if you want to create bytes based\n        on character or integer sequences. Use `Decode Bytes To String` if you\n        need to convert byte strings to Unicode strings and `Convert To String`\n        in ``BuiltIn`` if you need to convert arbitrary objects to Unicode.\n        '
        return bytes(string.encode(encoding, errors))

    def decode_bytes_to_string(self, bytes, encoding, errors='strict'):
        if False:
            return 10
        'Decodes the given ``bytes`` to a Unicode string using the given ``encoding``.\n\n        ``errors`` argument controls what to do if decoding some bytes fails.\n        All values accepted by ``decode`` method in Python are valid, but in\n        practice the following values are most useful:\n\n        - ``strict``: fail if characters cannot be decoded (default)\n        - ``ignore``: ignore characters that cannot be decoded\n        - ``replace``: replace characters that cannot be decoded with\n          a replacement character\n\n        Examples:\n        | ${string} = | Decode Bytes To String | ${bytes} | UTF-8 |\n        | ${string} = | Decode Bytes To String | ${bytes} | ASCII | errors=ignore |\n\n        Use `Encode String To Bytes` if you need to convert Unicode strings to\n        byte strings, and `Convert To String` in ``BuiltIn`` if you need to\n        convert arbitrary objects to Unicode strings.\n        '
        if is_string(bytes):
            raise TypeError('Cannot decode strings.')
        return bytes.decode(encoding, errors)

    def format_string(self, template, *positional, **named):
        if False:
            return 10
        "Formats a ``template`` using the given ``positional`` and ``named`` arguments.\n\n        The template can be either be a string or an absolute path to\n        an existing file. In the latter case the file is read and its contents\n        are used as the template. If the template file contains non-ASCII\n        characters, it must be encoded using UTF-8.\n\n        The template is formatted using Python's\n        [https://docs.python.org/library/string.html#format-string-syntax|format\n        string syntax]. Placeholders are marked using ``{}`` with possible\n        field name and format specification inside. Literal curly braces\n        can be inserted by doubling them like `{{` and `}}`.\n\n        Examples:\n        | ${to} = | Format String | To: {} <{}>                    | ${user}      | ${email} |\n        | ${to} = | Format String | To: {name} <{email}>           | name=${name} | email=${email} |\n        | ${to} = | Format String | To: {user.name} <{user.email}> | user=${user} |\n        | ${xx} = | Format String | {:*^30}                        | centered     |\n        | ${yy} = | Format String | {0:{width}{base}}              | ${42}        | base=X | width=10 |\n        | ${zz} = | Format String | ${CURDIR}/template.txt         | positional   | named=value |\n\n        New in Robot Framework 3.1.\n        "
        if os.path.isabs(template) and os.path.isfile(template):
            template = template.replace('/', os.sep)
            logger.info('Reading template from file <a href="%s">%s</a>.' % (template, template), html=True)
            with FileReader(template) as reader:
                template = reader.read()
        return template.format(*positional, **named)

    def get_line_count(self, string):
        if False:
            return 10
        'Returns and logs the number of lines in the given string.'
        count = len(string.splitlines())
        logger.info('%d lines' % count)
        return count

    def split_to_lines(self, string, start=0, end=None):
        if False:
            print('Hello World!')
        'Splits the given string to lines.\n\n        It is possible to get only a selection of lines from ``start``\n        to ``end`` so that ``start`` index is inclusive and ``end`` is\n        exclusive. Line numbering starts from 0, and it is possible to\n        use negative indices to refer to lines from the end.\n\n        Lines are returned without the newlines. The number of\n        returned lines is automatically logged.\n\n        Examples:\n        | @{lines} =        | Split To Lines | ${manylines} |    |    |\n        | @{ignore first} = | Split To Lines | ${manylines} | 1  |    |\n        | @{ignore last} =  | Split To Lines | ${manylines} |    | -1 |\n        | @{5th to 10th} =  | Split To Lines | ${manylines} | 4  | 10 |\n        | @{first two} =    | Split To Lines | ${manylines} |    | 1  |\n        | @{last two} =     | Split To Lines | ${manylines} | -2 |    |\n\n        Use `Get Line` if you only need to get a single line.\n        '
        start = self._convert_to_index(start, 'start')
        end = self._convert_to_index(end, 'end')
        lines = string.splitlines()[start:end]
        logger.info('%d lines returned' % len(lines))
        return lines

    def get_line(self, string, line_number):
        if False:
            print('Hello World!')
        'Returns the specified line from the given ``string``.\n\n        Line numbering starts from 0 and it is possible to use\n        negative indices to refer to lines from the end. The line is\n        returned without the newline character.\n\n        Examples:\n        | ${first} =    | Get Line | ${string} | 0  |\n        | ${2nd last} = | Get Line | ${string} | -2 |\n\n        Use `Split To Lines` if all lines are needed.\n        '
        line_number = self._convert_to_integer(line_number, 'line_number')
        return string.splitlines()[line_number]

    def get_lines_containing_string(self, string, pattern, case_insensitive=False):
        if False:
            return 10
        'Returns lines of the given ``string`` that contain the ``pattern``.\n\n        The ``pattern`` is always considered to be a normal string, not a glob\n        or regexp pattern. A line matches if the ``pattern`` is found anywhere\n        on it.\n\n        The match is case-sensitive by default, but giving ``case_insensitive``\n        a true value makes it case-insensitive. The value is considered true\n        if it is a non-empty string that is not equal to ``false``, ``none`` or\n        ``no``. If the value is not a string, its truth value is got directly\n        in Python.\n\n        Lines are returned as one string catenated back together with\n        newlines. Possible trailing newline is never returned. The\n        number of matching lines is automatically logged.\n\n        Examples:\n        | ${lines} = | Get Lines Containing String | ${result} | An example |\n        | ${ret} =   | Get Lines Containing String | ${ret} | FAIL | case-insensitive |\n\n        See `Get Lines Matching Pattern` and `Get Lines Matching Regexp`\n        if you need more complex pattern matching.\n        '
        if is_truthy(case_insensitive):
            pattern = pattern.casefold()
            contains = lambda line: pattern in line.casefold()
        else:
            contains = lambda line: pattern in line
        return self._get_matching_lines(string, contains)

    def get_lines_matching_pattern(self, string, pattern, case_insensitive=False):
        if False:
            i = 10
            return i + 15
        'Returns lines of the given ``string`` that match the ``pattern``.\n\n        The ``pattern`` is a _glob pattern_ where:\n        | ``*``        | matches everything |\n        | ``?``        | matches any single character |\n        | ``[chars]``  | matches any character inside square brackets (e.g. ``[abc]`` matches either ``a``, ``b`` or ``c``) |\n        | ``[!chars]`` | matches any character not inside square brackets |\n\n        A line matches only if it matches the ``pattern`` fully.\n\n        The match is case-sensitive by default, but giving ``case_insensitive``\n        a true value makes it case-insensitive. The value is considered true\n        if it is a non-empty string that is not equal to ``false``, ``none`` or\n        ``no``. If the value is not a string, its truth value is got directly\n        in Python.\n\n        Lines are returned as one string catenated back together with\n        newlines. Possible trailing newline is never returned. The\n        number of matching lines is automatically logged.\n\n        Examples:\n        | ${lines} = | Get Lines Matching Pattern | ${result} | Wild???? example |\n        | ${ret} = | Get Lines Matching Pattern | ${ret} | FAIL: * | case_insensitive=true |\n\n        See `Get Lines Matching Regexp` if you need more complex\n        patterns and `Get Lines Containing String` if searching\n        literal strings is enough.\n        '
        if is_truthy(case_insensitive):
            pattern = pattern.casefold()
            matches = lambda line: fnmatchcase(line.casefold(), pattern)
        else:
            matches = lambda line: fnmatchcase(line, pattern)
        return self._get_matching_lines(string, matches)

    def get_lines_matching_regexp(self, string, pattern, partial_match=False, flags=None):
        if False:
            print('Hello World!')
        'Returns lines of the given ``string`` that match the regexp ``pattern``.\n\n        See `BuiltIn.Should Match Regexp` for more information about\n        Python regular expression syntax in general and how to use it\n        in Robot Framework data in particular.\n\n        Lines match only if they match the pattern fully by default, but\n        partial matching can be enabled by giving the ``partial_match``\n        argument a true value. The value is considered true\n        if it is a non-empty string that is not equal to ``false``, ``none`` or\n        ``no``. If the value is not a string, its truth value is got directly\n        in Python.\n\n        If the pattern is empty, it matches only empty lines by default.\n        When partial matching is enabled, empty pattern matches all lines.\n\n        Possible flags altering how the expression is parsed (e.g. ``re.IGNORECASE``,\n        ``re.VERBOSE``) can be given using the ``flags`` argument (e.g.\n        ``flags=IGNORECASE | VERBOSE``) or embedded to the pattern (e.g.\n        ``(?ix)pattern``).\n\n        Lines are returned as one string concatenated back together with\n        newlines. Possible trailing newline is never returned. The\n        number of matching lines is automatically logged.\n\n        Examples:\n        | ${lines} = | Get Lines Matching Regexp | ${result} | Reg\\\\w{3} example |\n        | ${lines} = | Get Lines Matching Regexp | ${result} | Reg\\\\w{3} example | partial_match=true |\n        | ${ret} =   | Get Lines Matching Regexp | ${ret}    | (?i)FAIL: .* |\n        | ${ret} =   | Get Lines Matching Regexp | ${ret}    | FAIL: .* | flags=IGNORECASE |\n\n        See `Get Lines Matching Pattern` and `Get Lines Containing String` if you\n        do not need the full regular expression powers (and complexity).\n\n        The ``flags`` argument is new in Robot Framework 6.0.\n        '
        if is_truthy(partial_match):
            match = re.compile(pattern, flags=parse_re_flags(flags)).search
        else:
            match = re.compile(pattern + '$', flags=parse_re_flags(flags)).match
        return self._get_matching_lines(string, match)

    def _get_matching_lines(self, string, matches):
        if False:
            return 10
        lines = string.splitlines()
        matching = [line for line in lines if matches(line)]
        logger.info('%d out of %d lines matched' % (len(matching), len(lines)))
        return '\n'.join(matching)

    def get_regexp_matches(self, string, pattern, *groups, flags=None):
        if False:
            i = 10
            return i + 15
        "Returns a list of all non-overlapping matches in the given string.\n\n        ``string`` is the string to find matches from and ``pattern`` is the\n        regular expression. See `BuiltIn.Should Match Regexp` for more\n        information about Python regular expression syntax in general and how\n        to use it in Robot Framework data in particular.\n\n        If no groups are used, the returned list contains full matches. If one\n        group is used, the list contains only contents of that group. If\n        multiple groups are used, the list contains tuples that contain\n        individual group contents. All groups can be given as indexes (starting\n        from 1) and named groups also as names.\n\n        Possible flags altering how the expression is parsed (e.g. ``re.IGNORECASE``,\n        ``re.MULTILINE``) can be given using the ``flags`` argument (e.g.\n        ``flags=IGNORECASE | MULTILINE``) or embedded to the pattern (e.g.\n        ``(?im)pattern``).\n\n        Examples:\n        | ${no match} =    | Get Regexp Matches | the string | xxx     |\n        | ${matches} =     | Get Regexp Matches | the string | t..     |\n        | ${matches} =     | Get Regexp Matches | the string | T..     | flags=IGNORECASE |\n        | ${one group} =   | Get Regexp Matches | the string | t(..)   | 1 |\n        | ${named group} = | Get Regexp Matches | the string | t(?P<name>..) | name |\n        | ${two groups} =  | Get Regexp Matches | the string | t(.)(.) | 1 | 2 |\n        =>\n        | ${no match} = []\n        | ${matches} = ['the', 'tri']\n        | ${one group} = ['he', 'ri']\n        | ${named group} = ['he', 'ri']\n        | ${two groups} = [('h', 'e'), ('r', 'i')]\n\n        The ``flags`` argument is new in Robot Framework 6.0.\n        "
        regexp = re.compile(pattern, flags=parse_re_flags(flags))
        groups = [self._parse_group(g) for g in groups]
        return [m.group(*groups) for m in regexp.finditer(string)]

    def _parse_group(self, group):
        if False:
            i = 10
            return i + 15
        try:
            return int(group)
        except ValueError:
            return group

    def replace_string(self, string, search_for, replace_with, count=-1):
        if False:
            return 10
        'Replaces ``search_for`` in the given ``string`` with ``replace_with``.\n\n        ``search_for`` is used as a literal string. See `Replace String\n        Using Regexp` if more powerful pattern matching is needed.\n        If you need to just remove a string see `Remove String`.\n\n        If the optional argument ``count`` is given, only that many\n        occurrences from left are replaced. Negative ``count`` means\n        that all occurrences are replaced (default behaviour) and zero\n        means that nothing is done.\n\n        A modified version of the string is returned and the original\n        string is not altered.\n\n        Examples:\n        | ${str} =        | Replace String | Hello, world!  | world | tellus   |\n        | Should Be Equal | ${str}         | Hello, tellus! |       |          |\n        | ${str} =        | Replace String | Hello, world!  | l     | ${EMPTY} | count=1 |\n        | Should Be Equal | ${str}         | Helo, world!   |       |          |\n        '
        count = self._convert_to_integer(count, 'count')
        return string.replace(search_for, replace_with, count)

    def replace_string_using_regexp(self, string, pattern, replace_with, count=-1, flags=None):
        if False:
            return 10
        'Replaces ``pattern`` in the given ``string`` with ``replace_with``.\n\n        This keyword is otherwise identical to `Replace String`, but\n        the ``pattern`` to search for is considered to be a regular\n        expression.  See `BuiltIn.Should Match Regexp` for more\n        information about Python regular expression syntax in general\n        and how to use it in Robot Framework data in particular.\n\n        Possible flags altering how the expression is parsed (e.g. ``re.IGNORECASE``,\n        ``re.MULTILINE``) can be given using the ``flags`` argument (e.g.\n        ``flags=IGNORECASE | MULTILINE``) or embedded to the pattern (e.g.\n        ``(?im)pattern``).\n\n        If you need to just remove a string see `Remove String Using Regexp`.\n\n        Examples:\n        | ${str} = | Replace String Using Regexp | ${str} | 20\\\\d\\\\d-\\\\d\\\\d-\\\\d\\\\d | <DATE> |\n        | ${str} = | Replace String Using Regexp | ${str} | (Hello|Hi) | ${EMPTY} | count=1 |\n\n        The ``flags`` argument is new in Robot Framework 6.0.\n        '
        count = self._convert_to_integer(count, 'count')
        if count == 0:
            return string
        return re.sub(pattern, replace_with, string, max(count, 0), flags=parse_re_flags(flags))

    def remove_string(self, string, *removables):
        if False:
            return 10
        'Removes all ``removables`` from the given ``string``.\n\n        ``removables`` are used as literal strings. Each removable will be\n        matched to a temporary string from which preceding removables have\n        been already removed. See second example below.\n\n        Use `Remove String Using Regexp` if more powerful pattern matching is\n        needed. If only a certain number of matches should be removed,\n        `Replace String` or `Replace String Using Regexp` can be used.\n\n        A modified version of the string is returned and the original\n        string is not altered.\n\n        Examples:\n        | ${str} =        | Remove String | Robot Framework | work   |\n        | Should Be Equal | ${str}        | Robot Frame     |\n        | ${str} =        | Remove String | Robot Framework | o | bt |\n        | Should Be Equal | ${str}        | R Framewrk      |\n        '
        for removable in removables:
            string = self.replace_string(string, removable, '')
        return string

    def remove_string_using_regexp(self, string, *patterns, flags=None):
        if False:
            i = 10
            return i + 15
        'Removes ``patterns`` from the given ``string``.\n\n        This keyword is otherwise identical to `Remove String`, but\n        the ``patterns`` to search for are considered to be a regular\n        expression. See `Replace String Using Regexp` for more information\n        about the regular expression syntax. That keyword can also be\n        used if there is a need to remove only a certain number of\n        occurrences.\n\n        Possible flags altering how the expression is parsed (e.g. ``re.IGNORECASE``,\n        ``re.MULTILINE``) can be given using the ``flags`` argument (e.g.\n        ``flags=IGNORECASE | MULTILINE``) or embedded to the pattern (e.g.\n        ``(?im)pattern``).\n\n        The ``flags`` argument is new in Robot Framework 6.0.\n        '
        for pattern in patterns:
            string = self.replace_string_using_regexp(string, pattern, '', flags=flags)
        return string

    @keyword(types=None)
    def split_string(self, string, separator=None, max_split=-1):
        if False:
            i = 10
            return i + 15
        'Splits the ``string`` using ``separator`` as a delimiter string.\n\n        If a ``separator`` is not given, any whitespace string is a\n        separator. In that case also possible consecutive whitespace\n        as well as leading and trailing whitespace is ignored.\n\n        Split words are returned as a list. If the optional\n        ``max_split`` is given, at most ``max_split`` splits are done, and\n        the returned list will have maximum ``max_split + 1`` elements.\n\n        Examples:\n        | @{words} =         | Split String | ${string} |\n        | @{words} =         | Split String | ${string} | ,${SPACE} |\n        | ${pre} | ${post} = | Split String | ${string} | ::    | 1 |\n\n        See `Split String From Right` if you want to start splitting\n        from right, and `Fetch From Left` and `Fetch From Right` if\n        you only want to get first/last part of the string.\n        '
        if separator == '':
            separator = None
        max_split = self._convert_to_integer(max_split, 'max_split')
        return string.split(separator, max_split)

    @keyword(types=None)
    def split_string_from_right(self, string, separator=None, max_split=-1):
        if False:
            print('Hello World!')
        'Splits the ``string`` using ``separator`` starting from right.\n\n        Same as `Split String`, but splitting is started from right. This has\n        an effect only when ``max_split`` is given.\n\n        Examples:\n        | ${first} | ${rest} = | Split String            | ${string} | - | 1 |\n        | ${rest}  | ${last} = | Split String From Right | ${string} | - | 1 |\n        '
        if separator == '':
            separator = None
        max_split = self._convert_to_integer(max_split, 'max_split')
        return string.rsplit(separator, max_split)

    def split_string_to_characters(self, string):
        if False:
            for i in range(10):
                print('nop')
        'Splits the given ``string`` to characters.\n\n        Example:\n        | @{characters} = | Split String To Characters | ${string} |\n        '
        return list(string)

    def fetch_from_left(self, string, marker):
        if False:
            print('Hello World!')
        'Returns contents of the ``string`` before the first occurrence of ``marker``.\n\n        If the ``marker`` is not found, whole string is returned.\n\n        See also `Fetch From Right`, `Split String` and `Split String\n        From Right`.\n        '
        return string.split(marker)[0]

    def fetch_from_right(self, string, marker):
        if False:
            return 10
        'Returns contents of the ``string`` after the last occurrence of ``marker``.\n\n        If the ``marker`` is not found, whole string is returned.\n\n        See also `Fetch From Left`, `Split String` and `Split String\n        From Right`.\n        '
        return string.split(marker)[-1]

    def generate_random_string(self, length=8, chars='[LETTERS][NUMBERS]'):
        if False:
            i = 10
            return i + 15
        'Generates a string with a desired ``length`` from the given ``chars``.\n\n        ``length`` can be given as a number, a string representation of a number,\n        or as a range of numbers, such as ``5-10``. When a range of values is given\n        the range will be selected by random within the range.\n\n        The population sequence ``chars`` contains the characters to use\n        when generating the random string. It can contain any\n        characters, and it is possible to use special markers\n        explained in the table below:\n\n        |  = Marker =   |               = Explanation =                   |\n        | ``[LOWER]``   | Lowercase ASCII characters from ``a`` to ``z``. |\n        | ``[UPPER]``   | Uppercase ASCII characters from ``A`` to ``Z``. |\n        | ``[LETTERS]`` | Lowercase and uppercase ASCII characters.       |\n        | ``[NUMBERS]`` | Numbers from 0 to 9.                            |\n\n        Examples:\n        | ${ret} = | Generate Random String |\n        | ${low} = | Generate Random String | 12 | [LOWER]         |\n        | ${bin} = | Generate Random String | 8  | 01              |\n        | ${hex} = | Generate Random String | 4  | [NUMBERS]abcdef |\n        | ${rnd} = | Generate Random String | 5-10 | # Generates a string 5 to 10 characters long |\n\n        Giving ``length`` as a range of values is new in Robot Framework 5.0.\n        '
        if length == '':
            length = 8
        if isinstance(length, str) and re.match('^\\d+-\\d+$', length):
            (min_length, max_length) = length.split('-')
            length = randint(self._convert_to_integer(min_length, 'length'), self._convert_to_integer(max_length, 'length'))
        else:
            length = self._convert_to_integer(length, 'length')
        for (name, value) in [('[LOWER]', ascii_lowercase), ('[UPPER]', ascii_uppercase), ('[LETTERS]', ascii_lowercase + ascii_uppercase), ('[NUMBERS]', digits)]:
            chars = chars.replace(name, value)
        maxi = len(chars) - 1
        return ''.join((chars[randint(0, maxi)] for _ in range(length)))

    def get_substring(self, string, start, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns a substring from ``start`` index to ``end`` index.\n\n        The ``start`` index is inclusive and ``end`` is exclusive.\n        Indexing starts from 0, and it is possible to use\n        negative indices to refer to characters from the end.\n\n        Examples:\n        | ${ignore first} = | Get Substring | ${string} | 1  |    |\n        | ${ignore last} =  | Get Substring | ${string} |    | -1 |\n        | ${5th to 10th} =  | Get Substring | ${string} | 4  | 10 |\n        | ${first two} =    | Get Substring | ${string} |    | 1  |\n        | ${last two} =     | Get Substring | ${string} | -2 |    |\n        '
        start = self._convert_to_index(start, 'start')
        end = self._convert_to_index(end, 'end')
        return string[start:end]

    @keyword(types=None)
    def strip_string(self, string, mode='both', characters=None):
        if False:
            print('Hello World!')
        'Remove leading and/or trailing whitespaces from the given string.\n\n        ``mode`` is either ``left`` to remove leading characters, ``right`` to\n        remove trailing characters, ``both`` (default) to remove the\n        characters from both sides of the string or ``none`` to return the\n        unmodified string.\n\n        If the optional ``characters`` is given, it must be a string and the\n        characters in the string will be stripped in the string. Please note,\n        that this is not a substring to be removed but a list of characters,\n        see the example below.\n\n        Examples:\n        | ${stripped}=  | Strip String | ${SPACE}Hello${SPACE} | |\n        | Should Be Equal | ${stripped} | Hello | |\n        | ${stripped}=  | Strip String | ${SPACE}Hello${SPACE} | mode=left |\n        | Should Be Equal | ${stripped} | Hello${SPACE} | |\n        | ${stripped}=  | Strip String | aabaHelloeee | characters=abe |\n        | Should Be Equal | ${stripped} | Hello | |\n        '
        try:
            method = {'BOTH': string.strip, 'LEFT': string.lstrip, 'RIGHT': string.rstrip, 'NONE': lambda characters: string}[mode.upper()]
        except KeyError:
            raise ValueError("Invalid mode '%s'." % mode)
        return method(characters)

    def should_be_string(self, item, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the given ``item`` is not a string.\n\n        The default error message can be overridden with the optional ``msg`` argument.\n        '
        if not is_string(item):
            self._fail(msg, "'%s' is %s, not a string.", item, type_name(item))

    def should_not_be_string(self, item, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the given ``item`` is a string.\n\n        The default error message can be overridden with the optional ``msg`` argument.\n        '
        if is_string(item):
            self._fail(msg, "'%s' is a string.", item)

    def should_be_unicode_string(self, item, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the given ``item`` is not a Unicode string.\n\n        On Python 3 this keyword behaves exactly the same way `Should Be String`.\n        That keyword should be used instead and this keyword will be deprecated.\n        '
        if not is_string(item):
            self._fail(msg, "'%s' is not a Unicode string.", item)

    def should_be_byte_string(self, item, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the given ``item`` is not a byte string.\n\n        Use `Should Be String` if you want to verify the ``item`` is a string.\n\n        The default error message can be overridden with the optional ``msg`` argument.\n        '
        if not is_bytes(item):
            self._fail(msg, "'%s' is not a byte string.", item)

    def should_be_lower_case(self, string, msg=None):
        if False:
            return 10
        "Fails if the given ``string`` is not in lower case.\n\n        For example, ``'string'`` and ``'with specials!'`` would pass, and\n        ``'String'``, ``''`` and ``' '`` would fail.\n\n        The default error message can be overridden with the optional\n        ``msg`` argument.\n\n        See also `Should Be Upper Case` and `Should Be Title Case`.\n        "
        if not string.islower():
            self._fail(msg, "'%s' is not lower case.", string)

    def should_be_upper_case(self, string, msg=None):
        if False:
            print('Hello World!')
        "Fails if the given ``string`` is not in upper case.\n\n        For example, ``'STRING'`` and ``'WITH SPECIALS!'`` would pass, and\n        ``'String'``, ``''`` and ``' '`` would fail.\n\n        The default error message can be overridden with the optional\n        ``msg`` argument.\n\n        See also `Should Be Title Case` and `Should Be Lower Case`.\n        "
        if not string.isupper():
            self._fail(msg, "'%s' is not upper case.", string)

    @keyword(types=None)
    def should_be_title_case(self, string, msg=None, exclude=None):
        if False:
            print('Hello World!')
        'Fails if given ``string`` is not title.\n\n        ``string`` is a title cased string if there is at least one upper case\n        letter in each word.\n\n        For example, ``\'This Is Title\'`` and ``\'OK, Give Me My iPhone\'``\n        would pass. ``\'all words lower\'`` and ``\'Word In lower\'`` would fail.\n\n        This logic changed in Robot Framework 4.0 to be compatible with\n        `Convert to Title Case`. See `Convert to Title Case` for title case\n        algorithm and reasoning.\n\n        The default error message can be overridden with the optional\n        ``msg`` argument.\n\n        Words can be explicitly excluded with the optional ``exclude`` argument.\n\n        Explicitly excluded words can be given as a list or as a string with\n        words separated by a comma and an optional space. Excluded words are\n        actually considered to be regular expression patterns, so it is\n        possible to use something like "example[.!?]?" to match the word\n        "example" on it own and also if followed by ".", "!" or "?".\n        See `BuiltIn.Should Match Regexp` for more information about Python\n        regular expression syntax in general and how to use it in Robot\n        Framework data in particular.\n\n        See also `Should Be Upper Case` and `Should Be Lower Case`.\n        '
        if string != self.convert_to_title_case(string, exclude):
            self._fail(msg, "'%s' is not title case.", string)

    def _convert_to_index(self, value, name):
        if False:
            i = 10
            return i + 15
        if value == '':
            return 0
        if value is None:
            return None
        return self._convert_to_integer(value, name)

    def _convert_to_integer(self, value, name):
        if False:
            print('Hello World!')
        try:
            return int(value)
        except ValueError:
            raise ValueError("Cannot convert '%s' argument '%s' to an integer." % (name, value))

    def _fail(self, message, default_template, *items):
        if False:
            return 10
        if not message:
            message = default_template % tuple((safe_str(item) for item in items))
        raise AssertionError(message)