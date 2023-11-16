"""
Python Lexical Analyser

Regular Expressions
"""
from __future__ import absolute_import
import types
from . import Errors
maxint = 2 ** 31 - 1
BOL = 'bol'
EOL = 'eol'
EOF = 'eof'
nl_code = ord('\n')

def chars_to_ranges(s):
    if False:
        i = 10
        return i + 15
    '\n    Return a list of character codes consisting of pairs\n    [code1a, code1b, code2a, code2b,...] which cover all\n    the characters in |s|.\n    '
    char_list = list(s)
    char_list.sort()
    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        while i < n and code2 >= ord(char_list[i]):
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result

def uppercase_range(code1, code2):
    if False:
        while True:
            i = 10
    '\n    If the range of characters from code1 to code2-1 includes any\n    lower case letters, return the corresponding upper case range.\n    '
    code3 = max(code1, ord('a'))
    code4 = min(code2, ord('z') + 1)
    if code3 < code4:
        d = ord('A') - ord('a')
        return (code3 + d, code4 + d)
    else:
        return None

def lowercase_range(code1, code2):
    if False:
        i = 10
        return i + 15
    '\n    If the range of characters from code1 to code2-1 includes any\n    upper case letters, return the corresponding lower case range.\n    '
    code3 = max(code1, ord('A'))
    code4 = min(code2, ord('Z') + 1)
    if code3 < code4:
        d = ord('a') - ord('A')
        return (code3 + d, code4 + d)
    else:
        return None

def CodeRanges(code_list):
    if False:
        i = 10
        return i + 15
    '\n    Given a list of codes as returned by chars_to_ranges, return\n    an RE which will match a character in any of the ranges.\n    '
    re_list = [CodeRange(code_list[i], code_list[i + 1]) for i in range(0, len(code_list), 2)]
    return Alt(*re_list)

def CodeRange(code1, code2):
    if False:
        i = 10
        return i + 15
    '\n    CodeRange(code1, code2) is an RE which matches any character\n    with a code |c| in the range |code1| <= |c| < |code2|.\n    '
    if code1 <= nl_code < code2:
        return Alt(RawCodeRange(code1, nl_code), RawNewline, RawCodeRange(nl_code + 1, code2))
    else:
        return RawCodeRange(code1, code2)

class RE(object):
    """RE is the base class for regular expression constructors.
    The following operators are defined on REs:

         re1 + re2         is an RE which matches |re1| followed by |re2|
         re1 | re2         is an RE which matches either |re1| or |re2|
    """
    nullable = 1
    match_nl = 1
    str = None

    def build_machine(self, machine, initial_state, final_state, match_bol, nocase):
        if False:
            print('Hello World!')
        '\n        This method should add states to |machine| to implement this\n        RE, starting at |initial_state| and ending at |final_state|.\n        If |match_bol| is true, the RE must be able to match at the\n        beginning of a line. If nocase is true, upper and lower case\n        letters should be treated as equivalent.\n        '
        raise NotImplementedError('%s.build_machine not implemented' % self.__class__.__name__)

    def build_opt(self, m, initial_state, c):
        if False:
            return 10
        '\n        Given a state |s| of machine |m|, return a new state\n        reachable from |s| on character |c| or epsilon.\n        '
        s = m.new_state()
        initial_state.link_to(s)
        initial_state.add_transition(c, s)
        return s

    def __add__(self, other):
        if False:
            while True:
                i = 10
        return Seq(self, other)

    def __or__(self, other):
        if False:
            print('Hello World!')
        return Alt(self, other)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.str:
            return self.str
        else:
            return self.calc_str()

    def check_re(self, num, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, RE):
            self.wrong_type(num, value, 'Plex.RE instance')

    def check_string(self, num, value):
        if False:
            while True:
                i = 10
        if type(value) != type(''):
            self.wrong_type(num, value, 'string')

    def check_char(self, num, value):
        if False:
            while True:
                i = 10
        self.check_string(num, value)
        if len(value) != 1:
            raise Errors.PlexValueError('Invalid value for argument %d of Plex.%s.Expected a string of length 1, got: %s' % (num, self.__class__.__name__, repr(value)))

    def wrong_type(self, num, value, expected):
        if False:
            while True:
                i = 10
        if type(value) == types.InstanceType:
            got = '%s.%s instance' % (value.__class__.__module__, value.__class__.__name__)
        else:
            got = type(value).__name__
        raise Errors.PlexTypeError('Invalid type for argument %d of Plex.%s (expected %s, got %s' % (num, self.__class__.__name__, expected, got))

def Char(c):
    if False:
        print('Hello World!')
    '\n    Char(c) is an RE which matches the character |c|.\n    '
    if len(c) == 1:
        result = CodeRange(ord(c), ord(c) + 1)
    else:
        result = SpecialSymbol(c)
    result.str = 'Char(%s)' % repr(c)
    return result

class RawCodeRange(RE):
    """
    RawCodeRange(code1, code2) is a low-level RE which matches any character
    with a code |c| in the range |code1| <= |c| < |code2|, where the range
    does not include newline. For internal use only.
    """
    nullable = 0
    match_nl = 0
    range = None
    uppercase_range = None
    lowercase_range = None

    def __init__(self, code1, code2):
        if False:
            i = 10
            return i + 15
        self.range = (code1, code2)
        self.uppercase_range = uppercase_range(code1, code2)
        self.lowercase_range = lowercase_range(code1, code2)

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            print('Hello World!')
        if match_bol:
            initial_state = self.build_opt(m, initial_state, BOL)
        initial_state.add_transition(self.range, final_state)
        if nocase:
            if self.uppercase_range:
                initial_state.add_transition(self.uppercase_range, final_state)
            if self.lowercase_range:
                initial_state.add_transition(self.lowercase_range, final_state)

    def calc_str(self):
        if False:
            while True:
                i = 10
        return 'CodeRange(%d,%d)' % (self.code1, self.code2)

class _RawNewline(RE):
    """
    RawNewline is a low-level RE which matches a newline character.
    For internal use only.
    """
    nullable = 0
    match_nl = 1

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            i = 10
            return i + 15
        if match_bol:
            initial_state = self.build_opt(m, initial_state, BOL)
        s = self.build_opt(m, initial_state, EOL)
        s.add_transition((nl_code, nl_code + 1), final_state)
RawNewline = _RawNewline()

class SpecialSymbol(RE):
    """
    SpecialSymbol(sym) is an RE which matches the special input
    symbol |sym|, which is one of BOL, EOL or EOF.
    """
    nullable = 0
    match_nl = 0
    sym = None

    def __init__(self, sym):
        if False:
            print('Hello World!')
        self.sym = sym

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            while True:
                i = 10
        if match_bol and self.sym == EOL:
            initial_state = self.build_opt(m, initial_state, BOL)
        initial_state.add_transition(self.sym, final_state)

class Seq(RE):
    """Seq(re1, re2, re3...) is an RE which matches |re1| followed by
    |re2| followed by |re3|..."""

    def __init__(self, *re_list):
        if False:
            while True:
                i = 10
        nullable = 1
        for (i, re) in enumerate(re_list):
            self.check_re(i, re)
            nullable = nullable and re.nullable
        self.re_list = re_list
        self.nullable = nullable
        i = len(re_list)
        match_nl = 0
        while i:
            i -= 1
            re = re_list[i]
            if re.match_nl:
                match_nl = 1
                break
            if not re.nullable:
                break
        self.match_nl = match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            i = 10
            return i + 15
        re_list = self.re_list
        if len(re_list) == 0:
            initial_state.link_to(final_state)
        else:
            s1 = initial_state
            n = len(re_list)
            for (i, re) in enumerate(re_list):
                if i < n - 1:
                    s2 = m.new_state()
                else:
                    s2 = final_state
                re.build_machine(m, s1, s2, match_bol, nocase)
                s1 = s2
                match_bol = re.match_nl or (match_bol and re.nullable)

    def calc_str(self):
        if False:
            print('Hello World!')
        return 'Seq(%s)' % ','.join(map(str, self.re_list))

class Alt(RE):
    """Alt(re1, re2, re3...) is an RE which matches either |re1| or
    |re2| or |re3|..."""

    def __init__(self, *re_list):
        if False:
            print('Hello World!')
        self.re_list = re_list
        nullable = 0
        match_nl = 0
        nullable_res = []
        non_nullable_res = []
        i = 1
        for re in re_list:
            self.check_re(i, re)
            if re.nullable:
                nullable_res.append(re)
                nullable = 1
            else:
                non_nullable_res.append(re)
            if re.match_nl:
                match_nl = 1
            i += 1
        self.nullable_res = nullable_res
        self.non_nullable_res = non_nullable_res
        self.nullable = nullable
        self.match_nl = match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            return 10
        for re in self.nullable_res:
            re.build_machine(m, initial_state, final_state, match_bol, nocase)
        if self.non_nullable_res:
            if match_bol:
                initial_state = self.build_opt(m, initial_state, BOL)
            for re in self.non_nullable_res:
                re.build_machine(m, initial_state, final_state, 0, nocase)

    def calc_str(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Alt(%s)' % ','.join(map(str, self.re_list))

class Rep1(RE):
    """Rep1(re) is an RE which matches one or more repetitions of |re|."""

    def __init__(self, re):
        if False:
            i = 10
            return i + 15
        self.check_re(1, re)
        self.re = re
        self.nullable = re.nullable
        self.match_nl = re.match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            i = 10
            return i + 15
        s1 = m.new_state()
        s2 = m.new_state()
        initial_state.link_to(s1)
        self.re.build_machine(m, s1, s2, match_bol or self.re.match_nl, nocase)
        s2.link_to(s1)
        s2.link_to(final_state)

    def calc_str(self):
        if False:
            while True:
                i = 10
        return 'Rep1(%s)' % self.re

class SwitchCase(RE):
    """
    SwitchCase(re, nocase) is an RE which matches the same strings as RE,
    but treating upper and lower case letters according to |nocase|. If
    |nocase| is true, case is ignored, otherwise it is not.
    """
    re = None
    nocase = None

    def __init__(self, re, nocase):
        if False:
            while True:
                i = 10
        self.re = re
        self.nocase = nocase
        self.nullable = re.nullable
        self.match_nl = re.match_nl

    def build_machine(self, m, initial_state, final_state, match_bol, nocase):
        if False:
            for i in range(10):
                print('nop')
        self.re.build_machine(m, initial_state, final_state, match_bol, self.nocase)

    def calc_str(self):
        if False:
            i = 10
            return i + 15
        if self.nocase:
            name = 'NoCase'
        else:
            name = 'Case'
        return '%s(%s)' % (name, self.re)
Empty = Seq()
Empty.__doc__ = '\n    Empty is an RE which matches the empty string.\n    '
Empty.str = 'Empty'

def Str1(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    Str1(s) is an RE which matches the literal string |s|.\n    '
    result = Seq(*tuple(map(Char, s)))
    result.str = 'Str(%s)' % repr(s)
    return result

def Str(*strs):
    if False:
        while True:
            i = 10
    '\n    Str(s) is an RE which matches the literal string |s|.\n    Str(s1, s2, s3, ...) is an RE which matches any of |s1| or |s2| or |s3|...\n    '
    if len(strs) == 1:
        return Str1(strs[0])
    else:
        result = Alt(*tuple(map(Str1, strs)))
        result.str = 'Str(%s)' % ','.join(map(repr, strs))
        return result

def Any(s):
    if False:
        i = 10
        return i + 15
    '\n    Any(s) is an RE which matches any character in the string |s|.\n    '
    result = CodeRanges(chars_to_ranges(s))
    result.str = 'Any(%s)' % repr(s)
    return result

def AnyBut(s):
    if False:
        return 10
    '\n    AnyBut(s) is an RE which matches any character (including\n    newline) which is not in the string |s|.\n    '
    ranges = chars_to_ranges(s)
    ranges.insert(0, -maxint)
    ranges.append(maxint)
    result = CodeRanges(ranges)
    result.str = 'AnyBut(%s)' % repr(s)
    return result
AnyChar = AnyBut('')
AnyChar.__doc__ = '\n    AnyChar is an RE which matches any single character (including a newline).\n    '
AnyChar.str = 'AnyChar'

def Range(s1, s2=None):
    if False:
        return 10
    '\n    Range(c1, c2) is an RE which matches any single character in the range\n    |c1| to |c2| inclusive.\n    Range(s) where |s| is a string of even length is an RE which matches\n    any single character in the ranges |s[0]| to |s[1]|, |s[2]| to |s[3]|,...\n    '
    if s2:
        result = CodeRange(ord(s1), ord(s2) + 1)
        result.str = 'Range(%s,%s)' % (s1, s2)
    else:
        ranges = []
        for i in range(0, len(s1), 2):
            ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
        result = Alt(*ranges)
        result.str = 'Range(%s)' % repr(s1)
    return result

def Opt(re):
    if False:
        i = 10
        return i + 15
    '\n    Opt(re) is an RE which matches either |re| or the empty string.\n    '
    result = Alt(re, Empty)
    result.str = 'Opt(%s)' % re
    return result

def Rep(re):
    if False:
        return 10
    '\n    Rep(re) is an RE which matches zero or more repetitions of |re|.\n    '
    result = Opt(Rep1(re))
    result.str = 'Rep(%s)' % re
    return result

def NoCase(re):
    if False:
        while True:
            i = 10
    '\n    NoCase(re) is an RE which matches the same strings as RE, but treating\n    upper and lower case letters as equivalent.\n    '
    return SwitchCase(re, nocase=1)

def Case(re):
    if False:
        i = 10
        return i + 15
    '\n    Case(re) is an RE which matches the same strings as RE, but treating\n    upper and lower case letters as distinct, i.e. it cancels the effect\n    of any enclosing NoCase().\n    '
    return SwitchCase(re, nocase=0)
Bol = Char(BOL)
Bol.__doc__ = '\n    Bol is an RE which matches the beginning of a line.\n    '
Bol.str = 'Bol'
Eol = Char(EOL)
Eol.__doc__ = '\n    Eol is an RE which matches the end of a line.\n    '
Eol.str = 'Eol'
Eof = Char(EOF)
Eof.__doc__ = '\n    Eof is an RE which matches the end of the file.\n    '
Eof.str = 'Eof'