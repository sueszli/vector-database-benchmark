"""
This module implements Git's wildmatch pattern matching which itself is
derived from Rsync's wildmatch. Git uses wildmatch for its ".gitignore"
files.
"""
from __future__ import unicode_literals
import re
import warnings
from .. import util
from ..compat import unicode
from ..pattern import RegexPattern
_BYTES_ENCODING = 'latin1'

class GitWildMatchPattern(RegexPattern):
    """
	The :class:`GitWildMatchPattern` class represents a compiled Git
	wildmatch pattern.
	"""
    __slots__ = ()

    @classmethod
    def pattern_to_regex(cls, pattern):
        if False:
            return 10
        '\n\t\tConvert the pattern into a regular expression.\n\n\t\t*pattern* (:class:`unicode` or :class:`bytes`) is the pattern to\n\t\tconvert into a regular expression.\n\n\t\tReturns the uncompiled regular expression (:class:`unicode`, :class:`bytes`,\n\t\tor :data:`None`), and whether matched files should be included\n\t\t(:data:`True`), excluded (:data:`False`), or if it is a\n\t\tnull-operation (:data:`None`).\n\t\t'
        if isinstance(pattern, unicode):
            return_type = unicode
        elif isinstance(pattern, bytes):
            return_type = bytes
            pattern = pattern.decode(_BYTES_ENCODING)
        else:
            raise TypeError('pattern:{!r} is not a unicode or byte string.'.format(pattern))
        pattern = pattern.strip()
        if pattern.startswith('#'):
            regex = None
            include = None
        elif pattern == '/':
            regex = None
            include = None
        elif pattern:
            if pattern.startswith('!'):
                include = False
                pattern = pattern[1:]
            else:
                include = True
            if pattern.startswith('\\'):
                pattern = pattern[1:]
            pattern_segs = pattern.split('/')
            if not pattern_segs[0]:
                del pattern_segs[0]
            elif len(pattern_segs) == 1 or (len(pattern_segs) == 2 and (not pattern_segs[1])):
                if pattern_segs[0] != '**':
                    pattern_segs.insert(0, '**')
            else:
                pass
            if not pattern_segs[-1] and len(pattern_segs) > 1:
                pattern_segs[-1] = '**'
            output = ['^']
            need_slash = False
            end = len(pattern_segs) - 1
            for (i, seg) in enumerate(pattern_segs):
                if seg == '**':
                    if i == 0 and i == end:
                        output.append('.+')
                    elif i == 0:
                        output.append('(?:.+/)?')
                        need_slash = False
                    elif i == end:
                        output.append('/.*')
                    else:
                        output.append('(?:/.+)?')
                        need_slash = True
                elif seg == '*':
                    if need_slash:
                        output.append('/')
                    output.append('[^/]+')
                    need_slash = True
                else:
                    if need_slash:
                        output.append('/')
                    output.append(cls._translate_segment_glob(seg))
                    if i == end and include is True:
                        output.append('(?:/.*)?')
                    need_slash = True
            output.append('$')
            regex = ''.join(output)
        else:
            regex = None
            include = None
        if regex is not None and return_type is bytes:
            regex = regex.encode(_BYTES_ENCODING)
        return (regex, include)

    @staticmethod
    def _translate_segment_glob(pattern):
        if False:
            i = 10
            return i + 15
        '\n\t\tTranslates the glob pattern to a regular expression. This is used in\n\t\tthe constructor to translate a path segment glob pattern to its\n\t\tcorresponding regular expression.\n\n\t\t*pattern* (:class:`str`) is the glob pattern.\n\n\t\tReturns the regular expression (:class:`str`).\n\t\t'
        escape = False
        regex = ''
        (i, end) = (0, len(pattern))
        while i < end:
            char = pattern[i]
            i += 1
            if escape:
                escape = False
                regex += re.escape(char)
            elif char == '\\':
                escape = True
            elif char == '*':
                regex += '[^/]*'
            elif char == '?':
                regex += '[^/]'
            elif char == '[':
                j = i
                if j < end and pattern[j] == '!':
                    j += 1
                if j < end and pattern[j] == ']':
                    j += 1
                while j < end and pattern[j] != ']':
                    j += 1
                if j < end:
                    j += 1
                    expr = '['
                    if pattern[i] == '!':
                        expr += '^'
                        i += 1
                    elif pattern[i] == '^':
                        expr += '\\^'
                        i += 1
                    expr += pattern[i:j].replace('\\', '\\\\')
                    regex += expr
                    i = j
                else:
                    regex += '\\['
            else:
                regex += re.escape(char)
        return regex

    @staticmethod
    def escape(s):
        if False:
            return 10
        '\n\t\tEscape special characters in the given string.\n\n\t\t*s* (:class:`unicode` or :class:`bytes`) a filename or a string\n\t\tthat you want to escape, usually before adding it to a `.gitignore`\n\n\t\tReturns the escaped string (:class:`unicode`, :class:`bytes`)\n\t\t'
        meta_characters = '[]!*#?'
        return ''.join(('\\' + x if x in meta_characters else x for x in s))
util.register_pattern('gitwildmatch', GitWildMatchPattern)

class GitIgnorePattern(GitWildMatchPattern):
    """
	The :class:`GitIgnorePattern` class is deprecated by :class:`GitWildMatchPattern`.
	This class only exists to maintain compatibility with v0.4.
	"""

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        '\n\t\tWarn about deprecation.\n\t\t'
        self._deprecated()
        return super(GitIgnorePattern, self).__init__(*args, **kw)

    @staticmethod
    def _deprecated():
        if False:
            while True:
                i = 10
        '\n\t\tWarn about deprecation.\n\t\t'
        warnings.warn("GitIgnorePattern ('gitignore') is deprecated. Use GitWildMatchPattern ('gitwildmatch') instead.", DeprecationWarning, stacklevel=3)

    @classmethod
    def pattern_to_regex(cls, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tWarn about deprecation.\n\t\t'
        cls._deprecated()
        return super(GitIgnorePattern, cls).pattern_to_regex(*args, **kw)
util.register_pattern('gitignore', GitIgnorePattern)