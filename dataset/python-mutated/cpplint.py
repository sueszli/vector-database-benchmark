"""Does google-lint on c++ files.

The goal of this script is to identify places in the code that *may*
be in non-compliance with google style.  It does not attempt to fix
up these problems -- the point is to educate.  It does also not
attempt to find all problems, or to ensure that everything it does
find is legitimately a problem.

In particular, we can get very confused by /* and // inside strings!
We do a small hack, which is to ignore //'s with "'s after them on the
same line, but it is far from perfect (in either direction).
"""
import codecs
import copy
import getopt
import math
import os
import re
import sre_compile
import string
import sys
import unicodedata
python2_version = False
if sys.version_info[0] < 3:
    python2_version = True
_USAGE = '\nSyntax: cpplint.py [--verbose=#] [--output=vs7] [--filter=-x,+y,...]\n                   [--counting=total|toplevel|detailed] [--root=subdir]\n                   [--linelength=digits] [--headers=x,y,...]\n                   [--quiet]\n        <file> [file] ...\n\n  The style guidelines this tries to follow are those in\n    https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml\n\n  Every problem is given a confidence score from 1-5, with 5 meaning we are\n  certain of the problem, and 1 meaning it could be a legitimate construct.\n  This will miss some errors, and is not a substitute for a code review.\n\n  To suppress false-positive errors of a certain category, add a\n  \'NOLINT(category)\' comment to the line.  NOLINT or NOLINT(*)\n  suppresses errors of all categories on that line.\n\n  The files passed in will be linted; at least one file must be provided.\n  Default linted extensions are .cc, .cpp, .cu, .cuh and .h.  Change the\n  extensions with the --extensions flag.\n\n  Flags:\n\n    output=vs7\n      By default, the output is formatted to ease emacs parsing.  Visual Studio\n      compatible output (vs7) may also be used.  Other formats are unsupported.\n\n    verbose=#\n      Specify a number 0-5 to restrict errors to certain verbosity levels.\n\n    quiet\n      Don\'t print anything if no errors are found.\n\n    filter=-x,+y,...\n      Specify a comma-separated list of category-filters to apply: only\n      error messages whose category names pass the filters will be printed.\n      (Category names are printed with the message and look like\n      "[whitespace/indent]".)  Filters are evaluated left to right.\n      "-FOO" and "FOO" means "do not print categories that start with FOO".\n      "+FOO" means "do print categories that start with FOO".\n\n      Examples: --filter=-whitespace,+whitespace/braces\n                --filter=whitespace,runtime/printf,+runtime/printf_format\n                --filter=-,+build/include_what_you_use\n\n      To see a list of all the categories used in cpplint, pass no arg:\n         --filter=\n\n    counting=total|toplevel|detailed\n      The total number of errors found is always printed. If\n      \'toplevel\' is provided, then the count of errors in each of\n      the top-level categories like \'build\' and \'whitespace\' will\n      also be printed. If \'detailed\' is provided, then a count\n      is provided for each category like \'build/class\'.\n\n    root=subdir\n      The root directory used for deriving header guard CPP variable.\n      By default, the header guard CPP variable is calculated as the relative\n      path to the directory that contains .git, .hg, or .svn.  When this flag\n      is specified, the relative path is calculated from the specified\n      directory. If the specified directory does not exist, this flag is\n      ignored.\n\n      Examples:\n        Assuming that top/src/.git exists (and cwd=top/src), the header guard\n        CPP variables for top/src/chrome/browser/ui/browser.h are:\n\n        No flag => CHROME_BROWSER_UI_BROWSER_H_\n        --root=chrome => BROWSER_UI_BROWSER_H_\n        --root=chrome/browser => UI_BROWSER_H_\n        --root=.. => SRC_CHROME_BROWSER_UI_BROWSER_H_\n\n    linelength=digits\n      This is the allowed line length for the project. The default value is\n      80 characters.\n\n      Examples:\n        --linelength=120\n\n    extensions=extension,extension,...\n      The allowed file extensions that cpplint will check\n\n      Examples:\n        --extensions=hpp,cpp\n\n    headers=x,y,...\n      The header extensions that cpplint will treat as .h in checks. Values are\n      automatically added to --extensions list.\n\n      Examples:\n        --headers=hpp,hxx\n        --headers=hpp\n\n    cpplint.py supports per-directory configurations specified in CPPLINT.cfg\n    files. CPPLINT.cfg file can contain a number of key=value pairs.\n    Currently the following options are supported:\n\n      set noparent\n      filter=+filter1,-filter2,...\n      exclude_files=regex\n      linelength=80\n      root=subdir\n      headers=x,y,...\n\n    "set noparent" option prevents cpplint from traversing directory tree\n    upwards looking for more .cfg files in parent directories. This option\n    is usually placed in the top-level project directory.\n\n    The "filter" option is similar in function to --filter flag. It specifies\n    message filters in addition to the |_DEFAULT_FILTERS| and those specified\n    through --filter command-line flag.\n\n    "exclude_files" allows to specify a regular expression to be matched against\n    a file name. If the expression matches, the file is skipped and not run\n    through liner.\n\n    "linelength" allows to specify the allowed line length for the project.\n\n    The "root" option is similar in function to the --root flag (see example\n    above). Paths are relative to the directory of the CPPLINT.cfg.\n\n    The "headers" option is similar in function to the --headers flag\n    (see example above).\n\n    CPPLINT.cfg has an effect on files in the same directory and all\n    sub-directories, unless overridden by a nested configuration file.\n\n      Example file:\n        filter=-build/include_order,+build/include_alpha\n        exclude_files=.*\\.cc\n\n    The above example disables build/include_order warning and enables\n    build/include_alpha as well as excludes all .cc from being\n    processed by linter, in the current directory (where the .cfg\n    file is located) and all sub-directories.\n'
_ERROR_CATEGORIES = ['build/class', 'build/c++11', 'build/c++14', 'build/c++tr1', 'build/deprecated', 'build/endif_comment', 'build/explicit_make_pair', 'build/forward_decl', 'build/header_guard', 'build/include', 'build/include_alpha', 'build/include_order', 'build/include_what_you_use', 'build/namespaces', 'build/printf_format', 'build/storage_class', 'legal/copyright', 'readability/alt_tokens', 'readability/braces', 'readability/casting', 'readability/check', 'readability/constructors', 'readability/fn_size', 'readability/inheritance', 'readability/multiline_comment', 'readability/multiline_string', 'readability/namespace', 'readability/nolint', 'readability/nul', 'readability/strings', 'readability/todo', 'readability/utf8', 'runtime/arrays', 'runtime/casting', 'runtime/explicit', 'runtime/int', 'runtime/init', 'runtime/invalid_increment', 'runtime/member_string_references', 'runtime/memset', 'runtime/indentation_namespace', 'runtime/operator', 'runtime/printf', 'runtime/printf_format', 'runtime/references', 'runtime/string', 'runtime/threadsafe_fn', 'runtime/vlog', 'whitespace/blank_line', 'whitespace/braces', 'whitespace/comma', 'whitespace/comments', 'whitespace/empty_conditional_body', 'whitespace/empty_if_body', 'whitespace/empty_loop_body', 'whitespace/end_of_line', 'whitespace/ending_newline', 'whitespace/forcolon', 'whitespace/indent', 'whitespace/line_length', 'whitespace/newline', 'whitespace/operators', 'whitespace/parens', 'whitespace/semicolon', 'whitespace/tab', 'whitespace/todo']
_LEGACY_ERROR_CATEGORIES = ['readability/streams', 'readability/function']
_DEFAULT_FILTERS = ['-build/include_alpha']
_DEFAULT_C_SUPPRESSED_CATEGORIES = ['readability/casting']
_DEFAULT_KERNEL_SUPPRESSED_CATEGORIES = ['whitespace/tab']
_CPP_HEADERS = frozenset(['algobase.h', 'algo.h', 'alloc.h', 'builtinbuf.h', 'bvector.h', 'complex.h', 'defalloc.h', 'deque.h', 'editbuf.h', 'fstream.h', 'function.h', 'hash_map', 'hash_map.h', 'hash_set', 'hash_set.h', 'hashtable.h', 'heap.h', 'indstream.h', 'iomanip.h', 'iostream.h', 'istream.h', 'iterator.h', 'list.h', 'map.h', 'multimap.h', 'multiset.h', 'ostream.h', 'pair.h', 'parsestream.h', 'pfstream.h', 'procbuf.h', 'pthread_alloc', 'pthread_alloc.h', 'rope', 'rope.h', 'ropeimpl.h', 'set.h', 'slist', 'slist.h', 'stack.h', 'stdiostream.h', 'stl_alloc.h', 'stl_relops.h', 'streambuf.h', 'stream.h', 'strfile.h', 'strstream.h', 'tempbuf.h', 'tree.h', 'type_traits.h', 'vector.h', 'algorithm', 'array', 'atomic', 'bitset', 'chrono', 'codecvt', 'complex', 'condition_variable', 'deque', 'exception', 'forward_list', 'fstream', 'functional', 'future', 'initializer_list', 'iomanip', 'ios', 'iosfwd', 'iostream', 'istream', 'iterator', 'limits', 'list', 'locale', 'map', 'memory', 'mutex', 'new', 'numeric', 'ostream', 'queue', 'random', 'ratio', 'regex', 'scoped_allocator', 'set', 'sstream', 'stack', 'stdexcept', 'streambuf', 'string', 'strstream', 'system_error', 'thread', 'tuple', 'typeindex', 'typeinfo', 'type_traits', 'unordered_map', 'unordered_set', 'utility', 'valarray', 'vector', 'cassert', 'ccomplex', 'cctype', 'cerrno', 'cfenv', 'cfloat', 'cinttypes', 'ciso646', 'climits', 'clocale', 'cmath', 'csetjmp', 'csignal', 'cstdalign', 'cstdarg', 'cstdbool', 'cstddef', 'cstdint', 'cstdio', 'cstdlib', 'cstring', 'ctgmath', 'ctime', 'cuchar', 'cwchar', 'cwctype'])
_TYPES = re.compile('^(?:(char(16_t|32_t)?)|wchar_t|bool|short|int|long|signed|unsigned|float|double|(ptrdiff_t|size_t|max_align_t|nullptr_t)|(u?int(_fast|_least)?(8|16|32|64)_t)|(u?int(max|ptr)_t)|)$')
_THIRD_PARTY_HEADERS_PATTERN = re.compile('^(?:[^/]*[A-Z][^/]*\\.h|lua\\.h|lauxlib\\.h|lualib\\.h)$')
_TEST_FILE_SUFFIX = '(_test|_unittest|_regtest)$'
_EMPTY_CONDITIONAL_BODY_PATTERN = re.compile('^\\s*$', re.DOTALL)
_CHECK_MACROS = ['DCHECK', 'CHECK', 'EXPECT_TRUE', 'ASSERT_TRUE', 'EXPECT_FALSE', 'ASSERT_FALSE']
_CHECK_REPLACEMENT = dict([(m, {}) for m in _CHECK_MACROS])
for (op, replacement) in [('==', 'EQ'), ('!=', 'NE'), ('>=', 'GE'), ('>', 'GT'), ('<=', 'LE'), ('<', 'LT')]:
    _CHECK_REPLACEMENT['DCHECK'][op] = 'DCHECK_%s' % replacement
    _CHECK_REPLACEMENT['CHECK'][op] = 'CHECK_%s' % replacement
    _CHECK_REPLACEMENT['EXPECT_TRUE'][op] = 'EXPECT_%s' % replacement
    _CHECK_REPLACEMENT['ASSERT_TRUE'][op] = 'ASSERT_%s' % replacement
for (op, inv_replacement) in [('==', 'NE'), ('!=', 'EQ'), ('>=', 'LT'), ('>', 'LE'), ('<=', 'GT'), ('<', 'GE')]:
    _CHECK_REPLACEMENT['EXPECT_FALSE'][op] = 'EXPECT_%s' % inv_replacement
    _CHECK_REPLACEMENT['ASSERT_FALSE'][op] = 'ASSERT_%s' % inv_replacement
_ALT_TOKEN_REPLACEMENT = {'and': '&&', 'bitor': '|', 'or': '||', 'xor': '^', 'compl': '~', 'bitand': '&', 'and_eq': '&=', 'or_eq': '|=', 'xor_eq': '^=', 'not': '!', 'not_eq': '!='}
_ALT_TOKEN_REPLACEMENT_PATTERN = re.compile('[ =()](' + '|'.join(_ALT_TOKEN_REPLACEMENT.keys()) + ')(?=[ (]|$)')
_C_SYS_HEADER = 1
_CPP_SYS_HEADER = 2
_LIKELY_MY_HEADER = 3
_POSSIBLE_MY_HEADER = 4
_OTHER_HEADER = 5
_NO_ASM = 0
_INSIDE_ASM = 1
_END_ASM = 2
_BLOCK_ASM = 3
_MATCH_ASM = re.compile('^\\s*(?:asm|_asm|__asm|__asm__)(?:\\s+(volatile|__volatile__))?\\s*[{(]')
_SEARCH_C_FILE = re.compile('\\b(?:LINT_C_FILE|vim?:\\s*.*(\\s*|:)filetype=c(\\s*|:|$))')
_SEARCH_KERNEL_FILE = re.compile('\\b(?:LINT_KERNEL_FILE)')
_regexp_compile_cache = {}
_error_suppressions = {}
_root = None
_root_debug = False
_line_length = 80
_valid_extensions = set(['cc', 'h', 'cpp', 'cu', 'cuh'])
_hpp_headers = set(['h'])
_global_error_suppressions = {}

def ProcessHppHeadersOption(val):
    if False:
        for i in range(10):
            print('nop')
    global _hpp_headers
    try:
        _hpp_headers = set(val.split(','))
        _valid_extensions.update(_hpp_headers)
    except ValueError:
        PrintUsage('Header extensions must be comma seperated list.')

def IsHeaderExtension(file_extension):
    if False:
        print('Hello World!')
    return file_extension in _hpp_headers

def ParseNolintSuppressions(filename, raw_line, linenum, error):
    if False:
        i = 10
        return i + 15
    'Updates the global list of line error-suppressions.\n\n  Parses any NOLINT comments on the current line, updating the global\n  error_suppressions store.  Reports an error if the NOLINT comment\n  was malformed.\n\n  Args:\n    filename: str, the name of the input file.\n    raw_line: str, the line of input text, with comments.\n    linenum: int, the number of the current line.\n    error: function, an error handler.\n  '
    matched = Search('\\bNOLINT(NEXTLINE)?\\b(\\([^)]+\\))?', raw_line)
    if matched:
        if matched.group(1):
            suppressed_line = linenum + 1
        else:
            suppressed_line = linenum
        category = matched.group(2)
        if category in (None, '(*)'):
            _error_suppressions.setdefault(None, set()).add(suppressed_line)
        elif category.startswith('(') and category.endswith(')'):
            category = category[1:-1]
            if category in _ERROR_CATEGORIES:
                _error_suppressions.setdefault(category, set()).add(suppressed_line)
            elif category not in _LEGACY_ERROR_CATEGORIES:
                error(filename, linenum, 'readability/nolint', 5, 'Unknown NOLINT error category: %s' % category)

def ProcessGlobalSuppresions(lines):
    if False:
        print('Hello World!')
    'Updates the list of global error suppressions.\n\n  Parses any lint directives in the file that have global effect.\n\n  Args:\n    lines: An array of strings, each representing a line of the file, with the\n           last element being empty if the file is terminated with a newline.\n  '
    for line in lines:
        if _SEARCH_C_FILE.search(line):
            for category in _DEFAULT_C_SUPPRESSED_CATEGORIES:
                _global_error_suppressions[category] = True
        if _SEARCH_KERNEL_FILE.search(line):
            for category in _DEFAULT_KERNEL_SUPPRESSED_CATEGORIES:
                _global_error_suppressions[category] = True

def ResetNolintSuppressions():
    if False:
        return 10
    'Resets the set of NOLINT suppressions to empty.'
    _error_suppressions.clear()
    _global_error_suppressions.clear()

def IsErrorSuppressedByNolint(category, linenum):
    if False:
        print('Hello World!')
    'Returns true if the specified error category is suppressed on this line.\n\n  Consults the global error_suppressions map populated by\n  ParseNolintSuppressions/ProcessGlobalSuppresions/ResetNolintSuppressions.\n\n  Args:\n    category: str, the category of the error.\n    linenum: int, the current line number.\n  Returns:\n    bool, True iff the error should be suppressed due to a NOLINT comment or\n    global suppression.\n  '
    return _global_error_suppressions.get(category, False) or linenum in _error_suppressions.get(category, set()) or linenum in _error_suppressions.get(None, set())

def Match(pattern, s):
    if False:
        for i in range(10):
            print('nop')
    'Matches the string with the pattern, caching the compiled regexp.'
    if pattern not in _regexp_compile_cache:
        _regexp_compile_cache[pattern] = sre_compile.compile(pattern)
    return _regexp_compile_cache[pattern].match(s)

def ReplaceAll(pattern, rep, s):
    if False:
        return 10
    'Replaces instances of pattern in a string with a replacement.\n\n  The compiled regex is kept in a cache shared by Match and Search.\n\n  Args:\n    pattern: regex pattern\n    rep: replacement text\n    s: search string\n\n  Returns:\n    string with replacements made (or original string if no replacements)\n  '
    if pattern not in _regexp_compile_cache:
        _regexp_compile_cache[pattern] = sre_compile.compile(pattern)
    return _regexp_compile_cache[pattern].sub(rep, s)

def Search(pattern, s):
    if False:
        return 10
    'Searches the string for the pattern, caching the compiled regexp.'
    if pattern not in _regexp_compile_cache:
        _regexp_compile_cache[pattern] = sre_compile.compile(pattern)
    return _regexp_compile_cache[pattern].search(s)

def _IsSourceExtension(s):
    if False:
        return 10
    'File extension (excluding dot) matches a source file extension.'
    return s in ('c', 'cc', 'cpp', 'cxx')

class _IncludeState(object):
    """Tracks line numbers for includes, and the order in which includes appear.

  include_list contains list of lists of (header, line number) pairs.
  It's a lists of lists rather than just one flat list to make it
  easier to update across preprocessor boundaries.

  Call CheckNextIncludeOrder() once for each header in the file, passing
  in the type constants defined above. Calls in an illegal order will
  raise an _IncludeError with an appropriate error message.

  """
    _INITIAL_SECTION = 0
    _MY_H_SECTION = 1
    _C_SECTION = 2
    _CPP_SECTION = 3
    _OTHER_H_SECTION = 4
    _TYPE_NAMES = {_C_SYS_HEADER: 'C system header', _CPP_SYS_HEADER: 'C++ system header', _LIKELY_MY_HEADER: 'header this file implements', _POSSIBLE_MY_HEADER: 'header this file may implement', _OTHER_HEADER: 'other header'}
    _SECTION_NAMES = {_INITIAL_SECTION: "... nothing. (This can't be an error.)", _MY_H_SECTION: 'a header this file implements', _C_SECTION: 'C system header', _CPP_SECTION: 'C++ system header', _OTHER_H_SECTION: 'other header'}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.include_list = [[]]
        self.ResetSection('')

    def FindHeader(self, header):
        if False:
            for i in range(10):
                print('nop')
        'Check if a header has already been included.\n\n    Args:\n      header: header to check.\n    Returns:\n      Line number of previous occurrence, or -1 if the header has not\n      been seen before.\n    '
        for section_list in self.include_list:
            for f in section_list:
                if f[0] == header:
                    return f[1]
        return -1

    def ResetSection(self, directive):
        if False:
            return 10
        'Reset section checking for preprocessor directive.\n\n    Args:\n      directive: preprocessor directive (e.g. "if", "else").\n    '
        self._section = self._INITIAL_SECTION
        self._last_header = ''
        if directive in ('if', 'ifdef', 'ifndef'):
            self.include_list.append([])
        elif directive in ('else', 'elif'):
            self.include_list[-1] = []

    def SetLastHeader(self, header_path):
        if False:
            while True:
                i = 10
        self._last_header = header_path

    def CanonicalizeAlphabeticalOrder(self, header_path):
        if False:
            for i in range(10):
                print('nop')
        'Returns a path canonicalized for alphabetical comparison.\n\n    - replaces "-" with "_" so they both cmp the same.\n    - removes \'-inl\' since we don\'t require them to be after the main header.\n    - lowercase everything, just in case.\n\n    Args:\n      header_path: Path to be canonicalized.\n\n    Returns:\n      Canonicalized path.\n    '
        return header_path.replace('-inl.h', '.h').replace('-', '_').lower()

    def IsInAlphabeticalOrder(self, clean_lines, linenum, header_path):
        if False:
            for i in range(10):
                print('nop')
        'Check if a header is in alphabetical order with the previous header.\n\n    Args:\n      clean_lines: A CleansedLines instance containing the file.\n      linenum: The number of the line to check.\n      header_path: Canonicalized header to be checked.\n\n    Returns:\n      Returns true if the header is in alphabetical order.\n    '
        if self._last_header > header_path and Match('^\\s*#\\s*include\\b', clean_lines.elided[linenum - 1]):
            return False
        return True

    def CheckNextIncludeOrder(self, header_type):
        if False:
            return 10
        "Returns a non-empty error message if the next header is out of order.\n\n    This function also updates the internal state to be ready to check\n    the next include.\n\n    Args:\n      header_type: One of the _XXX_HEADER constants defined above.\n\n    Returns:\n      The empty string if the header is in the right order, or an\n      error message describing what's wrong.\n\n    "
        error_message = 'Found %s after %s' % (self._TYPE_NAMES[header_type], self._SECTION_NAMES[self._section])
        last_section = self._section
        if header_type == _C_SYS_HEADER:
            if self._section <= self._C_SECTION:
                self._section = self._C_SECTION
            else:
                self._last_header = ''
                return error_message
        elif header_type == _CPP_SYS_HEADER:
            if self._section <= self._CPP_SECTION:
                self._section = self._CPP_SECTION
            else:
                self._last_header = ''
                return error_message
        elif header_type == _LIKELY_MY_HEADER:
            if self._section <= self._MY_H_SECTION:
                self._section = self._MY_H_SECTION
            else:
                self._section = self._OTHER_H_SECTION
        elif header_type == _POSSIBLE_MY_HEADER:
            if self._section <= self._MY_H_SECTION:
                self._section = self._MY_H_SECTION
            else:
                self._section = self._OTHER_H_SECTION
        else:
            assert header_type == _OTHER_HEADER
            self._section = self._OTHER_H_SECTION
        if last_section != self._section:
            self._last_header = ''
        return ''

class _CppLintState(object):
    """Maintains module-wide state.."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.verbose_level = 1
        self.error_count = 0
        self.filters = _DEFAULT_FILTERS[:]
        self._filters_backup = self.filters[:]
        self.counting = 'total'
        self.errors_by_category = {}
        self.quiet = False
        self.output_format = 'emacs'

    def SetOutputFormat(self, output_format):
        if False:
            while True:
                i = 10
        'Sets the output format for errors.'
        self.output_format = output_format

    def SetQuiet(self, quiet):
        if False:
            print('Hello World!')
        "Sets the module's quiet settings, and returns the previous setting."
        last_quiet = self.quiet
        self.quiet = quiet
        return last_quiet

    def SetVerboseLevel(self, level):
        if False:
            print('Hello World!')
        "Sets the module's verbosity, and returns the previous setting."
        last_verbose_level = self.verbose_level
        self.verbose_level = level
        return last_verbose_level

    def SetCountingStyle(self, counting_style):
        if False:
            return 10
        "Sets the module's counting options."
        self.counting = counting_style

    def SetFilters(self, filters):
        if False:
            i = 10
            return i + 15
        'Sets the error-message filters.\n\n    These filters are applied when deciding whether to emit a given\n    error message.\n\n    Args:\n      filters: A string of comma-separated filters (eg "+whitespace/indent").\n               Each filter should start with + or -; else we die.\n\n    Raises:\n      ValueError: The comma-separated filters did not all start with \'+\' or \'-\'.\n                  E.g. "-,+whitespace,-whitespace/indent,whitespace/badfilter"\n    '
        self.filters = _DEFAULT_FILTERS[:]
        self.AddFilters(filters)

    def AddFilters(self, filters):
        if False:
            print('Hello World!')
        ' Adds more filters to the existing list of error-message filters. '
        for filt in filters.split(','):
            clean_filt = filt.strip()
            if clean_filt:
                self.filters.append(clean_filt)
        for filt in self.filters:
            if not (filt.startswith('+') or filt.startswith('-')):
                raise ValueError('Every filter in --filters must start with + or - (%s does not)' % filt)

    def BackupFilters(self):
        if False:
            while True:
                i = 10
        ' Saves the current filter list to backup storage.'
        self._filters_backup = self.filters[:]

    def RestoreFilters(self):
        if False:
            return 10
        ' Restores filters previously backed up.'
        self.filters = self._filters_backup[:]

    def ResetErrorCounts(self):
        if False:
            print('Hello World!')
        "Sets the module's error statistic back to zero."
        self.error_count = 0
        self.errors_by_category = {}

    def IncrementErrorCount(self, category):
        if False:
            while True:
                i = 10
        "Bumps the module's error statistic."
        self.error_count += 1
        if self.counting in ('toplevel', 'detailed'):
            if self.counting != 'detailed':
                category = category.split('/')[0]
            if category not in self.errors_by_category:
                self.errors_by_category[category] = 0
            self.errors_by_category[category] += 1

    def PrintErrorCounts(self):
        if False:
            i = 10
            return i + 15
        'Print a summary of errors by category, and the total.'
        for (_, category, count) in self.errors_by_category.items():
            sys.stderr.write("Category '%s' errors found: %d\n" % (category, count))
        sys.stdout.write('Total errors found: %d\n' % self.error_count)
_cpplint_state = _CppLintState()

def _OutputFormat():
    if False:
        while True:
            i = 10
    "Gets the module's output format."
    return _cpplint_state.output_format

def _SetOutputFormat(output_format):
    if False:
        while True:
            i = 10
    "Sets the module's output format."
    _cpplint_state.SetOutputFormat(output_format)

def _Quiet():
    if False:
        while True:
            i = 10
    "Return's the module's quiet setting."
    return _cpplint_state.quiet

def _SetQuiet(quiet):
    if False:
        i = 10
        return i + 15
    "Set the module's quiet status, and return previous setting."
    return _cpplint_state.SetQuiet(quiet)

def _VerboseLevel():
    if False:
        print('Hello World!')
    "Returns the module's verbosity setting."
    return _cpplint_state.verbose_level

def _SetVerboseLevel(level):
    if False:
        return 10
    "Sets the module's verbosity, and returns the previous setting."
    return _cpplint_state.SetVerboseLevel(level)

def _SetCountingStyle(level):
    if False:
        return 10
    "Sets the module's counting options."
    _cpplint_state.SetCountingStyle(level)

def _Filters():
    if False:
        i = 10
        return i + 15
    "Returns the module's list of output filters, as a list."
    return _cpplint_state.filters

def _SetFilters(filters):
    if False:
        i = 10
        return i + 15
    'Sets the module\'s error-message filters.\n\n  These filters are applied when deciding whether to emit a given\n  error message.\n\n  Args:\n    filters: A string of comma-separated filters (eg "whitespace/indent").\n             Each filter should start with + or -; else we die.\n  '
    _cpplint_state.SetFilters(filters)

def _AddFilters(filters):
    if False:
        return 10
    'Adds more filter overrides.\n\n  Unlike _SetFilters, this function does not reset the current list of filters\n  available.\n\n  Args:\n    filters: A string of comma-separated filters (eg "whitespace/indent").\n             Each filter should start with + or -; else we die.\n  '
    _cpplint_state.AddFilters(filters)

def _BackupFilters():
    if False:
        print('Hello World!')
    ' Saves the current filter list to backup storage.'
    _cpplint_state.BackupFilters()

def _RestoreFilters():
    if False:
        i = 10
        return i + 15
    ' Restores filters previously backed up.'
    _cpplint_state.RestoreFilters()

class _FunctionState(object):
    """Tracks current function name and the number of lines in its body."""
    _NORMAL_TRIGGER = 250
    _TEST_TRIGGER = 400

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.in_a_function = False
        self.lines_in_function = 0
        self.current_function = ''

    def Begin(self, function_name):
        if False:
            i = 10
            return i + 15
        'Start analyzing function body.\n\n    Args:\n      function_name: The name of the function being tracked.\n    '
        self.in_a_function = True
        self.lines_in_function = 0
        self.current_function = function_name

    def Count(self):
        if False:
            for i in range(10):
                print('nop')
        'Count line in current function body.'
        if self.in_a_function:
            self.lines_in_function += 1

    def Check(self, error, filename, linenum):
        if False:
            i = 10
            return i + 15
        'Report if too many lines in function body.\n\n    Args:\n      error: The function to call with any errors found.\n      filename: The name of the current file.\n      linenum: The number of the line to check.\n    '
        if not self.in_a_function:
            return
        if Match('T(EST|est)', self.current_function):
            base_trigger = self._TEST_TRIGGER
        else:
            base_trigger = self._NORMAL_TRIGGER
        trigger = base_trigger * 2 ** _VerboseLevel()
        if self.lines_in_function > trigger:
            error_level = int(math.log(self.lines_in_function / base_trigger, 2))
            if error_level > 5:
                error_level = 5
            error(filename, linenum, 'readability/fn_size', error_level, 'Small and focused functions are preferred: %s has %d non-comment lines (error triggered by exceeding %d lines).' % (self.current_function, self.lines_in_function, trigger))

    def End(self):
        if False:
            i = 10
            return i + 15
        'Stop analyzing function body.'
        self.in_a_function = False

class _IncludeError(Exception):
    """Indicates a problem with the include order in a file."""
    pass

class FileInfo(object):
    """Provides utility functions for filenames.

  FileInfo provides easy access to the components of a file's path
  relative to the project root.
  """

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        self._filename = filename

    def FullName(self):
        if False:
            return 10
        'Make Windows paths like Unix.'
        return os.path.abspath(self._filename).replace('\\', '/')

    def RepositoryName(self):
        if False:
            return 10
        'FullName after removing the local path to the repository.\n\n    If we have a real absolute path name here we can try to do something smart:\n    detecting the root of the checkout and truncating /path/to/checkout from\n    the name so that we get header guards that don\'t include things like\n    "C:\\Documents and Settings\\..." or "/home/username/..." in them and thus\n    people on different computers who have checked the source out to different\n    locations won\'t see bogus errors.\n    '
        fullname = self.FullName()
        if os.path.exists(fullname):
            project_dir = os.path.dirname(fullname)
            if os.path.exists(os.path.join(project_dir, '.svn')):
                root_dir = project_dir
                one_up_dir = os.path.dirname(root_dir)
                while os.path.exists(os.path.join(one_up_dir, '.svn')):
                    root_dir = os.path.dirname(root_dir)
                    one_up_dir = os.path.dirname(one_up_dir)
                prefix = os.path.commonprefix([root_dir, project_dir])
                return fullname[len(prefix) + 1:]
            root_dir = current_dir = os.path.dirname(fullname)
            while current_dir != os.path.dirname(current_dir):
                if os.path.exists(os.path.join(current_dir, '.git')) or os.path.exists(os.path.join(current_dir, '.hg')) or os.path.exists(os.path.join(current_dir, '.svn')):
                    root_dir = current_dir
                current_dir = os.path.dirname(current_dir)
            if os.path.exists(os.path.join(root_dir, '.git')) or os.path.exists(os.path.join(root_dir, '.hg')) or os.path.exists(os.path.join(root_dir, '.svn')):
                prefix = os.path.commonprefix([root_dir, project_dir])
                return fullname[len(prefix) + 1:]
        return fullname

    def Split(self):
        if False:
            i = 10
            return i + 15
        "Splits the file into the directory, basename, and extension.\n\n    For 'chrome/browser/browser.cc', Split() would\n    return ('chrome/browser', 'browser', '.cc')\n\n    Returns:\n      A tuple of (directory, basename, extension).\n    "
        googlename = self.RepositoryName()
        (project, rest) = os.path.split(googlename)
        return (project,) + os.path.splitext(rest)

    def BaseName(self):
        if False:
            i = 10
            return i + 15
        'File base name - text after the final slash, before the final period.'
        return self.Split()[1]

    def Extension(self):
        if False:
            while True:
                i = 10
        'File extension - text following the final period.'
        return self.Split()[2]

    def NoExtension(self):
        if False:
            return 10
        'File has no source file extension.'
        return '/'.join(self.Split()[0:2])

    def IsSource(self):
        if False:
            i = 10
            return i + 15
        'File has a source file extension.'
        return _IsSourceExtension(self.Extension()[1:])

def _ShouldPrintError(category, confidence, linenum):
    if False:
        return 10
    'If confidence >= verbose, category passes filter and is not suppressed.'
    if IsErrorSuppressedByNolint(category, linenum):
        return False
    if confidence < _cpplint_state.verbose_level:
        return False
    is_filtered = False
    for one_filter in _Filters():
        if one_filter.startswith('-'):
            if category.startswith(one_filter[1:]):
                is_filtered = True
        elif one_filter.startswith('+'):
            if category.startswith(one_filter[1:]):
                is_filtered = False
        else:
            assert False
    if is_filtered:
        return False
    return True

def Error(filename, linenum, category, confidence, message):
    if False:
        while True:
            i = 10
    'Logs the fact we\'ve found a lint error.\n\n  We log where the error was found, and also our confidence in the error,\n  that is, how certain we are this is a legitimate style regression, and\n  not a misidentification or a use that\'s sometimes justified.\n\n  False positives can be suppressed by the use of\n  "cpplint(category)"  comments on the offending line.  These are\n  parsed into _error_suppressions.\n\n  Args:\n    filename: The name of the file containing the error.\n    linenum: The number of the line containing the error.\n    category: A string used to describe the "category" this bug\n      falls under: "whitespace", say, or "runtime".  Categories\n      may have a hierarchy separated by slashes: "whitespace/indent".\n    confidence: A number from 1-5 representing a confidence score for\n      the error, with 5 meaning that we are certain of the problem,\n      and 1 meaning that it could be a legitimate construct.\n    message: The error message.\n  '
    if _ShouldPrintError(category, confidence, linenum):
        _cpplint_state.IncrementErrorCount(category)
        if _cpplint_state.output_format == 'vs7':
            sys.stderr.write('%s(%s): error cpplint: [%s] %s [%d]\n' % (filename, linenum, category, message, confidence))
        elif _cpplint_state.output_format == 'eclipse':
            sys.stderr.write('%s:%s: warning: %s  [%s] [%d]\n' % (filename, linenum, message, category, confidence))
        else:
            sys.stderr.write('%s:%s:  %s  [%s] [%d]\n' % (filename, linenum, message, category, confidence))
_RE_PATTERN_CLEANSE_LINE_ESCAPES = re.compile('\\\\([abfnrtv?"\\\\\\\']|\\d+|x[0-9a-fA-F]+)')
_RE_PATTERN_C_COMMENTS = '/\\*(?:[^*]|\\*(?!/))*\\*/'
_RE_PATTERN_CLEANSE_LINE_C_COMMENTS = re.compile('(\\s*' + _RE_PATTERN_C_COMMENTS + '\\s*$|' + _RE_PATTERN_C_COMMENTS + '\\s+|' + '\\s+' + _RE_PATTERN_C_COMMENTS + '(?=\\W)|' + _RE_PATTERN_C_COMMENTS + ')')

def IsCppString(line):
    if False:
        while True:
            i = 10
    "Does line terminate so, that the next symbol is in string constant.\n\n  This function does not consider single-line nor multi-line comments.\n\n  Args:\n    line: is a partial line of code starting from the 0..n.\n\n  Returns:\n    True, if next character appended to 'line' is inside a\n    string constant.\n  "
    line = line.replace('\\\\', 'XX')
    return line.count('"') - line.count('\\"') - line.count('\'"\'') & 1 == 1

def CleanseRawStrings(raw_lines):
    if False:
        while True:
            i = 10
    'Removes C++11 raw strings from lines.\n\n    Before:\n      static const char kData[] = R"(\n          multi-line string\n          )";\n\n    After:\n      static const char kData[] = ""\n          (replaced by blank line)\n          "";\n\n  Args:\n    raw_lines: list of raw lines.\n\n  Returns:\n    list of lines with C++11 raw strings replaced by empty strings.\n  '
    delimiter = None
    lines_without_raw_strings = []
    for line in raw_lines:
        if delimiter:
            end = line.find(delimiter)
            if end >= 0:
                leading_space = Match('^(\\s*)\\S', line)
                line = leading_space.group(1) + '""' + line[end + len(delimiter):]
                delimiter = None
            else:
                line = '""'
        while delimiter is None:
            matched = Match('^(.*?)\\b(?:R|u8R|uR|UR|LR)"([^\\s\\\\()]*)\\((.*)$', line)
            if matched and (not Match('^([^\\\'"]|\\\'(\\\\.|[^\\\'])*\\\'|"(\\\\.|[^"])*")*//', matched.group(1))):
                delimiter = ')' + matched.group(2) + '"'
                end = matched.group(3).find(delimiter)
                if end >= 0:
                    line = matched.group(1) + '""' + matched.group(3)[end + len(delimiter):]
                    delimiter = None
                else:
                    line = matched.group(1) + '""'
            else:
                break
        lines_without_raw_strings.append(line)
    return lines_without_raw_strings

def FindNextMultiLineCommentStart(lines, lineix):
    if False:
        print('Hello World!')
    'Find the beginning marker for a multiline comment.'
    while lineix < len(lines):
        if lines[lineix].strip().startswith('/*'):
            if lines[lineix].strip().find('*/', 2) < 0:
                return lineix
        lineix += 1
    return len(lines)

def FindNextMultiLineCommentEnd(lines, lineix):
    if False:
        for i in range(10):
            print('nop')
    'We are inside a comment, find the end marker.'
    while lineix < len(lines):
        if lines[lineix].strip().endswith('*/'):
            return lineix
        lineix += 1
    return len(lines)

def RemoveMultiLineCommentsFromRange(lines, begin, end):
    if False:
        print('Hello World!')
    'Clears a range of lines for multi-line comments.'
    for i in range(begin, end):
        lines[i] = '/**/'

def RemoveMultiLineComments(filename, lines, error):
    if False:
        return 10
    'Removes multiline (c-style) comments from lines.'
    lineix = 0
    while lineix < len(lines):
        lineix_begin = FindNextMultiLineCommentStart(lines, lineix)
        if lineix_begin >= len(lines):
            return
        lineix_end = FindNextMultiLineCommentEnd(lines, lineix_begin)
        if lineix_end >= len(lines):
            error(filename, lineix_begin + 1, 'readability/multiline_comment', 5, 'Could not find end of multi-line comment')
            return
        RemoveMultiLineCommentsFromRange(lines, lineix_begin, lineix_end + 1)
        lineix = lineix_end + 1

def CleanseComments(line):
    if False:
        return 10
    'Removes //-comments and single-line C-style /* */ comments.\n\n  Args:\n    line: A line of C++ source.\n\n  Returns:\n    The line with single-line comments removed.\n  '
    commentpos = line.find('//')
    if commentpos != -1 and (not IsCppString(line[:commentpos])):
        line = line[:commentpos].rstrip()
    return _RE_PATTERN_CLEANSE_LINE_C_COMMENTS.sub('', line)

class CleansedLines(object):
    """Holds 4 copies of all lines with different preprocessing applied to them.

  1) elided member contains lines without strings and comments.
  2) lines member contains lines without comments.
  3) raw_lines member contains all the lines without processing.
  4) lines_without_raw_strings member is same as raw_lines, but with C++11 raw
     strings removed.
  All these members are of <type 'list'>, and of the same length.
  """

    def __init__(self, lines):
        if False:
            while True:
                i = 10
        self.elided = []
        self.lines = []
        self.raw_lines = lines
        self.num_lines = len(lines)
        self.lines_without_raw_strings = CleanseRawStrings(lines)
        for linenum in range(len(self.lines_without_raw_strings)):
            self.lines.append(CleanseComments(self.lines_without_raw_strings[linenum]))
            elided = self._CollapseStrings(self.lines_without_raw_strings[linenum])
            self.elided.append(CleanseComments(elided))

    def NumLines(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of lines represented.'
        return self.num_lines

    @staticmethod
    def _CollapseStrings(elided):
        if False:
            print('Hello World!')
        'Collapses strings and chars on a line to simple "" or \'\' blocks.\n\n    We nix strings first so we\'re not fooled by text like \'"http://"\'\n\n    Args:\n      elided: The line being processed.\n\n    Returns:\n      The line with collapsed strings.\n    '
        if _RE_PATTERN_INCLUDE.match(elided):
            return elided
        elided = _RE_PATTERN_CLEANSE_LINE_ESCAPES.sub('', elided)
        collapsed = ''
        while True:
            match = Match('^([^\\\'"]*)([\\\'"])(.*)$', elided)
            if not match:
                collapsed += elided
                break
            (head, quote, tail) = match.groups()
            if quote == '"':
                second_quote = tail.find('"')
                if second_quote >= 0:
                    collapsed += head + '""'
                    elided = tail[second_quote + 1:]
                else:
                    collapsed += elided
                    break
            elif Search('\\b(?:0[bBxX]?|[1-9])[0-9a-fA-F]*$', head):
                match_literal = Match("^((?:\\'?[0-9a-zA-Z_])*)(.*)$", "'" + tail)
                collapsed += head + match_literal.group(1).replace("'", '')
                elided = match_literal.group(2)
            else:
                second_quote = tail.find("'")
                if second_quote >= 0:
                    collapsed += head + "''"
                    elided = tail[second_quote + 1:]
                else:
                    collapsed += elided
                    break
        return collapsed

def FindEndOfExpressionInLine(line, startpos, stack):
    if False:
        print('Hello World!')
    'Find the position just after the end of current parenthesized expression.\n\n  Args:\n    line: a CleansedLines line.\n    startpos: start searching at this position.\n    stack: nesting stack at startpos.\n\n  Returns:\n    On finding matching end: (index just after matching end, None)\n    On finding an unclosed expression: (-1, None)\n    Otherwise: (-1, new stack at end of this line)\n  '
    for i in range(startpos, len(line)):
        char = line[i]
        if char in '([{':
            stack.append(char)
        elif char == '<':
            if i > 0 and line[i - 1] == '<':
                if stack and stack[-1] == '<':
                    stack.pop()
                    if not stack:
                        return (-1, None)
            elif i > 0 and Search('\\boperator\\s*$', line[0:i]):
                continue
            else:
                stack.append('<')
        elif char in ')]}':
            while stack and stack[-1] == '<':
                stack.pop()
            if not stack:
                return (-1, None)
            if stack[-1] == '(' and char == ')' or (stack[-1] == '[' and char == ']') or (stack[-1] == '{' and char == '}'):
                stack.pop()
                if not stack:
                    return (i + 1, None)
            else:
                return (-1, None)
        elif char == '>':
            if i > 0 and (line[i - 1] == '-' or Search('\\boperator\\s*$', line[0:i - 1])):
                continue
            if stack:
                if stack[-1] == '<':
                    stack.pop()
                    if not stack:
                        return (i + 1, None)
        elif char == ';':
            while stack and stack[-1] == '<':
                stack.pop()
            if not stack:
                return (-1, None)
    return (-1, stack)

def CloseExpression(clean_lines, linenum, pos):
    if False:
        while True:
            i = 10
    "If input points to ( or { or [ or <, finds the position that closes it.\n\n  If lines[linenum][pos] points to a '(' or '{' or '[' or '<', finds the\n  linenum/pos that correspond to the closing of the expression.\n\n  TODO(unknown): cpplint spends a fair bit of time matching parentheses.\n  Ideally we would want to index all opening and closing parentheses once\n  and have CloseExpression be just a simple lookup, but due to preprocessor\n  tricks, this is not so easy.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    pos: A position on the line.\n\n  Returns:\n    A tuple (line, linenum, pos) pointer *past* the closing brace, or\n    (line, len(lines), -1) if we never find a close.  Note we ignore\n    strings and comments when matching; and the line we return is the\n    'cleansed' line at linenum.\n  "
    line = clean_lines.elided[linenum]
    if line[pos] not in '({[<' or Match('<[<=]', line[pos:]):
        return (line, clean_lines.NumLines(), -1)
    (end_pos, stack) = FindEndOfExpressionInLine(line, pos, [])
    if end_pos > -1:
        return (line, linenum, end_pos)
    while stack and linenum < clean_lines.NumLines() - 1:
        linenum += 1
        line = clean_lines.elided[linenum]
        (end_pos, stack) = FindEndOfExpressionInLine(line, 0, stack)
        if end_pos > -1:
            return (line, linenum, end_pos)
    return (line, clean_lines.NumLines(), -1)

def FindStartOfExpressionInLine(line, endpos, stack):
    if False:
        for i in range(10):
            print('nop')
    'Find position at the matching start of current expression.\n\n  This is almost the reverse of FindEndOfExpressionInLine, but note\n  that the input position and returned position differs by 1.\n\n  Args:\n    line: a CleansedLines line.\n    endpos: start searching at this position.\n    stack: nesting stack at endpos.\n\n  Returns:\n    On finding matching start: (index at matching start, None)\n    On finding an unclosed expression: (-1, None)\n    Otherwise: (-1, new stack at beginning of this line)\n  '
    i = endpos
    while i >= 0:
        char = line[i]
        if char in ')]}':
            stack.append(char)
        elif char == '>':
            if i > 0 and (line[i - 1] == '-' or Match('\\s>=\\s', line[i - 1:]) or Search('\\boperator\\s*$', line[0:i])):
                i -= 1
            else:
                stack.append('>')
        elif char == '<':
            if i > 0 and line[i - 1] == '<':
                i -= 1
            elif stack and stack[-1] == '>':
                stack.pop()
                if not stack:
                    return (i, None)
        elif char in '([{':
            while stack and stack[-1] == '>':
                stack.pop()
            if not stack:
                return (-1, None)
            if char == '(' and stack[-1] == ')' or (char == '[' and stack[-1] == ']') or (char == '{' and stack[-1] == '}'):
                stack.pop()
                if not stack:
                    return (i, None)
            else:
                return (-1, None)
        elif char == ';':
            while stack and stack[-1] == '>':
                stack.pop()
            if not stack:
                return (-1, None)
        i -= 1
    return (-1, stack)

def ReverseCloseExpression(clean_lines, linenum, pos):
    if False:
        while True:
            i = 10
    "If input points to ) or } or ] or >, finds the position that opens it.\n\n  If lines[linenum][pos] points to a ')' or '}' or ']' or '>', finds the\n  linenum/pos that correspond to the opening of the expression.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    pos: A position on the line.\n\n  Returns:\n    A tuple (line, linenum, pos) pointer *at* the opening brace, or\n    (line, 0, -1) if we never find the matching opening brace.  Note\n    we ignore strings and comments when matching; and the line we\n    return is the 'cleansed' line at linenum.\n  "
    line = clean_lines.elided[linenum]
    if line[pos] not in ')}]>':
        return (line, 0, -1)
    (start_pos, stack) = FindStartOfExpressionInLine(line, pos, [])
    if start_pos > -1:
        return (line, linenum, start_pos)
    while stack and linenum > 0:
        linenum -= 1
        line = clean_lines.elided[linenum]
        (start_pos, stack) = FindStartOfExpressionInLine(line, len(line) - 1, stack)
        if start_pos > -1:
            return (line, linenum, start_pos)
    return (line, 0, -1)

def CheckForCopyright(filename, lines, error):
    if False:
        return 10
    'Logs an error if no Copyright message appears at the top of the file.'
    for line in range(1, min(len(lines), 11)):
        if re.search('Copyright', lines[line], re.I):
            break
    else:
        error(filename, 0, 'legal/copyright', 5, 'No copyright message found.  You should have a line: "Copyright [year] <Copyright Owner>"')

def GetIndentLevel(line):
    if False:
        while True:
            i = 10
    'Return the number of leading spaces in line.\n\n  Args:\n    line: A string to check.\n\n  Returns:\n    An integer count of leading spaces, possibly zero.\n  '
    indent = Match('^( *)\\S', line)
    if indent:
        return len(indent.group(1))
    else:
        return 0

def PathSplitToList(path):
    if False:
        for i in range(10):
            print('nop')
    "Returns the path split into a list by the separator.\n\n  Args:\n    path: An absolute or relative path (e.g. '/a/b/c/' or '../a')\n\n  Returns:\n    A list of path components (e.g. ['a', 'b', 'c]).\n  "
    lst = []
    while True:
        (head, tail) = os.path.split(path)
        if head == path:
            lst.append(head)
            break
        if tail == path:
            lst.append(tail)
            break
        path = head
        lst.append(tail)
    lst.reverse()
    return lst

def GetHeaderGuardCPPVariable(filename):
    if False:
        print('Hello World!')
    'Returns the CPP variable that should be used as a header guard.\n\n  Args:\n    filename: The name of a C++ header file.\n\n  Returns:\n    The CPP variable that should be used as a header guard in the\n    named file.\n\n  '
    filename = re.sub('_flymake\\.h$', '.h', filename)
    filename = re.sub('/\\.flymake/([^/]*)$', '/\\1', filename)
    filename = filename.replace('C++', 'cpp').replace('c++', 'cpp')
    fileinfo = FileInfo(filename)
    file_path_from_root = fileinfo.RepositoryName()

    def FixupPathFromRoot():
        if False:
            i = 10
            return i + 15
        if _root_debug:
            sys.stderr.write("\n_root fixup, _root = '%s', repository name = '%s'\n" % (_root, fileinfo.RepositoryName()))
        if not _root:
            if _root_debug:
                sys.stderr.write('_root unspecified\n')
            return file_path_from_root

        def StripListPrefix(lst, prefix):
            if False:
                for i in range(10):
                    print('nop')
            if lst[:len(prefix)] != prefix:
                return None
            return lst[len(prefix):]
        maybe_path = StripListPrefix(PathSplitToList(file_path_from_root), PathSplitToList(_root))
        if _root_debug:
            sys.stderr.write('_root lstrip (maybe_path=%s, file_path_from_root=%s,' + ' _root=%s)\n' % (maybe_path, file_path_from_root, _root))
        if maybe_path:
            return os.path.join(*maybe_path)
        full_path = fileinfo.FullName()
        root_abspath = os.path.abspath(_root)
        maybe_path = StripListPrefix(PathSplitToList(full_path), PathSplitToList(root_abspath))
        if _root_debug:
            sys.stderr.write('_root prepend (maybe_path=%s, full_path=%s, ' + 'root_abspath=%s)\n' % (maybe_path, full_path, root_abspath))
        if maybe_path:
            return os.path.join(*maybe_path)
        if _root_debug:
            sys.stderr.write('_root ignore, returning %s\n' % file_path_from_root)
        return file_path_from_root
    file_path_from_root = FixupPathFromRoot()
    return re.sub('[^a-zA-Z0-9]', '_', file_path_from_root).upper() + '_'

def CheckForHeaderGuard(filename, clean_lines, error):
    if False:
        print('Hello World!')
    'Checks that the file contains a header guard.\n\n  Logs an error if no #ifndef header guard is present.  For other\n  headers, checks that the full pathname is used.\n\n  Args:\n    filename: The name of the C++ header file.\n    clean_lines: A CleansedLines instance containing the file.\n    error: The function to call with any errors found.\n  '
    raw_lines = clean_lines.lines_without_raw_strings
    for i in raw_lines:
        if Search('//\\s*NOLINT\\(build/header_guard\\)', i):
            return
    cppvar = GetHeaderGuardCPPVariable(filename)
    ifndef = ''
    ifndef_linenum = 0
    define = ''
    endif = ''
    endif_linenum = 0
    for (linenum, line) in enumerate(raw_lines):
        linesplit = line.split()
        if len(linesplit) >= 2:
            if not ifndef and linesplit[0] == '#ifndef':
                ifndef = linesplit[1]
                ifndef_linenum = linenum
            if not define and linesplit[0] == '#define':
                define = linesplit[1]
        if line.startswith('#endif'):
            endif = line
            endif_linenum = linenum
    if not ifndef or not define or ifndef != define:
        error(filename, 0, 'build/header_guard', 5, 'No #ifndef header guard found, suggested CPP variable is: %s' % cppvar)
        return
    if ifndef != cppvar:
        error_level = 0
        if ifndef != cppvar + '_':
            error_level = 5
        ParseNolintSuppressions(filename, raw_lines[ifndef_linenum], ifndef_linenum, error)
        error(filename, ifndef_linenum, 'build/header_guard', error_level, '#ifndef header guard has wrong style, please use: %s' % cppvar)
    ParseNolintSuppressions(filename, raw_lines[endif_linenum], endif_linenum, error)
    match = Match('#endif\\s*//\\s*' + cppvar + '(_)?\\b', endif)
    if match:
        if match.group(1) == '_':
            error(filename, endif_linenum, 'build/header_guard', 0, '#endif line should be "#endif  // %s"' % cppvar)
        return
    no_single_line_comments = True
    for i in range(1, len(raw_lines) - 1):
        line = raw_lines[i]
        if Match('^(?:(?:\\\'(?:\\.|[^\\\'])*\\\')|(?:"(?:\\.|[^"])*")|[^\\\'"])*//', line):
            no_single_line_comments = False
            break
    if no_single_line_comments:
        match = Match('#endif\\s*/\\*\\s*' + cppvar + '(_)?\\s*\\*/', endif)
        if match:
            if match.group(1) == '_':
                error(filename, endif_linenum, 'build/header_guard', 0, '#endif line should be "#endif  /* %s */"' % cppvar)
            return
    error(filename, endif_linenum, 'build/header_guard', 5, '#endif line should be "#endif  // %s"' % cppvar)

def CheckHeaderFileIncluded(filename, include_state, error):
    if False:
        print('Hello World!')
    'Logs an error if a .cc file does not include its header.'
    fileinfo = FileInfo(filename)
    if Search(_TEST_FILE_SUFFIX, fileinfo.BaseName()):
        return
    headerfile = filename[0:len(filename) - len(fileinfo.Extension())] + '.h'
    if not os.path.exists(headerfile):
        return
    headername = FileInfo(headerfile).RepositoryName()
    first_include = 0
    for section_list in include_state.include_list:
        for f in section_list:
            if headername in f[0] or f[0] in headername:
                return
            if not first_include:
                first_include = f[1]
    error(filename, first_include, 'build/include', 5, '%s should include its header file %s' % (fileinfo.RepositoryName(), headername))

def CheckForBadCharacters(filename, lines, error):
    if False:
        for i in range(10):
            print('nop')
    "Logs an error for each line containing bad characters.\n\n  Two kinds of bad characters:\n\n  1. Unicode replacement characters: These indicate that either the file\n  contained invalid UTF-8 (likely) or Unicode replacement characters (which\n  it shouldn't).  Note that it's possible for this to throw off line\n  numbering if the invalid UTF-8 occurred adjacent to a newline.\n\n  2. NUL bytes.  These are problematic for some tools.\n\n  Args:\n    filename: The name of the current file.\n    lines: An array of strings, each representing a line of the file.\n    error: The function to call with any errors found.\n  "
    for (linenum, line) in enumerate(lines):
        if u'�' in line:
            error(filename, linenum, 'readability/utf8', 5, 'Line contains invalid UTF-8 (or Unicode replacement character).')
        if '\x00' in line:
            error(filename, linenum, 'readability/nul', 5, 'Line contains NUL byte.')

def CheckForNewlineAtEOF(filename, lines, error):
    if False:
        print('Hello World!')
    'Logs an error if there is no newline char at the end of the file.\n\n  Args:\n    filename: The name of the current file.\n    lines: An array of strings, each representing a line of the file.\n    error: The function to call with any errors found.\n  '
    if len(lines) < 3 or lines[-2]:
        error(filename, len(lines) - 2, 'whitespace/ending_newline', 5, 'Could not find a newline character at the end of the file.')

def CheckForMultilineCommentsAndStrings(filename, clean_lines, linenum, error):
    if False:
        print('Hello World!')
    'Logs an error if we see /* ... */ or "..." that extend past one line.\n\n  /* ... */ comments are legit inside macros, for one line.\n  Otherwise, we prefer // comments, so it\'s ok to warn about the\n  other.  Likewise, it\'s ok for strings to extend across multiple\n  lines, as long as a line continuation character (backslash)\n  terminates each line. Although not currently prohibited by the C++\n  style guide, it\'s ugly and unnecessary. We don\'t do well with either\n  in this lint program, so we warn about both.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    line = line.replace('\\\\', '')
    if line.count('/*') > line.count('*/'):
        error(filename, linenum, 'readability/multiline_comment', 5, 'Complex multi-line /*...*/-style comment found. Lint may give bogus warnings.  Consider replacing these with //-style comments, with #if 0...#endif, or with more clearly structured multi-line comments.')
    if (line.count('"') - line.count('\\"')) % 2:
        error(filename, linenum, 'readability/multiline_string', 5, 'Multi-line string ("...") found.  This lint script doesn\'t do well with such strings, and may give bogus warnings.  Use C++11 raw strings or concatenation instead.')
_UNSAFE_FUNC_PREFIX = '(?:[-+*/=%^&|(<]\\s*|>\\s+)'
_THREADING_LIST = (('asctime(', 'asctime_r(', _UNSAFE_FUNC_PREFIX + 'asctime\\([^)]+\\)'), ('ctime(', 'ctime_r(', _UNSAFE_FUNC_PREFIX + 'ctime\\([^)]+\\)'), ('getgrgid(', 'getgrgid_r(', _UNSAFE_FUNC_PREFIX + 'getgrgid\\([^)]+\\)'), ('getgrnam(', 'getgrnam_r(', _UNSAFE_FUNC_PREFIX + 'getgrnam\\([^)]+\\)'), ('getlogin(', 'getlogin_r(', _UNSAFE_FUNC_PREFIX + 'getlogin\\(\\)'), ('getpwnam(', 'getpwnam_r(', _UNSAFE_FUNC_PREFIX + 'getpwnam\\([^)]+\\)'), ('getpwuid(', 'getpwuid_r(', _UNSAFE_FUNC_PREFIX + 'getpwuid\\([^)]+\\)'), ('gmtime(', 'gmtime_r(', _UNSAFE_FUNC_PREFIX + 'gmtime\\([^)]+\\)'), ('localtime(', 'localtime_r(', _UNSAFE_FUNC_PREFIX + 'localtime\\([^)]+\\)'), ('rand(', 'rand_r(', _UNSAFE_FUNC_PREFIX + 'rand\\(\\)'), ('strtok(', 'strtok_r(', _UNSAFE_FUNC_PREFIX + 'strtok\\([^)]+\\)'), ('ttyname(', 'ttyname_r(', _UNSAFE_FUNC_PREFIX + 'ttyname\\([^)]+\\)'))

def CheckPosixThreading(filename, clean_lines, linenum, error):
    if False:
        print('Hello World!')
    'Checks for calls to thread-unsafe functions.\n\n  Much code has been originally written without consideration of\n  multi-threading. Also, engineers are relying on their old experience;\n  they have learned posix before threading extensions were added. These\n  tests guide the engineers to use thread-safe functions (when using\n  posix directly).\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    for (single_thread_func, multithread_safe_func, pattern) in _THREADING_LIST:
        if Search(pattern, line):
            error(filename, linenum, 'runtime/threadsafe_fn', 2, 'Consider using ' + multithread_safe_func + '...) instead of ' + single_thread_func + '...) for improved thread safety.')

def CheckVlogArguments(filename, clean_lines, linenum, error):
    if False:
        while True:
            i = 10
    'Checks that VLOG() is only used for defining a logging level.\n\n  For example, VLOG(2) is correct. VLOG(INFO), VLOG(WARNING), VLOG(ERROR), and\n  VLOG(FATAL) are not.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    if Search('\\bVLOG\\((INFO|ERROR|WARNING|DFATAL|FATAL)\\)', line):
        error(filename, linenum, 'runtime/vlog', 5, 'VLOG() should be used with numeric verbosity level.  Use LOG() if you want symbolic severity levels.')
_RE_PATTERN_INVALID_INCREMENT = re.compile('^\\s*\\*\\w+(\\+\\+|--);')

def CheckInvalidIncrement(filename, clean_lines, linenum, error):
    if False:
        for i in range(10):
            print('nop')
    'Checks for invalid increment *count++.\n\n  For example following function:\n  void increment_counter(int* count) {\n    *count++;\n  }\n  is invalid, because it effectively does count++, moving pointer, and should\n  be replaced with ++*count, (*count)++ or *count += 1.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    if _RE_PATTERN_INVALID_INCREMENT.match(line):
        error(filename, linenum, 'runtime/invalid_increment', 5, 'Changing pointer instead of value (or unused value of operator*).')

def IsMacroDefinition(clean_lines, linenum):
    if False:
        i = 10
        return i + 15
    if Search('^#define', clean_lines[linenum]):
        return True
    if linenum > 0 and Search('\\\\$', clean_lines[linenum - 1]):
        return True
    return False

def IsForwardClassDeclaration(clean_lines, linenum):
    if False:
        for i in range(10):
            print('nop')
    return Match('^\\s*(\\btemplate\\b)*.*class\\s+\\w+;\\s*$', clean_lines[linenum])

class _BlockInfo(object):
    """Stores information about a generic block of code."""

    def __init__(self, linenum, seen_open_brace):
        if False:
            while True:
                i = 10
        self.starting_linenum = linenum
        self.seen_open_brace = seen_open_brace
        self.open_parentheses = 0
        self.inline_asm = _NO_ASM
        self.check_namespace_indentation = False

    def CheckBegin(self, filename, clean_lines, linenum, error):
        if False:
            return 10
        'Run checks that applies to text up to the opening brace.\n\n    This is mostly for checking the text after the class identifier\n    and the "{", usually where the base class is specified.  For other\n    blocks, there isn\'t much to check, so we always pass.\n\n    Args:\n      filename: The name of the current file.\n      clean_lines: A CleansedLines instance containing the file.\n      linenum: The number of the line to check.\n      error: The function to call with any errors found.\n    '
        pass

    def CheckEnd(self, filename, clean_lines, linenum, error):
        if False:
            i = 10
            return i + 15
        'Run checks that applies to text after the closing brace.\n\n    This is mostly used for checking end of namespace comments.\n\n    Args:\n      filename: The name of the current file.\n      clean_lines: A CleansedLines instance containing the file.\n      linenum: The number of the line to check.\n      error: The function to call with any errors found.\n    '
        pass

    def IsBlockInfo(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns true if this block is a _BlockInfo.\n\n    This is convenient for verifying that an object is an instance of\n    a _BlockInfo, but not an instance of any of the derived classes.\n\n    Returns:\n      True for this class, False for derived classes.\n    '
        return self.__class__ == _BlockInfo

class _ExternCInfo(_BlockInfo):
    """Stores information about an 'extern "C"' block."""

    def __init__(self, linenum):
        if False:
            return 10
        _BlockInfo.__init__(self, linenum, True)

class _ClassInfo(_BlockInfo):
    """Stores information about a class."""

    def __init__(self, name, class_or_struct, clean_lines, linenum):
        if False:
            i = 10
            return i + 15
        _BlockInfo.__init__(self, linenum, False)
        self.name = name
        self.is_derived = False
        self.check_namespace_indentation = True
        if class_or_struct == 'struct':
            self.access = 'public'
            self.is_struct = True
        else:
            self.access = 'private'
            self.is_struct = False
        self.class_indent = GetIndentLevel(clean_lines.raw_lines[linenum])
        self.last_line = 0
        depth = 0
        for i in range(linenum, clean_lines.NumLines()):
            line = clean_lines.elided[i]
            depth += line.count('{') - line.count('}')
            if not depth:
                self.last_line = i
                break

    def CheckBegin(self, filename, clean_lines, linenum, error):
        if False:
            while True:
                i = 10
        if Search('(^|[^:]):($|[^:])', clean_lines.elided[linenum]):
            self.is_derived = True

    def CheckEnd(self, filename, clean_lines, linenum, error):
        if False:
            for i in range(10):
                print('nop')
        seen_last_thing_in_class = False
        for i in range(linenum - 1, self.starting_linenum, -1):
            match = Search('\\b(DISALLOW_COPY_AND_ASSIGN|DISALLOW_IMPLICIT_CONSTRUCTORS)\\(' + self.name + '\\)', clean_lines.elided[i])
            if match:
                if seen_last_thing_in_class:
                    error(filename, i, 'readability/constructors', 3, match.group(1) + ' should be the last thing in the class')
                break
            if not Match('^\\s*$', clean_lines.elided[i]):
                seen_last_thing_in_class = True
        indent = Match('^( *)\\}', clean_lines.elided[linenum])
        if indent and len(indent.group(1)) != self.class_indent:
            if self.is_struct:
                parent = 'struct ' + self.name
            else:
                parent = 'class ' + self.name
            error(filename, linenum, 'whitespace/indent', 3, 'Closing brace should be aligned with beginning of %s' % parent)

class _NamespaceInfo(_BlockInfo):
    """Stores information about a namespace."""

    def __init__(self, name, linenum):
        if False:
            return 10
        _BlockInfo.__init__(self, linenum, False)
        self.name = name or ''
        self.check_namespace_indentation = True

    def CheckEnd(self, filename, clean_lines, linenum, error):
        if False:
            return 10
        'Check end of namespace comments.'
        line = clean_lines.raw_lines[linenum]
        if linenum - self.starting_linenum < 10 and (not Match('^\\s*};*\\s*(//|/\\*).*\\bnamespace\\b', line)):
            return
        if self.name:
            if not Match('^\\s*};*\\s*(//|/\\*).*\\bnamespace\\s+' + re.escape(self.name) + '[\\*/\\.\\\\\\s]*$', line):
                error(filename, linenum, 'readability/namespace', 5, 'Namespace should be terminated with "// namespace %s"' % self.name)
        elif not Match('^\\s*};*\\s*(//|/\\*).*\\bnamespace[\\*/\\.\\\\\\s]*$', line):
            if Match('^\\s*}.*\\b(namespace anonymous|anonymous namespace)\\b', line):
                error(filename, linenum, 'readability/namespace', 5, 'Anonymous namespace should be terminated with "// namespace" or "// anonymous namespace"')
            else:
                error(filename, linenum, 'readability/namespace', 5, 'Anonymous namespace should be terminated with "// namespace"')

class _PreprocessorInfo(object):
    """Stores checkpoints of nesting stacks when #if/#else is seen."""

    def __init__(self, stack_before_if):
        if False:
            i = 10
            return i + 15
        self.stack_before_if = stack_before_if
        self.stack_before_else = []
        self.seen_else = False

class NestingState(object):
    """Holds states related to parsing braces."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.stack = []
        self.previous_stack_top = []
        self.pp_stack = []

    def SeenOpenBrace(self):
        if False:
            return 10
        'Check if we have seen the opening brace for the innermost block.\n\n    Returns:\n      True if we have seen the opening brace, False if the innermost\n      block is still expecting an opening brace.\n    '
        return not self.stack or self.stack[-1].seen_open_brace

    def InNamespaceBody(self):
        if False:
            while True:
                i = 10
        'Check if we are currently one level inside a namespace body.\n\n    Returns:\n      True if top of the stack is a namespace block, False otherwise.\n    '
        return self.stack and isinstance(self.stack[-1], _NamespaceInfo)

    def InExternC(self):
        if False:
            i = 10
            return i + 15
        'Check if we are currently one level inside an \'extern "C"\' block.\n\n    Returns:\n      True if top of the stack is an extern block, False otherwise.\n    '
        return self.stack and isinstance(self.stack[-1], _ExternCInfo)

    def InClassDeclaration(self):
        if False:
            return 10
        'Check if we are currently one level inside a class or struct declaration.\n\n    Returns:\n      True if top of the stack is a class/struct, False otherwise.\n    '
        return self.stack and isinstance(self.stack[-1], _ClassInfo)

    def InAsmBlock(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if we are currently one level inside an inline ASM block.\n\n    Returns:\n      True if the top of the stack is a block containing inline ASM.\n    '
        return self.stack and self.stack[-1].inline_asm != _NO_ASM

    def InTemplateArgumentList(self, clean_lines, linenum, pos):
        if False:
            while True:
                i = 10
        'Check if current position is inside template argument list.\n\n    Args:\n      clean_lines: A CleansedLines instance containing the file.\n      linenum: The number of the line to check.\n      pos: position just after the suspected template argument.\n    Returns:\n      True if (linenum, pos) is inside template arguments.\n    '
        while linenum < clean_lines.NumLines():
            line = clean_lines.elided[linenum]
            match = Match('^[^{};=\\[\\]\\.<>]*(.)', line[pos:])
            if not match:
                linenum += 1
                pos = 0
                continue
            token = match.group(1)
            pos += len(match.group(0))
            if token in ('{', '}', ';'):
                return False
            if token in ('>', '=', '[', ']', '.'):
                return True
            if token != '<':
                pos += 1
                if pos >= len(line):
                    linenum += 1
                    pos = 0
                continue
            (_, end_line, end_pos) = CloseExpression(clean_lines, linenum, pos - 1)
            if end_pos < 0:
                return False
            linenum = end_line
            pos = end_pos
        return False

    def UpdatePreprocessor(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Update preprocessor stack.\n\n    We need to handle preprocessors due to classes like this:\n      #ifdef SWIG\n      struct ResultDetailsPageElementExtensionPoint {\n      #else\n      struct ResultDetailsPageElementExtensionPoint : public Extension {\n      #endif\n\n    We make the following assumptions (good enough for most files):\n    - Preprocessor condition evaluates to true from #if up to first\n      #else/#elif/#endif.\n\n    - Preprocessor condition evaluates to false from #else/#elif up\n      to #endif.  We still perform lint checks on these lines, but\n      these do not affect nesting stack.\n\n    Args:\n      line: current line to check.\n    '
        if Match('^\\s*#\\s*(if|ifdef|ifndef)\\b', line):
            self.pp_stack.append(_PreprocessorInfo(copy.deepcopy(self.stack)))
        elif Match('^\\s*#\\s*(else|elif)\\b', line):
            if self.pp_stack:
                if not self.pp_stack[-1].seen_else:
                    self.pp_stack[-1].seen_else = True
                    self.pp_stack[-1].stack_before_else = copy.deepcopy(self.stack)
                self.stack = copy.deepcopy(self.pp_stack[-1].stack_before_if)
            else:
                pass
        elif Match('^\\s*#\\s*endif\\b', line):
            if self.pp_stack:
                if self.pp_stack[-1].seen_else:
                    self.stack = self.pp_stack[-1].stack_before_else
                self.pp_stack.pop()
            else:
                pass

    def Update(self, filename, clean_lines, linenum, error):
        if False:
            return 10
        'Update nesting state with current line.\n\n    Args:\n      filename: The name of the current file.\n      clean_lines: A CleansedLines instance containing the file.\n      linenum: The number of the line to check.\n      error: The function to call with any errors found.\n    '
        line = clean_lines.elided[linenum]
        if self.stack:
            self.previous_stack_top = self.stack[-1]
        else:
            self.previous_stack_top = None
        self.UpdatePreprocessor(line)
        if self.stack:
            inner_block = self.stack[-1]
            depth_change = line.count('(') - line.count(')')
            inner_block.open_parentheses += depth_change
            if inner_block.inline_asm in (_NO_ASM, _END_ASM):
                if depth_change != 0 and inner_block.open_parentheses == 1 and _MATCH_ASM.match(line):
                    inner_block.inline_asm = _INSIDE_ASM
                else:
                    inner_block.inline_asm = _NO_ASM
            elif inner_block.inline_asm == _INSIDE_ASM and inner_block.open_parentheses == 0:
                inner_block.inline_asm = _END_ASM
        while True:
            namespace_decl_match = Match('^\\s*namespace\\b\\s*([:\\w]+)?(.*)$', line)
            if not namespace_decl_match:
                break
            new_namespace = _NamespaceInfo(namespace_decl_match.group(1), linenum)
            self.stack.append(new_namespace)
            line = namespace_decl_match.group(2)
            if line.find('{') != -1:
                new_namespace.seen_open_brace = True
                line = line[line.find('{') + 1:]
        class_decl_match = Match('^(\\s*(?:template\\s*<[\\w\\s<>,:]*>\\s*)?(class|struct)\\s+(?:[A-Z_]+\\s+)*(\\w+(?:::\\w+)*))(.*)$', line)
        if class_decl_match and (not self.stack or self.stack[-1].open_parentheses == 0):
            end_declaration = len(class_decl_match.group(1))
            if not self.InTemplateArgumentList(clean_lines, linenum, end_declaration):
                self.stack.append(_ClassInfo(class_decl_match.group(3), class_decl_match.group(2), clean_lines, linenum))
                line = class_decl_match.group(4)
        if not self.SeenOpenBrace():
            self.stack[-1].CheckBegin(filename, clean_lines, linenum, error)
        if self.stack and isinstance(self.stack[-1], _ClassInfo):
            classinfo = self.stack[-1]
            access_match = Match('^(.*)\\b(public|private|protected|signals)(\\s+(?:slots\\s*)?)?:(?:[^:]|$)', line)
            if access_match:
                classinfo.access = access_match.group(2)
                indent = access_match.group(1)
                if len(indent) != classinfo.class_indent + 1 and Match('^\\s*$', indent):
                    if classinfo.is_struct:
                        parent = 'struct ' + classinfo.name
                    else:
                        parent = 'class ' + classinfo.name
                    slots = ''
                    if access_match.group(3):
                        slots = access_match.group(3)
                    error(filename, linenum, 'whitespace/indent', 3, '%s%s: should be indented +1 space inside %s' % (access_match.group(2), slots, parent))
        while True:
            matched = Match('^[^{;)}]*([{;)}])(.*)$', line)
            if not matched:
                break
            token = matched.group(1)
            if token == '{':
                if not self.SeenOpenBrace():
                    self.stack[-1].seen_open_brace = True
                elif Match('^extern\\s*"[^"]*"\\s*\\{', line):
                    self.stack.append(_ExternCInfo(linenum))
                else:
                    self.stack.append(_BlockInfo(linenum, True))
                    if _MATCH_ASM.match(line):
                        self.stack[-1].inline_asm = _BLOCK_ASM
            elif token == ';' or token == ')':
                if not self.SeenOpenBrace():
                    self.stack.pop()
            elif self.stack:
                self.stack[-1].CheckEnd(filename, clean_lines, linenum, error)
                self.stack.pop()
            line = matched.group(2)

    def InnermostClass(self):
        if False:
            return 10
        'Get class info on the top of the stack.\n\n    Returns:\n      A _ClassInfo object if we are inside a class, or None otherwise.\n    '
        for i in range(len(self.stack), 0, -1):
            classinfo = self.stack[i - 1]
            if isinstance(classinfo, _ClassInfo):
                return classinfo
        return None

    def CheckCompletedBlocks(self, filename, error):
        if False:
            print('Hello World!')
        'Checks that all classes and namespaces have been completely parsed.\n\n    Call this when all lines in a file have been processed.\n    Args:\n      filename: The name of the current file.\n      error: The function to call with any errors found.\n    '
        for obj in self.stack:
            if isinstance(obj, _ClassInfo):
                error(filename, obj.starting_linenum, 'build/class', 5, 'Failed to find complete declaration of class %s' % obj.name)
            elif isinstance(obj, _NamespaceInfo):
                error(filename, obj.starting_linenum, 'build/namespaces', 5, 'Failed to find complete declaration of namespace %s' % obj.name)

def CheckForNonStandardConstructs(filename, clean_lines, linenum, nesting_state, error):
    if False:
        i = 10
        return i + 15
    'Logs an error if we see certain non-ANSI constructs ignored by gcc-2.\n\n  Complain about several constructs which gcc-2 accepts, but which are\n  not standard C++.  Warning about these in lint is one way to ease the\n  transition to new compilers.\n  - put storage class first (e.g. "static const" instead of "const static").\n  - "%lld" instead of %qd" in printf-type functions.\n  - "%1$d" is non-standard in printf-type functions.\n  - "\\%" is an undefined character escape sequence.\n  - text after #endif is not allowed.\n  - invalid inner-style forward declaration.\n  - >? and <? operators, and their >?= and <?= cousins.\n\n  Additionally, check for constructor/destructor style violations and reference\n  members, as it is very convenient to do so while checking for\n  gcc-2 compliance.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: A callable to which errors are reported, which takes 4 arguments:\n           filename, line number, error level, and message\n  '
    line = clean_lines.lines[linenum]
    if Search('printf\\s*\\(.*".*%[-+ ]?\\d*q', line):
        error(filename, linenum, 'runtime/printf_format', 3, '%q in format strings is deprecated.  Use %ll instead.')
    if Search('printf\\s*\\(.*".*%\\d+\\$', line):
        error(filename, linenum, 'runtime/printf_format', 2, '%N$ formats are unconventional.  Try rewriting to avoid them.')
    line = line.replace('\\\\', '')
    if Search('("|\\\').*\\\\(%|\\[|\\(|{)', line):
        error(filename, linenum, 'build/printf_format', 3, '%, [, (, and { are undefined character escapes.  Unescape them.')
    line = clean_lines.elided[linenum]
    if Search('\\b(const|volatile|void|char|short|int|long|float|double|signed|unsigned|schar|u?int8|u?int16|u?int32|u?int64)\\s+(register|static|extern|typedef)\\b', line):
        error(filename, linenum, 'build/storage_class', 5, 'Storage-class specifier (static, extern, typedef, etc) should be at the beginning of the declaration.')
    if Match('\\s*#\\s*endif\\s*[^/\\s]+', line):
        error(filename, linenum, 'build/endif_comment', 5, 'Uncommented text after #endif is non-standard.  Use a comment.')
    if Match('\\s*class\\s+(\\w+\\s*::\\s*)+\\w+\\s*;', line):
        error(filename, linenum, 'build/forward_decl', 5, 'Inner-style forward declarations are invalid.  Remove this line.')
    if Search('(\\w+|[+-]?\\d+(\\.\\d*)?)\\s*(<|>)\\?=?\\s*(\\w+|[+-]?\\d+)(\\.\\d*)?', line):
        error(filename, linenum, 'build/deprecated', 3, '>? and <? (max and min) operators are non-standard and deprecated.')
    if Search('^\\s*const\\s*string\\s*&\\s*\\w+\\s*;', line):
        error(filename, linenum, 'runtime/member_string_references', 2, 'const string& members are dangerous. It is much better to use alternatives, such as pointers or simple constants.')
    classinfo = nesting_state.InnermostClass()
    if not classinfo or not classinfo.seen_open_brace:
        return
    base_classname = classinfo.name.split('::')[-1]
    explicit_constructor_match = Match('\\s+(?:(?:inline|constexpr)\\s+)*(explicit\\s+)?(?:(?:inline|constexpr)\\s+)*%s\\s*\\(((?:[^()]|\\([^()]*\\))*)\\)' % re.escape(base_classname), line)
    if explicit_constructor_match:
        is_marked_explicit = explicit_constructor_match.group(1)
        if not explicit_constructor_match.group(2):
            constructor_args = []
        else:
            constructor_args = explicit_constructor_match.group(2).split(',')
        i = 0
        while i < len(constructor_args):
            constructor_arg = constructor_args[i]
            while constructor_arg.count('<') > constructor_arg.count('>') or constructor_arg.count('(') > constructor_arg.count(')'):
                constructor_arg += ',' + constructor_args[i + 1]
                del constructor_args[i + 1]
            constructor_args[i] = constructor_arg
            i += 1
        defaulted_args = [arg for arg in constructor_args if '=' in arg]
        noarg_constructor = not constructor_args or (len(constructor_args) == 1 and constructor_args[0].strip() == 'void')
        onearg_constructor = len(constructor_args) == 1 and (not noarg_constructor) or (len(constructor_args) >= 1 and (not noarg_constructor) and (len(defaulted_args) >= len(constructor_args) - 1))
        initializer_list_constructor = bool(onearg_constructor and Search('\\bstd\\s*::\\s*initializer_list\\b', constructor_args[0]))
        copy_constructor = bool(onearg_constructor and Match('(const\\s+)?%s(\\s*<[^>]*>)?(\\s+const)?\\s*(?:<\\w+>\\s*)?&' % re.escape(base_classname), constructor_args[0].strip()))
        if not is_marked_explicit and onearg_constructor and (not initializer_list_constructor) and (not copy_constructor):
            if defaulted_args:
                error(filename, linenum, 'runtime/explicit', 5, 'Constructors callable with one argument should be marked explicit.')
            else:
                error(filename, linenum, 'runtime/explicit', 5, 'Single-parameter constructors should be marked explicit.')
        elif is_marked_explicit and (not onearg_constructor):
            if noarg_constructor:
                error(filename, linenum, 'runtime/explicit', 5, 'Zero-parameter constructors should not be marked explicit.')

def CheckSpacingForFunctionCall(filename, clean_lines, linenum, error):
    if False:
        print('Hello World!')
    'Checks for the correctness of various spacing around function calls.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    fncall = line
    for pattern in ('\\bif\\s*\\((.*)\\)\\s*{', '\\bfor\\s*\\((.*)\\)\\s*{', '\\bwhile\\s*\\((.*)\\)\\s*[{;]', '\\bswitch\\s*\\((.*)\\)\\s*{'):
        match = Search(pattern, line)
        if match:
            fncall = match.group(1)
            break
    if not Search('\\b(if|for|while|switch|return|new|delete|catch|sizeof|elif)\\b', fncall) and (not Search(' \\([^)]+\\)\\([^)]*(\\)|,$)', fncall)) and (not Search(' \\([^)]+\\)\\[[^\\]]+\\]', fncall)):
        if Search('\\w\\s*\\(\\s(?!\\s*\\\\$)', fncall):
            error(filename, linenum, 'whitespace/parens', 4, 'Extra space after ( in function call')
        elif Search('\\(\\s+(?!(\\s*\\\\)|\\()', fncall):
            error(filename, linenum, 'whitespace/parens', 2, 'Extra space after (')
        if Search('\\w\\s+\\(', fncall) and (not Search('_{0,2}asm_{0,2}\\s+_{0,2}volatile_{0,2}\\s+\\(', fncall)) and (not Search('#\\s*define|typedef|using\\s+\\w+\\s*=', fncall)) and (not Search('\\w\\s+\\((\\w+::)*\\*\\w+\\)\\(', fncall)) and (not Search('\\bcase\\s+\\(', fncall)):
            if Search('\\boperator_*\\b', line):
                error(filename, linenum, 'whitespace/parens', 0, 'Extra space before ( in function call')
            else:
                error(filename, linenum, 'whitespace/parens', 4, 'Extra space before ( in function call')
        if Search('[^)]\\s+\\)\\s*[^{\\s]', fncall):
            if Search('^\\s+\\)', fncall):
                error(filename, linenum, 'whitespace/parens', 2, 'Closing ) should be moved to the previous line')
            else:
                error(filename, linenum, 'whitespace/parens', 2, 'Extra space before )')

def IsBlankLine(line):
    if False:
        return 10
    'Returns true if the given line is blank.\n\n  We consider a line to be blank if the line is empty or consists of\n  only white spaces.\n\n  Args:\n    line: A line of a string.\n\n  Returns:\n    True, if the given line is blank.\n  '
    return not line or line.isspace()

def CheckForNamespaceIndentation(filename, nesting_state, clean_lines, line, error):
    if False:
        print('Hello World!')
    is_namespace_indent_item = len(nesting_state.stack) > 1 and nesting_state.stack[-1].check_namespace_indentation and isinstance(nesting_state.previous_stack_top, _NamespaceInfo) and (nesting_state.previous_stack_top == nesting_state.stack[-2])
    if ShouldCheckNamespaceIndentation(nesting_state, is_namespace_indent_item, clean_lines.elided, line):
        CheckItemIndentationInNamespace(filename, clean_lines.elided, line, error)

def CheckForFunctionLengths(filename, clean_lines, linenum, function_state, error):
    if False:
        i = 10
        return i + 15
    'Reports for long function bodies.\n\n  For an overview why this is done, see:\n  https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml#Write_Short_Functions\n\n  Uses a simplistic algorithm assuming other style guidelines\n  (especially spacing) are followed.\n  Only checks unindented functions, so class members are unchecked.\n  Trivial bodies are unchecked, so constructors with huge initializer lists\n  may be missed.\n  Blank/comment lines are not counted so as to avoid encouraging the removal\n  of vertical space and comments just to get through a lint check.\n  NOLINT *on the last line of a function* disables this check.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    function_state: Current function name and lines in body so far.\n    error: The function to call with any errors found.\n  '
    lines = clean_lines.lines
    line = lines[linenum]
    joined_line = ''
    starting_func = False
    regexp = '(\\w(\\w|::|\\*|\\&|\\s)*)\\('
    match_result = Match(regexp, line)
    if match_result:
        function_name = match_result.group(1).split()[-1]
        if function_name == 'TEST' or function_name == 'TEST_F' or (not Match('[A-Z_]+$', function_name)):
            starting_func = True
    if starting_func:
        body_found = False
        for start_linenum in range(linenum, clean_lines.NumLines()):
            start_line = lines[start_linenum]
            joined_line += ' ' + start_line.lstrip()
            if Search('(;|})', start_line):
                body_found = True
                break
            elif Search('{', start_line):
                body_found = True
                function = Search('((\\w|:)*)\\(', line).group(1)
                if Match('TEST', function):
                    parameter_regexp = Search('(\\(.*\\))', joined_line)
                    if parameter_regexp:
                        function += parameter_regexp.group(1)
                else:
                    function += '()'
                function_state.Begin(function)
                break
        if not body_found:
            error(filename, linenum, 'readability/fn_size', 5, 'Lint failed to find start of function body.')
    elif Match('^\\}\\s*$', line):
        function_state.Check(error, filename, linenum)
        function_state.End()
    elif not Match('^\\s*$', line):
        function_state.Count()
_RE_PATTERN_TODO = re.compile('^//(\\s*)TODO(\\(.+?\\))?:?(\\s|$)?')

def CheckComment(line, filename, linenum, next_line_start, error):
    if False:
        while True:
            i = 10
    'Checks for common mistakes in comments.\n\n  Args:\n    line: The line in question.\n    filename: The name of the current file.\n    linenum: The number of the line to check.\n    next_line_start: The first non-whitespace column of the next line.\n    error: The function to call with any errors found.\n  '
    commentpos = line.find('//')
    if commentpos != -1:
        if re.sub('\\\\.', '', line[0:commentpos]).count('"') % 2 == 0:
            if not (Match('^.*{ *//', line) and next_line_start == commentpos) and (commentpos >= 1 and line[commentpos - 1] not in string.whitespace or (commentpos >= 2 and line[commentpos - 2] not in string.whitespace)):
                error(filename, linenum, 'whitespace/comments', 2, 'At least two spaces is best between code and comments')
            comment = line[commentpos:]
            match = _RE_PATTERN_TODO.match(comment)
            if match:
                leading_whitespace = match.group(1)
                if len(leading_whitespace) > 1:
                    error(filename, linenum, 'whitespace/todo', 2, 'Too many spaces before TODO')
                username = match.group(2)
                if not username:
                    error(filename, linenum, 'readability/todo', 2, 'Missing username in TODO; it should look like "// TODO(my_username): Stuff."')
                middle_whitespace = match.group(3)
                if middle_whitespace != ' ' and middle_whitespace != '':
                    error(filename, linenum, 'whitespace/todo', 2, 'TODO(my_username) should be followed by a space')
            if Match('//[^ ]*\\w', comment) and (not Match('(///|//\\!)(\\s+|$)', comment)):
                error(filename, linenum, 'whitespace/comments', 4, 'Should have a space between // and comment')

def CheckSpacing(filename, clean_lines, linenum, nesting_state, error):
    if False:
        for i in range(10):
            print('nop')
    "Checks for the correctness of various spacing issues in the code.\n\n  Things we check for: spaces around operators, spaces after\n  if/for/while/switch, no spaces around parens in function calls, two\n  spaces between code and comment, don't start a block with a blank\n  line, don't end a function with a blank line, don't add a blank line\n  after public/protected/private, don't have too many blank lines in a row.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: The function to call with any errors found.\n  "
    raw = clean_lines.lines_without_raw_strings
    line = raw[linenum]
    if IsBlankLine(line) and (not nesting_state.InNamespaceBody()) and (not nesting_state.InExternC()):
        elided = clean_lines.elided
        prev_line = elided[linenum - 1]
        prevbrace = prev_line.rfind('{')
        if prevbrace != -1 and prev_line[prevbrace:].find('}') == -1:
            exception = False
            if Match(' {6}\\w', prev_line):
                search_position = linenum - 2
                while search_position >= 0 and Match(' {6}\\w', elided[search_position]):
                    search_position -= 1
                exception = search_position >= 0 and elided[search_position][:5] == '    :'
            else:
                exception = Match(' {4}\\w[^\\(]*\\)\\s*(const\\s*)?(\\{\\s*$|:)', prev_line) or Match(' {4}:', prev_line)
            if not exception:
                error(filename, linenum, 'whitespace/blank_line', 2, 'Redundant blank line at the start of a code block should be deleted.')
        if linenum + 1 < clean_lines.NumLines():
            next_line = raw[linenum + 1]
            if next_line and Match('\\s*}', next_line) and (next_line.find('} else ') == -1):
                error(filename, linenum, 'whitespace/blank_line', 3, 'Redundant blank line at the end of a code block should be deleted.')
        matched = Match('\\s*(public|protected|private):', prev_line)
        if matched:
            error(filename, linenum, 'whitespace/blank_line', 3, 'Do not leave a blank line after "%s:"' % matched.group(1))
    next_line_start = 0
    if linenum + 1 < clean_lines.NumLines():
        next_line = raw[linenum + 1]
        next_line_start = len(next_line) - len(next_line.lstrip())
    CheckComment(line, filename, linenum, next_line_start, error)
    line = clean_lines.elided[linenum]
    if Search('\\w\\s+\\[', line) and (not Search('(?:delete|return|auto)\\s+\\[', line)):
        error(filename, linenum, 'whitespace/braces', 5, 'Extra space before [')
    if Search('for *\\(.*[^:]:[^: ]', line) or Search('for *\\(.*[^: ]:[^:]', line):
        error(filename, linenum, 'whitespace/forcolon', 2, 'Missing space around colon in range-based for loop')

def CheckOperatorSpacing(filename, clean_lines, linenum, error):
    if False:
        return 10
    'Checks for horizontal spacing around operators.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    while True:
        match = Match('^(.*\\boperator\\b)(\\S+)(\\s*\\(.*)$', line)
        if match:
            line = match.group(1) + '_' * len(match.group(2)) + match.group(3)
        else:
            break
    if (Search('[\\w.]=', line) or Search('=[\\w.]', line)) and (not Search('\\b(if|while|for) ', line)) and (not Search('(>=|<=|==|!=|&=|\\^=|\\|=|\\+=|\\*=|\\/=|\\%=)', line)) and (not Search('operator=', line)):
        error(filename, linenum, 'whitespace/operators', 4, 'Missing spaces around =')
    match = Search('[^<>=!\\s](==|!=|<=|>=|\\|\\|)[^<>=!\\s,;\\)]', line)
    if match:
        error(filename, linenum, 'whitespace/operators', 3, 'Missing spaces around %s' % match.group(1))
    elif not Match('#.*include', line):
        match = Match('^(.*[^\\s<])<[^\\s=<,]', line)
        if match:
            (_, _, end_pos) = CloseExpression(clean_lines, linenum, len(match.group(1)))
            if end_pos <= -1:
                error(filename, linenum, 'whitespace/operators', 3, 'Missing spaces around <')
        match = Match('^(.*[^-\\s>])>[^\\s=>,]', line)
        if match:
            (_, _, start_pos) = ReverseCloseExpression(clean_lines, linenum, len(match.group(1)))
            if start_pos <= -1:
                error(filename, linenum, 'whitespace/operators', 3, 'Missing spaces around >')
    match = Search('(operator|[^\\s(<])(?:L|UL|LL|ULL|l|ul|ll|ull)?<<([^\\s,=<])', line)
    if match and (not (match.group(1).isdigit() and match.group(2).isdigit())) and (not (match.group(1) == 'operator' and match.group(2) == ';')):
        error(filename, linenum, 'whitespace/operators', 3, 'Missing spaces around <<')
    match = Search('>>[a-zA-Z_]', line)
    if match:
        error(filename, linenum, 'whitespace/operators', 3, 'Missing spaces around >>')
    match = Search('(!\\s|~\\s|[\\s]--[\\s;]|[\\s]\\+\\+[\\s;])', line)
    if match:
        error(filename, linenum, 'whitespace/operators', 4, 'Extra space for operator %s' % match.group(1))

def CheckParenthesisSpacing(filename, clean_lines, linenum, error):
    if False:
        return 10
    'Checks for horizontal spacing around parentheses.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    match = Search(' (if\\(|for\\(|while\\(|switch\\()', line)
    if match:
        error(filename, linenum, 'whitespace/parens', 5, 'Missing space before ( in %s' % match.group(1))
    match = Search('\\b(if|for|while|switch)\\s*\\(([ ]*)(.).*[^ ]+([ ]*)\\)\\s*{\\s*$', line)
    if match:
        if len(match.group(2)) != len(match.group(4)):
            if not (match.group(3) == ';' and len(match.group(2)) == 1 + len(match.group(4)) or (not match.group(2) and Search('\\bfor\\s*\\(.*; \\)', line))):
                error(filename, linenum, 'whitespace/parens', 5, 'Mismatching spaces inside () in %s' % match.group(1))
        if len(match.group(2)) not in [0, 1]:
            error(filename, linenum, 'whitespace/parens', 5, 'Should have zero or one spaces inside ( and ) in %s' % match.group(1))

def CheckCommaSpacing(filename, clean_lines, linenum, error):
    if False:
        i = 10
        return i + 15
    'Checks for horizontal spacing near commas and semicolons.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    raw = clean_lines.lines_without_raw_strings
    line = clean_lines.elided[linenum]
    if Search(',[^,\\s]', ReplaceAll('\\boperator\\s*,\\s*\\(', 'F(', line)) and Search(',[^,\\s]', raw[linenum]):
        error(filename, linenum, 'whitespace/comma', 3, 'Missing space after ,')
    if Search(';[^\\s};\\\\)/]', line):
        error(filename, linenum, 'whitespace/semicolon', 3, 'Missing space after ;')

def _IsType(clean_lines, nesting_state, expr):
    if False:
        i = 10
        return i + 15
    'Check if expression looks like a type name, returns true if so.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    expr: The expression to check.\n  Returns:\n    True, if token looks like a type.\n  '
    last_word = Match('^.*(\\b\\S+)$', expr)
    if last_word:
        token = last_word.group(1)
    else:
        token = expr
    if _TYPES.match(token):
        return True
    typename_pattern = '\\b(?:typename|class|struct)\\s+' + re.escape(token) + '\\b'
    block_index = len(nesting_state.stack) - 1
    while block_index >= 0:
        if isinstance(nesting_state.stack[block_index], _NamespaceInfo):
            return False
        last_line = nesting_state.stack[block_index].starting_linenum
        next_block_start = 0
        if block_index > 0:
            next_block_start = nesting_state.stack[block_index - 1].starting_linenum
        first_line = last_line
        while first_line >= next_block_start:
            if clean_lines.elided[first_line].find('template') >= 0:
                break
            first_line -= 1
        if first_line < next_block_start:
            block_index -= 1
            continue
        for i in range(first_line, last_line + 1, 1):
            if Search(typename_pattern, clean_lines.elided[i]):
                return True
        block_index -= 1
    return False

def CheckBracesSpacing(filename, clean_lines, linenum, nesting_state, error):
    if False:
        for i in range(10):
            print('nop')
    'Checks for horizontal spacing near commas.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    match = Match('^(.*[^ ({>]){', line)
    if match:
        leading_text = match.group(1)
        (endline, endlinenum, endpos) = CloseExpression(clean_lines, linenum, len(match.group(1)))
        trailing_text = ''
        if endpos > -1:
            trailing_text = endline[endpos:]
        for offset in range(endlinenum + 1, min(endlinenum + 3, clean_lines.NumLines() - 1)):
            trailing_text += clean_lines.elided[offset]
        if not Match('^[\\s}]*[{.;,)<>\\]:]', trailing_text) and (not _IsType(clean_lines, nesting_state, leading_text)):
            error(filename, linenum, 'whitespace/braces', 5, 'Missing space before {')
    if Search('}else', line):
        error(filename, linenum, 'whitespace/braces', 5, 'Missing space before else')
    if Search(':\\s*;\\s*$', line):
        error(filename, linenum, 'whitespace/semicolon', 5, 'Semicolon defining empty statement. Use {} instead.')
    elif Search('^\\s*;\\s*$', line):
        error(filename, linenum, 'whitespace/semicolon', 5, 'Line contains only semicolon. If this should be an empty statement, use {} instead.')
    elif Search('\\s+;\\s*$', line) and (not Search('\\bfor\\b', line)):
        error(filename, linenum, 'whitespace/semicolon', 5, 'Extra space before last semicolon. If this should be an empty statement, use {} instead.')

def IsDecltype(clean_lines, linenum, column):
    if False:
        for i in range(10):
            print('nop')
    'Check if the token ending on (linenum, column) is decltype().\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: the number of the line to check.\n    column: end column of the token to check.\n  Returns:\n    True if this token is decltype() expression, False otherwise.\n  '
    (text, _, start_col) = ReverseCloseExpression(clean_lines, linenum, column)
    if start_col < 0:
        return False
    if Search('\\bdecltype\\s*$', text[0:start_col]):
        return True
    return False

def CheckSectionSpacing(filename, clean_lines, class_info, linenum, error):
    if False:
        while True:
            i = 10
    'Checks for additional blank line issues related to sections.\n\n  Currently the only thing checked here is blank line before protected/private.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    class_info: A _ClassInfo objects.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    if class_info.last_line - class_info.starting_linenum <= 24 or linenum <= class_info.starting_linenum:
        return
    matched = Match('\\s*(public|protected|private):', clean_lines.lines[linenum])
    if matched:
        prev_line = clean_lines.lines[linenum - 1]
        if not IsBlankLine(prev_line) and (not Search('\\b(class|struct)\\b', prev_line)) and (not Search('\\\\$', prev_line)):
            end_class_head = class_info.starting_linenum
            for i in range(class_info.starting_linenum, linenum):
                if Search('\\{\\s*$', clean_lines.lines[i]):
                    end_class_head = i
                    break
            if end_class_head < linenum - 1:
                error(filename, linenum, 'whitespace/blank_line', 3, '"%s:" should be preceded by a blank line' % matched.group(1))

def GetPreviousNonBlankLine(clean_lines, linenum):
    if False:
        i = 10
        return i + 15
    'Return the most recent non-blank line and its line number.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file contents.\n    linenum: The number of the line to check.\n\n  Returns:\n    A tuple with two elements.  The first element is the contents of the last\n    non-blank line before the current line, or the empty string if this is the\n    first non-blank line.  The second is the line number of that line, or -1\n    if this is the first non-blank line.\n  '
    prevlinenum = linenum - 1
    while prevlinenum >= 0:
        prevline = clean_lines.elided[prevlinenum]
        if not IsBlankLine(prevline):
            return (prevline, prevlinenum)
        prevlinenum -= 1
    return ('', -1)

def CheckBraces(filename, clean_lines, linenum, error):
    if False:
        for i in range(10):
            print('nop')
    'Looks for misplaced braces (e.g. at the end of line).\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    if Match('\\s*{\\s*$', line):
        prevline = GetPreviousNonBlankLine(clean_lines, linenum)[0]
        if not Search('[,;:}{(]\\s*$', prevline) and (not Match('\\s*#', prevline)) and (not (GetLineWidth(prevline) > _line_length - 2 and '[]' in prevline)):
            error(filename, linenum, 'whitespace/braces', 4, '{ should almost always be at the end of the previous line')
    if Match('\\s*else\\b\\s*(?:if\\b|\\{|$)', line):
        prevline = GetPreviousNonBlankLine(clean_lines, linenum)[0]
        if Match('\\s*}\\s*$', prevline):
            error(filename, linenum, 'whitespace/newline', 4, 'An else should appear on the same line as the preceding }')
    if Search('else if\\s*\\(', line):
        brace_on_left = bool(Search('}\\s*else if\\s*\\(', line))
        pos = line.find('else if')
        pos = line.find('(', pos)
        if pos > 0:
            (endline, _, endpos) = CloseExpression(clean_lines, linenum, pos)
            brace_on_right = endline[endpos:].find('{') != -1
            if brace_on_left != brace_on_right:
                error(filename, linenum, 'readability/braces', 5, 'If an else has a brace on one side, it should have it on both')
    elif Search('}\\s*else[^{]*$', line) or Match('[^}]*else\\s*{', line):
        error(filename, linenum, 'readability/braces', 5, 'If an else has a brace on one side, it should have it on both')
    if Search('\\belse [^\\s{]', line) and (not Search('\\belse if\\b', line)):
        error(filename, linenum, 'whitespace/newline', 4, 'Else clause should never be on same line as else (use 2 lines)')
    if Match('\\s*do [^\\s{]', line):
        error(filename, linenum, 'whitespace/newline', 4, 'do/while clauses should not be on a single line')
    if_else_match = Search('\\b(if\\s*\\(|else\\b)', line)
    if if_else_match and (not Match('\\s*#', line)):
        if_indent = GetIndentLevel(line)
        (endline, endlinenum, endpos) = (line, linenum, if_else_match.end())
        if_match = Search('\\bif\\s*\\(', line)
        if if_match:
            pos = if_match.end() - 1
            (endline, endlinenum, endpos) = CloseExpression(clean_lines, linenum, pos)
        if not Match('\\s*{', endline[endpos:]) and (not (Match('\\s*$', endline[endpos:]) and endlinenum < len(clean_lines.elided) - 1 and Match('\\s*{', clean_lines.elided[endlinenum + 1]))):
            while endlinenum < len(clean_lines.elided) and ';' not in clean_lines.elided[endlinenum][endpos:]:
                endlinenum += 1
                endpos = 0
            if endlinenum < len(clean_lines.elided):
                endline = clean_lines.elided[endlinenum]
                endpos = endline.find(';')
                if not Match(';[\\s}]*(\\\\?)$', endline[endpos:]):
                    if not Match('^[^{};]*\\[[^\\[\\]]*\\][^{}]*\\{[^{}]*\\}\\s*\\)*[;,]\\s*$', endline):
                        error(filename, linenum, 'readability/braces', 4, 'If/else bodies with multiple statements require braces')
                elif endlinenum < len(clean_lines.elided) - 1:
                    next_line = clean_lines.elided[endlinenum + 1]
                    next_indent = GetIndentLevel(next_line)
                    if if_match and Match('\\s*else\\b', next_line) and (next_indent != if_indent):
                        error(filename, linenum, 'readability/braces', 4, 'Else clause should be indented at the same level as if. Ambiguous nested if/else chains require braces.')
                    elif next_indent > if_indent:
                        error(filename, linenum, 'readability/braces', 4, 'If/else bodies with multiple statements require braces')

def CheckTrailingSemicolon(filename, clean_lines, linenum, error):
    if False:
        while True:
            i = 10
    'Looks for redundant trailing semicolon.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    match = Match('^(.*\\)\\s*)\\{', line)
    if match:
        closing_brace_pos = match.group(1).rfind(')')
        opening_parenthesis = ReverseCloseExpression(clean_lines, linenum, closing_brace_pos)
        if opening_parenthesis[2] > -1:
            line_prefix = opening_parenthesis[0][0:opening_parenthesis[2]]
            macro = Search('\\b([A-Z_][A-Z0-9_]*)\\s*$', line_prefix)
            func = Match('^(.*\\])\\s*$', line_prefix)
            if macro and macro.group(1) not in ('TEST', 'TEST_F', 'MATCHER', 'MATCHER_P', 'TYPED_TEST', 'EXCLUSIVE_LOCKS_REQUIRED', 'SHARED_LOCKS_REQUIRED', 'LOCKS_EXCLUDED', 'INTERFACE_DEF') or (func and (not Search('\\boperator\\s*\\[\\s*\\]', func.group(1)))) or Search('\\b(?:struct|union)\\s+alignas\\s*$', line_prefix) or Search('\\bdecltype$', line_prefix) or Search('\\s+=\\s*$', line_prefix):
                match = None
        if match and opening_parenthesis[1] > 1 and Search('\\]\\s*$', clean_lines.elided[opening_parenthesis[1] - 1]):
            match = None
    else:
        match = Match('^(.*(?:else|\\)\\s*const)\\s*)\\{', line)
        if not match:
            prevline = GetPreviousNonBlankLine(clean_lines, linenum)[0]
            if prevline and Search('[;{}]\\s*$', prevline):
                match = Match('^(\\s*)\\{', line)
    if match:
        (endline, endlinenum, endpos) = CloseExpression(clean_lines, linenum, len(match.group(1)))
        if endpos > -1 and Match('^\\s*;', endline[endpos:]):
            raw_lines = clean_lines.raw_lines
            ParseNolintSuppressions(filename, raw_lines[endlinenum - 1], endlinenum - 1, error)
            ParseNolintSuppressions(filename, raw_lines[endlinenum], endlinenum, error)
            error(filename, endlinenum, 'readability/braces', 4, "You don't need a ; after a }")

def CheckEmptyBlockBody(filename, clean_lines, linenum, error):
    if False:
        i = 10
        return i + 15
    'Look for empty loop/conditional body with only a single semicolon.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    matched = Match('\\s*(for|while|if)\\s*\\(', line)
    if matched:
        (end_line, end_linenum, end_pos) = CloseExpression(clean_lines, linenum, line.find('('))
        if end_pos >= 0 and Match(';', end_line[end_pos:]):
            if matched.group(1) == 'if':
                error(filename, end_linenum, 'whitespace/empty_conditional_body', 5, 'Empty conditional bodies should use {}')
            else:
                error(filename, end_linenum, 'whitespace/empty_loop_body', 5, 'Empty loop bodies should use {} or continue')
        if end_pos >= 0 and matched.group(1) == 'if':
            opening_linenum = end_linenum
            opening_line_fragment = end_line[end_pos:]
            while not Search('^\\s*\\{', opening_line_fragment):
                if Search('^(?!\\s*$)', opening_line_fragment):
                    return
                opening_linenum += 1
                if opening_linenum == len(clean_lines.elided):
                    return
                opening_line_fragment = clean_lines.elided[opening_linenum]
            opening_line = clean_lines.elided[opening_linenum]
            opening_pos = opening_line_fragment.find('{')
            if opening_linenum == end_linenum:
                opening_pos += end_pos
            (closing_line, closing_linenum, closing_pos) = CloseExpression(clean_lines, opening_linenum, opening_pos)
            if closing_pos < 0:
                return
            if clean_lines.raw_lines[opening_linenum] != CleanseComments(clean_lines.raw_lines[opening_linenum]):
                return
            if closing_linenum > opening_linenum:
                body = list(opening_line[opening_pos + 1:])
                body.extend(clean_lines.raw_lines[opening_linenum + 1:closing_linenum])
                body.append(clean_lines.elided[closing_linenum][:closing_pos - 1])
                body = '\n'.join(body)
            else:
                body = opening_line[opening_pos + 1:closing_pos - 1]
            if not _EMPTY_CONDITIONAL_BODY_PATTERN.search(body):
                return
            current_linenum = closing_linenum
            current_line_fragment = closing_line[closing_pos:]
            while Search('^\\s*$|^(?=\\s*else)', current_line_fragment):
                if Search('^(?=\\s*else)', current_line_fragment):
                    return
                current_linenum += 1
                if current_linenum == len(clean_lines.elided):
                    break
                current_line_fragment = clean_lines.elided[current_linenum]
            error(filename, end_linenum, 'whitespace/empty_if_body', 4, 'If statement had no body and no else clause')

def FindCheckMacro(line):
    if False:
        print('Hello World!')
    'Find a replaceable CHECK-like macro.\n\n  Args:\n    line: line to search on.\n  Returns:\n    (macro name, start position), or (None, -1) if no replaceable\n    macro is found.\n  '
    for macro in _CHECK_MACROS:
        i = line.find(macro)
        if i >= 0:
            matched = Match('^(.*\\b' + macro + '\\s*)\\(', line)
            if not matched:
                continue
            return (macro, len(matched.group(1)))
    return (None, -1)

def CheckCheck(filename, clean_lines, linenum, error):
    if False:
        i = 10
        return i + 15
    'Checks the use of CHECK and EXPECT macros.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    lines = clean_lines.elided
    (check_macro, start_pos) = FindCheckMacro(lines[linenum])
    if not check_macro:
        return
    (last_line, end_line, end_pos) = CloseExpression(clean_lines, linenum, start_pos)
    if end_pos < 0:
        return
    if not Match('\\s*;', last_line[end_pos:]):
        return
    if linenum == end_line:
        expression = lines[linenum][start_pos + 1:end_pos - 1]
    else:
        expression = lines[linenum][start_pos + 1:]
        for i in range(linenum + 1, end_line):
            expression += lines[i]
        expression += last_line[0:end_pos - 1]
    lhs = ''
    rhs = ''
    operator = None
    while expression:
        matched = Match('^\\s*(<<|<<=|>>|>>=|->\\*|->|&&|\\|\\||==|!=|>=|>|<=|<|\\()(.*)$', expression)
        if matched:
            token = matched.group(1)
            if token == '(':
                expression = matched.group(2)
                (end, _) = FindEndOfExpressionInLine(expression, 0, ['('])
                if end < 0:
                    return
                lhs += '(' + expression[0:end]
                expression = expression[end:]
            elif token in ('&&', '||'):
                return
            elif token in ('<<', '<<=', '>>', '>>=', '->*', '->'):
                lhs += token
                expression = matched.group(2)
            else:
                operator = token
                rhs = matched.group(2)
                break
        else:
            matched = Match('^([^-=!<>()&|]+)(.*)$', expression)
            if not matched:
                matched = Match('^(\\s*\\S)(.*)$', expression)
                if not matched:
                    break
            lhs += matched.group(1)
            expression = matched.group(2)
    if not (lhs and operator and rhs):
        return
    if rhs.find('&&') > -1 or rhs.find('||') > -1:
        return
    lhs = lhs.strip()
    rhs = rhs.strip()
    match_constant = '^([-+]?(\\d+|0[xX][0-9a-fA-F]+)[lLuU]{0,3}|".*"|\\\'.*\\\')$'
    if Match(match_constant, lhs) or Match(match_constant, rhs):
        error(filename, linenum, 'readability/check', 2, 'Consider using %s instead of %s(a %s b)' % (_CHECK_REPLACEMENT[check_macro][operator], check_macro, operator))

def CheckAltTokens(filename, clean_lines, linenum, error):
    if False:
        for i in range(10):
            print('nop')
    'Check alternative keywords being used in boolean expressions.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    if Match('^\\s*#', line):
        return
    if line.find('/*') >= 0 or line.find('*/') >= 0:
        return
    for match in _ALT_TOKEN_REPLACEMENT_PATTERN.finditer(line):
        error(filename, linenum, 'readability/alt_tokens', 2, 'Use operator %s instead of %s' % (_ALT_TOKEN_REPLACEMENT[match.group(1)], match.group(1)))

def GetLineWidth(line):
    if False:
        print('Hello World!')
    'Determines the width of the line in column positions.\n\n  Args:\n    line: A string, which may be a Unicode string.\n\n  Returns:\n    The width of the line in column positions, accounting for Unicode\n    combining characters and wide characters.\n  '
    if python2_version and isinstance(line, unicode):
        width = 0
        for uc in unicodedata.normalize('NFC', line):
            if unicodedata.east_asian_width(uc) in ('W', 'F'):
                width += 2
            elif not unicodedata.combining(uc):
                width += 1
        return width
    else:
        return len(line)

def CheckStyle(filename, clean_lines, linenum, file_extension, nesting_state, error):
    if False:
        print('Hello World!')
    "Checks rules from the 'C++ style rules' section of cppguide.html.\n\n  Most of these rules are hard to test (naming, comment style), but we\n  do what we can.  In particular we check for 2-space indents, line lengths,\n  tab usage, spaces inside code, etc.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    file_extension: The extension (without the dot) of the filename.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: The function to call with any errors found.\n  "
    raw_lines = clean_lines.lines_without_raw_strings
    line = raw_lines[linenum]
    prev = raw_lines[linenum - 1] if linenum > 0 else ''
    if line.find('\t') != -1:
        error(filename, linenum, 'whitespace/tab', 1, 'Tab found; better to use spaces')
    scope_or_label_pattern = '\\s*\\w+\\s*:\\s*\\\\?$'
    classinfo = nesting_state.InnermostClass()
    initial_spaces = 0
    cleansed_line = clean_lines.elided[linenum]
    while initial_spaces < len(line) and line[initial_spaces] == ' ':
        initial_spaces += 1
    if not Search('[",=><] *$', prev) and (initial_spaces == 1 or initial_spaces == 3) and (not Match(scope_or_label_pattern, cleansed_line)) and (not (clean_lines.raw_lines[linenum] != line and Match('^\\s*""', line))):
        error(filename, linenum, 'whitespace/indent', 3, 'Weird number of spaces at line-start.  Are you using a 2-space indent?')
    if line and line[-1].isspace():
        error(filename, linenum, 'whitespace/end_of_line', 4, 'Line ends in whitespace.  Consider deleting these extra spaces.')
    is_header_guard = False
    if IsHeaderExtension(file_extension):
        cppvar = GetHeaderGuardCPPVariable(filename)
        if line.startswith('#ifndef %s' % cppvar) or line.startswith('#define %s' % cppvar) or line.startswith('#endif  // %s' % cppvar):
            is_header_guard = True
    if not line.startswith('#include') and (not is_header_guard) and (not Match('^\\s*//.*http(s?)://\\S*$', line)) and (not Match('^\\s*//\\s*[^\\s]*$', line)) and (not Match('^// \\$Id:.*#[0-9]+ \\$$', line)):
        line_width = GetLineWidth(line)
        if line_width > _line_length:
            error(filename, linenum, 'whitespace/line_length', 2, 'Lines should be <= %i characters long' % _line_length)
    if cleansed_line.count(';') > 1 and cleansed_line.find('for') == -1 and (GetPreviousNonBlankLine(clean_lines, linenum)[0].find('for') == -1 or GetPreviousNonBlankLine(clean_lines, linenum)[0].find(';') != -1) and (not ((cleansed_line.find('case ') != -1 or cleansed_line.find('default:') != -1) and cleansed_line.find('break;') != -1)):
        error(filename, linenum, 'whitespace/newline', 0, 'More than one command on the same line')
    CheckBraces(filename, clean_lines, linenum, error)
    CheckTrailingSemicolon(filename, clean_lines, linenum, error)
    CheckEmptyBlockBody(filename, clean_lines, linenum, error)
    CheckSpacing(filename, clean_lines, linenum, nesting_state, error)
    CheckOperatorSpacing(filename, clean_lines, linenum, error)
    CheckParenthesisSpacing(filename, clean_lines, linenum, error)
    CheckCommaSpacing(filename, clean_lines, linenum, error)
    CheckBracesSpacing(filename, clean_lines, linenum, nesting_state, error)
    CheckSpacingForFunctionCall(filename, clean_lines, linenum, error)
    CheckCheck(filename, clean_lines, linenum, error)
    CheckAltTokens(filename, clean_lines, linenum, error)
    classinfo = nesting_state.InnermostClass()
    if classinfo:
        CheckSectionSpacing(filename, clean_lines, classinfo, linenum, error)
_RE_PATTERN_INCLUDE = re.compile('^\\s*#\\s*include\\s*([<"])([^>"]*)[>"].*$')
_RE_FIRST_COMPONENT = re.compile('^[^-_.]+')

def _DropCommonSuffixes(filename):
    if False:
        while True:
            i = 10
    "Drops common suffixes like _test.cc or -inl.h from filename.\n\n  For example:\n    >>> _DropCommonSuffixes('foo/foo-inl.h')\n    'foo/foo'\n    >>> _DropCommonSuffixes('foo/bar/foo.cc')\n    'foo/bar/foo'\n    >>> _DropCommonSuffixes('foo/foo_internal.h')\n    'foo/foo'\n    >>> _DropCommonSuffixes('foo/foo_unusualinternal.h')\n    'foo/foo_unusualinternal'\n\n  Args:\n    filename: The input filename.\n\n  Returns:\n    The filename with the common suffix removed.\n  "
    for suffix in ('test.cc', 'regtest.cc', 'unittest.cc', 'inl.h', 'impl.h', 'internal.h'):
        if filename.endswith(suffix) and len(filename) > len(suffix) and (filename[-len(suffix) - 1] in ('-', '_')):
            return filename[:-len(suffix) - 1]
    return os.path.splitext(filename)[0]

def _ClassifyInclude(fileinfo, include, is_system):
    if False:
        print('Hello World!')
    'Figures out what kind of header \'include\' is.\n\n  Args:\n    fileinfo: The current file cpplint is running over. A FileInfo instance.\n    include: The path to a #included file.\n    is_system: True if the #include used <> rather than "".\n\n  Returns:\n    One of the _XXX_HEADER constants.\n\n  For example:\n    >>> _ClassifyInclude(FileInfo(\'foo/foo.cc\'), \'stdio.h\', True)\n    _C_SYS_HEADER\n    >>> _ClassifyInclude(FileInfo(\'foo/foo.cc\'), \'string\', True)\n    _CPP_SYS_HEADER\n    >>> _ClassifyInclude(FileInfo(\'foo/foo.cc\'), \'foo/foo.h\', False)\n    _LIKELY_MY_HEADER\n    >>> _ClassifyInclude(FileInfo(\'foo/foo_unknown_extension.cc\'),\n    ...                  \'bar/foo_other_ext.h\', False)\n    _POSSIBLE_MY_HEADER\n    >>> _ClassifyInclude(FileInfo(\'foo/foo.cc\'), \'foo/bar.h\', False)\n    _OTHER_HEADER\n  '

    def basename(name):
        if False:
            while True:
                i = 10
        name = os.path.split(name)[1]
        name = name.split('/')[-1]
        return name
    is_cpp_h = include in _CPP_HEADERS or include.endswith('.hpp') or include.endswith('.hxx') or include.endswith('.H') or include.endswith('.hh') or ('.' not in basename(include))
    if is_system:
        if is_cpp_h:
            return _CPP_SYS_HEADER
        else:
            return _C_SYS_HEADER
    (target_dir, target_base) = os.path.split(_DropCommonSuffixes(fileinfo.RepositoryName()))
    (include_dir, include_base) = os.path.split(_DropCommonSuffixes(include))
    if target_base == include_base and (include_dir == target_dir or include_dir == os.path.normpath(target_dir + '/../public')):
        return _LIKELY_MY_HEADER
    target_first_component = _RE_FIRST_COMPONENT.match(target_base)
    include_first_component = _RE_FIRST_COMPONENT.match(include_base)
    if target_first_component and include_first_component and (target_first_component.group(0) == include_first_component.group(0)):
        return _POSSIBLE_MY_HEADER
    return _OTHER_HEADER

def CheckIncludeLine(filename, clean_lines, linenum, include_state, error):
    if False:
        return 10
    'Check rules that are applicable to #include lines.\n\n  Strings on #include lines are NOT removed from elided line, to make\n  certain tasks easier. However, to prevent false positives, checks\n  applicable to #include lines in CheckLanguage must be put here.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    include_state: An _IncludeState instance in which the headers are inserted.\n    error: The function to call with any errors found.\n  '
    fileinfo = FileInfo(filename)
    line = clean_lines.lines[linenum]
    match = Match('#include\\s*"([^/]+\\.h)"', line)
    if match and (not _THIRD_PARTY_HEADERS_PATTERN.match(match.group(1))):
        error(filename, linenum, 'build/include', 4, 'Include the directory when naming .h files')
    match = _RE_PATTERN_INCLUDE.search(line)
    if match:
        include = match.group(2)
        is_system = match.group(1) == '<'
        duplicate_line = include_state.FindHeader(include)
        if duplicate_line >= 0:
            error(filename, linenum, 'build/include', 4, '"%s" already included at %s:%s' % (include, filename, duplicate_line))
        elif include.endswith('.cc') and os.path.dirname(fileinfo.RepositoryName()) != os.path.dirname(include):
            error(filename, linenum, 'build/include', 4, 'Do not include .cc files from other packages')
        elif not _THIRD_PARTY_HEADERS_PATTERN.match(include):
            include_state.include_list[-1].append((include, linenum))
            error_message = include_state.CheckNextIncludeOrder(_ClassifyInclude(fileinfo, include, is_system))
            if error_message:
                error(filename, linenum, 'build/include_order', 4, '%s. Should be: %s.h, c system, c++ system, other.' % (error_message, fileinfo.BaseName()))
            canonical_include = include_state.CanonicalizeAlphabeticalOrder(include)
            if not include_state.IsInAlphabeticalOrder(clean_lines, linenum, canonical_include):
                error(filename, linenum, 'build/include_alpha', 4, 'Include "%s" not in alphabetical order' % include)
            include_state.SetLastHeader(canonical_include)

def _GetTextInside(text, start_pattern):
    if False:
        for i in range(10):
            print('nop')
    "Retrieves all the text between matching open and close parentheses.\n\n  Given a string of lines and a regular expression string, retrieve all the text\n  following the expression and between opening punctuation symbols like\n  (, [, or {, and the matching close-punctuation symbol. This properly nested\n  occurrences of the punctuations, so for the text like\n    printf(a(), b(c()));\n  a call to _GetTextInside(text, r'printf\\(') will return 'a(), b(c())'.\n  start_pattern must match string having an open punctuation symbol at the end.\n\n  Args:\n    text: The lines to extract text. Its comments and strings must be elided.\n           It can be single line and can span multiple lines.\n    start_pattern: The regexp string indicating where to start extracting\n                   the text.\n  Returns:\n    The extracted text.\n    None if either the opening string or ending punctuation could not be found.\n  "
    matching_punctuation = {'(': ')', '{': '}', '[': ']'}
    closing_punctuation = set([value for (_, value) in matching_punctuation.items()])
    match = re.search(start_pattern, text, re.M)
    if not match:
        return None
    start_position = match.end(0)
    assert start_position > 0, 'start_pattern must ends with an opening punctuation.'
    assert text[start_position - 1] in matching_punctuation, 'start_pattern must ends with an opening punctuation.'
    punctuation_stack = [matching_punctuation[text[start_position - 1]]]
    position = start_position
    while punctuation_stack and position < len(text):
        if text[position] == punctuation_stack[-1]:
            punctuation_stack.pop()
        elif text[position] in closing_punctuation:
            return None
        elif text[position] in matching_punctuation:
            punctuation_stack.append(matching_punctuation[text[position]])
        position += 1
    if punctuation_stack:
        return None
    return text[start_position:position - 1]
_RE_PATTERN_IDENT = '[_a-zA-Z]\\w*'
_RE_PATTERN_TYPE = '(?:const\\s+)?(?:typename\\s+|class\\s+|struct\\s+|union\\s+|enum\\s+)?(?:\\w|\\s*<(?:<(?:<[^<>]*>|[^<>])*>|[^<>])*>|::)+'
_RE_PATTERN_REF_PARAM = re.compile('(' + _RE_PATTERN_TYPE + '(?:\\s*(?:\\bconst\\b|[*]))*\\s*&\\s*' + _RE_PATTERN_IDENT + ')\\s*(?:=[^,()]+)?[,)]')
_RE_PATTERN_CONST_REF_PARAM = '(?:.*\\s*\\bconst\\s*&\\s*' + _RE_PATTERN_IDENT + '|const\\s+' + _RE_PATTERN_TYPE + '\\s*&\\s*' + _RE_PATTERN_IDENT + ')'
_RE_PATTERN_REF_STREAM_PARAM = '(?:.*stream\\s*&\\s*' + _RE_PATTERN_IDENT + ')'

def CheckLanguage(filename, clean_lines, linenum, file_extension, include_state, nesting_state, error):
    if False:
        for i in range(10):
            print('nop')
    "Checks rules from the 'C++ language rules' section of cppguide.html.\n\n  Some of these rules are hard to test (function overloading, using\n  uint32 inappropriately), but we do the best we can.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    file_extension: The extension (without the dot) of the filename.\n    include_state: An _IncludeState instance in which the headers are inserted.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: The function to call with any errors found.\n  "
    line = clean_lines.elided[linenum]
    if not line:
        return
    match = _RE_PATTERN_INCLUDE.search(line)
    if match:
        CheckIncludeLine(filename, clean_lines, linenum, include_state, error)
        return
    match = Match('^\\s*#\\s*(if|ifdef|ifndef|elif|else|endif)\\b', line)
    if match:
        include_state.ResetSection(match.group(1))
    fullname = os.path.abspath(filename).replace('\\', '/')
    CheckCasts(filename, clean_lines, linenum, error)
    CheckGlobalStatic(filename, clean_lines, linenum, error)
    CheckPrintf(filename, clean_lines, linenum, error)
    if IsHeaderExtension(file_extension):
        pass
    if Search('\\bshort port\\b', line):
        if not Search('\\bunsigned short port\\b', line):
            error(filename, linenum, 'runtime/int', 4, 'Use "unsigned short" for ports, not "short"')
    else:
        match = Search('\\b(short|long(?! +double)|long long)\\b', line)
        if match:
            error(filename, linenum, 'runtime/int', 4, 'Use int16/int64/etc, rather than the C type %s' % match.group(1))
    if Search('\\boperator\\s*&\\s*\\(\\s*\\)', line):
        error(filename, linenum, 'runtime/operator', 4, 'Unary operator& is dangerous.  Do not use it.')
    if Search('\\}\\s*if\\s*\\(', line):
        error(filename, linenum, 'readability/braces', 4, 'Did you mean "else if"? If not, start a new line for "if".')
    printf_args = _GetTextInside(line, '(?i)\\b(string)?printf\\s*\\(')
    if printf_args:
        match = Match('([\\w.\\->()]+)$', printf_args)
        if match and match.group(1) != '__VA_ARGS__':
            function_name = re.search('\\b((?:string)?printf)\\s*\\(', line, re.I).group(1)
            error(filename, linenum, 'runtime/printf', 4, 'Potential format string bug. Do %s("%%s", %s) instead.' % (function_name, match.group(1)))
    match = Search('memset\\s*\\(([^,]*),\\s*([^,]*),\\s*0\\s*\\)', line)
    if match and (not Match("^''|-?[0-9]+|0x[0-9A-Fa-f]$", match.group(2))):
        error(filename, linenum, 'runtime/memset', 4, 'Did you mean "memset(%s, 0, %s)"?' % (match.group(1), match.group(2)))
    if Search('\\busing namespace\\b', line):
        error(filename, linenum, 'build/namespaces', 5, 'Do not use namespace using-directives.  Use using-declarations instead.')
    match = Match('\\s*(.+::)?(\\w+) [a-z]\\w*\\[(.+)];', line)
    if match and match.group(2) != 'return' and (match.group(2) != 'delete') and (match.group(3).find(']') == -1):
        tokens = re.split('\\s|\\+|\\-|\\*|\\/|<<|>>]', match.group(3))
        is_const = True
        skip_next = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if Search('sizeof\\(.+\\)', tok):
                continue
            if Search('arraysize\\(\\w+\\)', tok):
                continue
            tok = tok.lstrip('(')
            tok = tok.rstrip(')')
            if not tok:
                continue
            if Match('\\d+', tok):
                continue
            if Match('0[xX][0-9a-fA-F]+', tok):
                continue
            if Match('k[A-Z0-9]\\w*', tok):
                continue
            if Match('(.+::)?k[A-Z0-9]\\w*', tok):
                continue
            if Match('(.+::)?[A-Z][A-Z0-9_]*', tok):
                continue
            if tok.startswith('sizeof'):
                skip_next = True
                continue
            is_const = False
            break
        if not is_const:
            error(filename, linenum, 'runtime/arrays', 1, "Do not use variable-length arrays.  Use an appropriately named ('k' followed by CamelCase) compile-time constant for the size.")
    if IsHeaderExtension(file_extension) and Search('\\bnamespace\\s*{', line) and (line[-1] != '\\'):
        error(filename, linenum, 'build/namespaces', 4, 'Do not use unnamed namespaces in header files.  See https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml#Namespaces for more information.')

def CheckGlobalStatic(filename, clean_lines, linenum, error):
    if False:
        print('Hello World!')
    'Check for unsafe global or static objects.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    if linenum + 1 < clean_lines.NumLines() and (not Search('[;({]', line)):
        line += clean_lines.elided[linenum + 1].strip()
    match = Match('((?:|static +)(?:|const +))(?::*std::)?string( +const)? +([a-zA-Z0-9_:]+)\\b(.*)', line)
    if match and (not Search('\\bstring\\b(\\s+const)?\\s*[\\*\\&]\\s*(const\\s+)?\\w', line)) and (not Search('\\boperator\\W', line)) and (not Match('\\s*(<.*>)?(::[a-zA-Z0-9_]+)*\\s*\\(([^"]|$)', match.group(4))):
        if Search('\\bconst\\b', line):
            error(filename, linenum, 'runtime/string', 4, 'For a static/global string constant, use a C style string instead: "%schar%s %s[]".' % (match.group(1), match.group(2) or '', match.group(3)))
        else:
            error(filename, linenum, 'runtime/string', 4, 'Static/global string variables are not permitted.')
    if Search('\\b([A-Za-z0-9_]*_)\\(\\1\\)', line) or Search('\\b([A-Za-z0-9_]*_)\\(CHECK_NOTNULL\\(\\1\\)\\)', line):
        error(filename, linenum, 'runtime/init', 4, 'You seem to be initializing a member variable with itself.')

def CheckPrintf(filename, clean_lines, linenum, error):
    if False:
        for i in range(10):
            print('nop')
    'Check for printf related issues.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    match = Search('snprintf\\s*\\(([^,]*),\\s*([0-9]*)\\s*,', line)
    if match and match.group(2) != '0':
        error(filename, linenum, 'runtime/printf', 3, 'If you can, use sizeof(%s) instead of %s as the 2nd arg to snprintf.' % (match.group(1), match.group(2)))
    if Search('\\bsprintf\\s*\\(', line):
        error(filename, linenum, 'runtime/printf', 5, 'Never use sprintf. Use snprintf instead.')
    match = Search('\\b(strcpy|strcat)\\s*\\(', line)
    if match:
        error(filename, linenum, 'runtime/printf', 4, 'Almost always, snprintf is better than %s' % match.group(1))

def IsDerivedFunction(clean_lines, linenum):
    if False:
        while True:
            i = 10
    'Check if current line contains an inherited function.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n  Returns:\n    True if current line contains a function with "override"\n    virt-specifier.\n  '
    for i in range(linenum, max(-1, linenum - 10), -1):
        match = Match('^([^()]*\\w+)\\(', clean_lines.elided[i])
        if match:
            (line, _, closing_paren) = CloseExpression(clean_lines, i, len(match.group(1)))
            return closing_paren >= 0 and Search('\\boverride\\b', line[closing_paren:])
    return False

def IsOutOfLineMethodDefinition(clean_lines, linenum):
    if False:
        while True:
            i = 10
    'Check if current line contains an out-of-line method definition.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n  Returns:\n    True if current line contains an out-of-line method definition.\n  '
    for i in range(linenum, max(-1, linenum - 10), -1):
        if Match('^([^()]*\\w+)\\(', clean_lines.elided[i]):
            return Match('^[^()]*\\w+::\\w+\\(', clean_lines.elided[i]) is not None
    return False

def IsInitializerList(clean_lines, linenum):
    if False:
        i = 10
        return i + 15
    'Check if current line is inside constructor initializer list.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n  Returns:\n    True if current line appears to be inside constructor initializer\n    list, False otherwise.\n  '
    for i in range(linenum, 1, -1):
        line = clean_lines.elided[i]
        if i == linenum:
            remove_function_body = Match('^(.*)\\{\\s*$', line)
            if remove_function_body:
                line = remove_function_body.group(1)
        if Search('\\s:\\s*\\w+[({]', line):
            return True
        if Search('\\}\\s*,\\s*$', line):
            return True
        if Search('[{};]\\s*$', line):
            return False
    return False

def CheckForNonConstReference(filename, clean_lines, linenum, nesting_state, error):
    if False:
        print('Hello World!')
    'Check for non-const references.\n\n  Separate from CheckLanguage since it scans backwards from current\n  line, instead of scanning forward.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    if '&' not in line:
        return
    if IsDerivedFunction(clean_lines, linenum):
        return
    if IsOutOfLineMethodDefinition(clean_lines, linenum):
        return
    if linenum > 1:
        previous = None
        if Match('\\s*::(?:[\\w<>]|::)+\\s*&\\s*\\S', line):
            previous = Search('\\b((?:const\\s*)?(?:[\\w<>]|::)+[\\w<>])\\s*$', clean_lines.elided[linenum - 1])
        elif Match('\\s*[a-zA-Z_]([\\w<>]|::)+\\s*&\\s*\\S', line):
            previous = Search('\\b((?:const\\s*)?(?:[\\w<>]|::)+::)\\s*$', clean_lines.elided[linenum - 1])
        if previous:
            line = previous.group(1) + line.lstrip()
        else:
            endpos = line.rfind('>')
            if endpos > -1:
                (_, startline, startpos) = ReverseCloseExpression(clean_lines, linenum, endpos)
                if startpos > -1 and startline < linenum:
                    line = ''
                    for i in range(startline, linenum + 1):
                        line += clean_lines.elided[i].strip()
    if nesting_state.previous_stack_top and (not (isinstance(nesting_state.previous_stack_top, _ClassInfo) or isinstance(nesting_state.previous_stack_top, _NamespaceInfo))):
        return
    if linenum > 0:
        for i in range(linenum - 1, max(0, linenum - 10), -1):
            previous_line = clean_lines.elided[i]
            if not Search('[),]\\s*$', previous_line):
                break
            if Match('^\\s*:\\s+\\S', previous_line):
                return
    if Search('\\\\\\s*$', line):
        return
    if IsInitializerList(clean_lines, linenum):
        return
    whitelisted_functions = '(?:[sS]wap(?:<\\w:+>)?|operator\\s*[<>][<>]|static_assert|COMPILE_ASSERT)\\s*\\('
    if Search(whitelisted_functions, line):
        return
    elif not Search('\\S+\\([^)]*$', line):
        for i in range(2):
            if linenum > i and Search(whitelisted_functions, clean_lines.elided[linenum - i - 1]):
                return
    decls = ReplaceAll('{[^}]*}', ' ', line)
    for parameter in re.findall(_RE_PATTERN_REF_PARAM, decls):
        if not Match(_RE_PATTERN_CONST_REF_PARAM, parameter) and (not Match(_RE_PATTERN_REF_STREAM_PARAM, parameter)):
            error(filename, linenum, 'runtime/references', 2, 'Is this a non-const reference? If so, make const or use a pointer: ' + ReplaceAll(' *<', '<', parameter))

def CheckCasts(filename, clean_lines, linenum, error):
    if False:
        print('Hello World!')
    'Various cast related checks.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    match = Search('(\\bnew\\s+(?:const\\s+)?|\\S<\\s*(?:const\\s+)?)?\\b(int|float|double|bool|char|int32|uint32|int64|uint64)(\\([^)].*)', line)
    expecting_function = ExpectingFunctionArgs(clean_lines, linenum)
    if match and (not expecting_function):
        matched_type = match.group(2)
        matched_new_or_template = match.group(1)
        if Match('\\([^()]+\\)\\s*\\[', match.group(3)):
            return
        matched_funcptr = match.group(3)
        if matched_new_or_template is None and (not (matched_funcptr and (Match('\\((?:[^() ]+::\\s*\\*\\s*)?[^() ]+\\)\\s*\\(', matched_funcptr) or matched_funcptr.startswith('(*)')))) and (not Match('\\s*using\\s+\\S+\\s*=\\s*' + matched_type, line)) and (not Search('new\\(\\S+\\)\\s*' + matched_type, line)):
            error(filename, linenum, 'readability/casting', 4, 'Using deprecated casting style.  Use static_cast<%s>(...) instead' % matched_type)
    if not expecting_function:
        CheckCStyleCast(filename, clean_lines, linenum, 'static_cast', '\\((int|float|double|bool|char|u?int(16|32|64))\\)', error)
    if CheckCStyleCast(filename, clean_lines, linenum, 'const_cast', '\\((char\\s?\\*+\\s?)\\)\\s*"', error):
        pass
    else:
        CheckCStyleCast(filename, clean_lines, linenum, 'reinterpret_cast', '\\((\\w+\\s?\\*+\\s?)\\)', error)
    match = Search('(?:[^\\w]&\\(([^)*][^)]*)\\)[\\w(])|(?:[^\\w]&(static|dynamic|down|reinterpret)_cast\\b)', line)
    if match:
        parenthesis_error = False
        match = Match('^(.*&(?:static|dynamic|down|reinterpret)_cast\\b)<', line)
        if match:
            (_, y1, x1) = CloseExpression(clean_lines, linenum, len(match.group(1)))
            if x1 >= 0 and clean_lines.elided[y1][x1] == '(':
                (_, y2, x2) = CloseExpression(clean_lines, y1, x1)
                if x2 >= 0:
                    extended_line = clean_lines.elided[y2][x2:]
                    if y2 < clean_lines.NumLines() - 1:
                        extended_line += clean_lines.elided[y2 + 1]
                    if Match('\\s*(?:->|\\[)', extended_line):
                        parenthesis_error = True
        if parenthesis_error:
            error(filename, linenum, 'readability/casting', 4, 'Are you taking an address of something dereferenced from a cast?  Wrapping the dereferenced expression in parentheses will make the binding more obvious')
        else:
            error(filename, linenum, 'runtime/casting', 4, 'Are you taking an address of a cast?  This is dangerous: could be a temp var.  Take the address before doing the cast, rather than after')

def CheckCStyleCast(filename, clean_lines, linenum, cast_type, pattern, error):
    if False:
        while True:
            i = 10
    'Checks for a C-style cast by looking for the pattern.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    cast_type: The string for the C++ cast to recommend.  This is either\n      reinterpret_cast, static_cast, or const_cast, depending.\n    pattern: The regular expression used to find C-style casts.\n    error: The function to call with any errors found.\n\n  Returns:\n    True if an error was emitted.\n    False otherwise.\n  '
    line = clean_lines.elided[linenum]
    match = Search(pattern, line)
    if not match:
        return False
    context = line[0:match.start(1) - 1]
    if Match('.*\\b(?:sizeof|alignof|alignas|[_A-Z][_A-Z0-9]*)\\s*$', context):
        return False
    if linenum > 0:
        for i in range(linenum - 1, max(0, linenum - 5), -1):
            context = clean_lines.elided[i] + context
    if Match('.*\\b[_A-Z][_A-Z0-9]*\\s*\\((?:\\([^()]*\\)|[^()])*$', context):
        return False
    if context.endswith(' operator++') or context.endswith(' operator--'):
        return False
    remainder = line[match.end(0):]
    if Match('^\\s*(?:;|const\\b|throw\\b|final\\b|override\\b|[=>{),]|->)', remainder):
        return False
    error(filename, linenum, 'readability/casting', 4, 'Using C-style cast.  Use %s<%s>(...) instead' % (cast_type, match.group(1)))
    return True

def ExpectingFunctionArgs(clean_lines, linenum):
    if False:
        return 10
    "Checks whether where function type arguments are expected.\n\n  Args:\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n\n  Returns:\n    True if the line at 'linenum' is inside something that expects arguments\n    of function types.\n  "
    line = clean_lines.elided[linenum]
    return Match('^\\s*MOCK_(CONST_)?METHOD\\d+(_T)?\\(', line) or (linenum >= 2 and (Match('^\\s*MOCK_(?:CONST_)?METHOD\\d+(?:_T)?\\((?:\\S+,)?\\s*$', clean_lines.elided[linenum - 1]) or Match('^\\s*MOCK_(?:CONST_)?METHOD\\d+(?:_T)?\\(\\s*$', clean_lines.elided[linenum - 2]) or Search('\\bstd::m?function\\s*\\<\\s*$', clean_lines.elided[linenum - 1])))
_HEADERS_CONTAINING_TEMPLATES = (('<atomic>', ('atomic',)), ('<deque>', ('deque',)), ('<functional>', ('unary_function', 'binary_function', 'plus', 'minus', 'multiplies', 'divides', 'modulus', 'negate', 'equal_to', 'not_equal_to', 'greater', 'less', 'greater_equal', 'less_equal', 'logical_and', 'logical_or', 'logical_not', 'unary_negate', 'not1', 'binary_negate', 'not2', 'bind1st', 'bind2nd', 'pointer_to_unary_function', 'pointer_to_binary_function', 'ptr_fun', 'mem_fun_t', 'mem_fun', 'mem_fun1_t', 'mem_fun1_ref_t', 'mem_fun_ref_t', 'const_mem_fun_t', 'const_mem_fun1_t', 'const_mem_fun_ref_t', 'const_mem_fun1_ref_t', 'mem_fun_ref')), ('<limits>', ('numeric_limits',)), ('<list>', ('list',)), ('<map>', ('map', 'multimap')), ('<memory>', ('allocator', 'make_shared', 'make_unique', 'shared_ptr', 'unique_ptr', 'weak_ptr')), ('<queue>', ('queue', 'priority_queue')), ('<set>', ('set', 'multiset')), ('<stack>', ('stack',)), ('<string>', ('char_traits', 'basic_string')), ('<tuple>', ('tuple',)), ('<unordered_map>', ('unordered_map', 'unordered_multimap')), ('<unordered_set>', ('unordered_set', 'unordered_multiset')), ('<utility>', ('pair',)), ('<vector>', ('vector',)), ('<hash_map>', ('hash_map', 'hash_multimap')), ('<hash_set>', ('hash_set', 'hash_multiset')), ('<slist>', ('slist',)))
_HEADERS_MAYBE_TEMPLATES = (('<algorithm>', ('min_element',)), ('<utility>', ('forward', 'make_pair', 'move')))
_RE_PATTERN_STRING = re.compile('\\bstring\\b')
_re_pattern_headers_maybe_templates = []
for (_header, _templates) in _HEADERS_MAYBE_TEMPLATES:
    for _template in _templates:
        _re_pattern_headers_maybe_templates.append((re.compile('[^>.]\\b' + _template + '(<.*?>)?\\([^\\)]'), _template, _header))
_re_pattern_templates = []
for (_header, _templates) in _HEADERS_CONTAINING_TEMPLATES:
    for _template in _templates:
        _re_pattern_templates.append((re.compile('(\\<|\\b)' + _template + '\\s*\\<'), _template + '<>', _header))

def FilesBelongToSameModule(filename_cc, filename_h):
    if False:
        for i in range(10):
            print('nop')
    "Check if these two filenames belong to the same module.\n\n  The concept of a 'module' here is a as follows:\n  foo.h, foo-inl.h, foo.cc, foo_test.cc and foo_unittest.cc belong to the\n  same 'module' if they are in the same directory.\n  some/path/public/xyzzy and some/path/internal/xyzzy are also considered\n  to belong to the same module here.\n\n  If the filename_cc contains a longer path than the filename_h, for example,\n  '/absolute/path/to/base/sysinfo.cc', and this file would include\n  'base/sysinfo.h', this function also produces the prefix needed to open the\n  header. This is used by the caller of this function to more robustly open the\n  header file. We don't have access to the real include paths in this context,\n  so we need this guesswork here.\n\n  Known bugs: tools/base/bar.cc and base/bar.h belong to the same module\n  according to this implementation. Because of this, this function gives\n  some false positives. This should be sufficiently rare in practice.\n\n  Args:\n    filename_cc: is the path for the .cc file\n    filename_h: is the path for the header path\n\n  Returns:\n    Tuple with a bool and a string:\n    bool: True if filename_cc and filename_h belong to the same module.\n    string: the additional prefix needed to open the header file.\n  "
    fileinfo = FileInfo(filename_cc)
    if not fileinfo.IsSource():
        return (False, '')
    filename_cc = filename_cc[:-len(fileinfo.Extension())]
    matched_test_suffix = Search(_TEST_FILE_SUFFIX, fileinfo.BaseName())
    if matched_test_suffix:
        filename_cc = filename_cc[:-len(matched_test_suffix.group(1))]
    filename_cc = filename_cc.replace('/public/', '/')
    filename_cc = filename_cc.replace('/internal/', '/')
    if not filename_h.endswith('.h'):
        return (False, '')
    filename_h = filename_h[:-len('.h')]
    if filename_h.endswith('-inl'):
        filename_h = filename_h[:-len('-inl')]
    filename_h = filename_h.replace('/public/', '/')
    filename_h = filename_h.replace('/internal/', '/')
    files_belong_to_same_module = filename_cc.endswith(filename_h)
    common_path = ''
    if files_belong_to_same_module:
        common_path = filename_cc[:-len(filename_h)]
    return (files_belong_to_same_module, common_path)

def UpdateIncludeState(filename, include_dict, io=codecs):
    if False:
        return 10
    'Fill up the include_dict with new includes found from the file.\n\n  Args:\n    filename: the name of the header to read.\n    include_dict: a dictionary in which the headers are inserted.\n    io: The io factory to use to read the file. Provided for testability.\n\n  Returns:\n    True if a header was successfully added. False otherwise.\n  '
    headerfile = None
    try:
        headerfile = io.open(filename, 'r', 'utf8', 'replace')
    except IOError:
        return False
    linenum = 0
    for line in headerfile:
        linenum += 1
        clean_line = CleanseComments(line)
        match = _RE_PATTERN_INCLUDE.search(clean_line)
        if match:
            include = match.group(2)
            include_dict.setdefault(include, linenum)
    return True

def CheckForIncludeWhatYouUse(filename, clean_lines, include_state, error, io=codecs):
    if False:
        print('Hello World!')
    'Reports for missing stl includes.\n\n  This function will output warnings to make sure you are including the headers\n  necessary for the stl containers and functions that you use. We only give one\n  reason to include a header. For example, if you use both equal_to<> and\n  less<> in a .h file, only one (the latter in the file) of these will be\n  reported as a reason to include the <functional>.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    include_state: An _IncludeState instance.\n    error: The function to call with any errors found.\n    io: The IO factory to use to read the header file. Provided for unittest\n        injection.\n  '
    required = {}
    for linenum in range(clean_lines.NumLines()):
        line = clean_lines.elided[linenum]
        if not line or line[0] == '#':
            continue
        matched = _RE_PATTERN_STRING.search(line)
        if matched:
            prefix = line[:matched.start()]
            if prefix.endswith('std::') or not prefix.endswith('::'):
                required['<string>'] = (linenum, 'string')
        for (pattern, template, header) in _re_pattern_headers_maybe_templates:
            if pattern.search(line):
                required[header] = (linenum, template)
        if not '<' in line:
            continue
        for (pattern, template, header) in _re_pattern_templates:
            matched = pattern.search(line)
            if matched:
                prefix = line[:matched.start()]
                if prefix.endswith('std::') or not prefix.endswith('::'):
                    required[header] = (linenum, template)
    include_dict = dict([item for sublist in include_state.include_list for item in sublist])
    header_found = False
    abs_filename = FileInfo(filename).FullName()
    abs_filename = re.sub('_flymake\\.cc$', '.cc', abs_filename)
    header_keys = list(include_dict.keys())
    for header in header_keys:
        (same_module, common_path) = FilesBelongToSameModule(abs_filename, header)
        fullpath = common_path + header
        if same_module and UpdateIncludeState(fullpath, include_dict, io):
            header_found = True
    if filename.endswith('.cc') and (not header_found):
        return
    for required_header_unstripped in required:
        template = required[required_header_unstripped][1]
        if required_header_unstripped.strip('<>"') not in include_dict:
            error(filename, required[required_header_unstripped][0], 'build/include_what_you_use', 4, 'Add #include ' + required_header_unstripped + ' for ' + template)
_RE_PATTERN_EXPLICIT_MAKEPAIR = re.compile('\\bmake_pair\\s*<')

def CheckMakePairUsesDeduction(filename, clean_lines, linenum, error):
    if False:
        print('Hello World!')
    "Check that make_pair's template arguments are deduced.\n\n  G++ 4.6 in C++11 mode fails badly if make_pair's template arguments are\n  specified explicitly, and such use isn't intended in any case.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  "
    line = clean_lines.elided[linenum]
    match = _RE_PATTERN_EXPLICIT_MAKEPAIR.search(line)
    if match:
        error(filename, linenum, 'build/explicit_make_pair', 4, 'For C++11-compatibility, omit template arguments from make_pair OR use pair directly OR if appropriate, construct a pair directly')

def CheckRedundantVirtual(filename, clean_lines, linenum, error):
    if False:
        return 10
    'Check if line contains a redundant "virtual" function-specifier.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    virtual = Match('^(.*)(\\bvirtual\\b)(.*)$', line)
    if not virtual:
        return
    if Search('\\b(public|protected|private)\\s+$', virtual.group(1)) or Match('^\\s+(public|protected|private)\\b', virtual.group(3)):
        return
    if Match('^.*[^:]:[^:].*$', line):
        return
    end_col = -1
    end_line = -1
    start_col = len(virtual.group(2))
    for start_line in range(linenum, min(linenum + 3, clean_lines.NumLines())):
        line = clean_lines.elided[start_line][start_col:]
        parameter_list = Match('^([^(]*)\\(', line)
        if parameter_list:
            (_, end_line, end_col) = CloseExpression(clean_lines, start_line, start_col + len(parameter_list.group(1)))
            break
        start_col = 0
    if end_col < 0:
        return
    for i in range(end_line, min(end_line + 3, clean_lines.NumLines())):
        line = clean_lines.elided[i][end_col:]
        match = Search('\\b(override|final)\\b', line)
        if match:
            error(filename, linenum, 'readability/inheritance', 4, '"virtual" is redundant since function is already declared as "%s"' % match.group(1))
        end_col = 0
        if Search('[^\\w]\\s*$', line):
            break

def CheckRedundantOverrideOrFinal(filename, clean_lines, linenum, error):
    if False:
        return 10
    'Check if line contains a redundant "override" or "final" virt-specifier.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    declarator_end = line.rfind(')')
    if declarator_end >= 0:
        fragment = line[declarator_end:]
    elif linenum > 1 and clean_lines.elided[linenum - 1].rfind(')') >= 0:
        fragment = line
    else:
        return
    if Search('\\boverride\\b', fragment) and Search('\\bfinal\\b', fragment):
        error(filename, linenum, 'readability/inheritance', 4, '"override" is redundant since function is already declared as "final"')

def IsBlockInNameSpace(nesting_state, is_forward_declaration):
    if False:
        print('Hello World!')
    'Checks that the new block is directly in a namespace.\n\n  Args:\n    nesting_state: The _NestingState object that contains info about our state.\n    is_forward_declaration: If the class is a forward declared class.\n  Returns:\n    Whether or not the new block is directly in a namespace.\n  '
    if is_forward_declaration:
        if len(nesting_state.stack) >= 1 and isinstance(nesting_state.stack[-1], _NamespaceInfo):
            return True
        else:
            return False
    return len(nesting_state.stack) > 1 and nesting_state.stack[-1].check_namespace_indentation and isinstance(nesting_state.stack[-2], _NamespaceInfo)

def ShouldCheckNamespaceIndentation(nesting_state, is_namespace_indent_item, raw_lines_no_comments, linenum):
    if False:
        print('Hello World!')
    'This method determines if we should apply our namespace indentation check.\n\n  Args:\n    nesting_state: The current nesting state.\n    is_namespace_indent_item: If we just put a new class on the stack, True.\n      If the top of the stack is not a class, or we did not recently\n      add the class, False.\n    raw_lines_no_comments: The lines without the comments.\n    linenum: The current line number we are processing.\n\n  Returns:\n    True if we should apply our namespace indentation check. Currently, it\n    only works for classes and namespaces inside of a namespace.\n  '
    is_forward_declaration = IsForwardClassDeclaration(raw_lines_no_comments, linenum)
    if not (is_namespace_indent_item or is_forward_declaration):
        return False
    if IsMacroDefinition(raw_lines_no_comments, linenum):
        return False
    return IsBlockInNameSpace(nesting_state, is_forward_declaration)

def CheckItemIndentationInNamespace(filename, raw_lines_no_comments, linenum, error):
    if False:
        print('Hello World!')
    line = raw_lines_no_comments[linenum]
    if Match('^\\s+', line):
        error(filename, linenum, 'runtime/indentation_namespace', 4, 'Do not indent within a namespace')

def ProcessLine(filename, file_extension, clean_lines, line, include_state, function_state, nesting_state, error, extra_check_functions=[]):
    if False:
        while True:
            i = 10
    'Processes a single line in the file.\n\n  Args:\n    filename: Filename of the file that is being processed.\n    file_extension: The extension (dot not included) of the file.\n    clean_lines: An array of strings, each representing a line of the file,\n                 with comments stripped.\n    line: Number of line being processed.\n    include_state: An _IncludeState instance in which the headers are inserted.\n    function_state: A _FunctionState instance which counts function lines, etc.\n    nesting_state: A NestingState instance which maintains information about\n                   the current stack of nested blocks being parsed.\n    error: A callable to which errors are reported, which takes 4 arguments:\n           filename, line number, error level, and message\n    extra_check_functions: An array of additional check functions that will be\n                           run on each source line. Each function takes 4\n                           arguments: filename, clean_lines, line, error\n  '
    raw_lines = clean_lines.raw_lines
    ParseNolintSuppressions(filename, raw_lines[line], line, error)
    nesting_state.Update(filename, clean_lines, line, error)
    CheckForNamespaceIndentation(filename, nesting_state, clean_lines, line, error)
    if nesting_state.InAsmBlock():
        return
    CheckForFunctionLengths(filename, clean_lines, line, function_state, error)
    CheckForMultilineCommentsAndStrings(filename, clean_lines, line, error)
    CheckStyle(filename, clean_lines, line, file_extension, nesting_state, error)
    CheckLanguage(filename, clean_lines, line, file_extension, include_state, nesting_state, error)
    CheckForNonStandardConstructs(filename, clean_lines, line, nesting_state, error)
    CheckVlogArguments(filename, clean_lines, line, error)
    CheckPosixThreading(filename, clean_lines, line, error)
    CheckInvalidIncrement(filename, clean_lines, line, error)
    CheckMakePairUsesDeduction(filename, clean_lines, line, error)
    CheckRedundantVirtual(filename, clean_lines, line, error)
    CheckRedundantOverrideOrFinal(filename, clean_lines, line, error)
    for check_fn in extra_check_functions:
        check_fn(filename, clean_lines, line, error)

def FlagCxx11Features(filename, clean_lines, linenum, error):
    if False:
        while True:
            i = 10
    'Flag those c++11 features that we only allow in certain places.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    include = Match('\\s*#\\s*include\\s+[<"]([^<"]+)[">]', line)
    if include and include.group(1).startswith('tr1/'):
        error(filename, linenum, 'build/c++tr1', 5, 'C++ TR1 headers such as <%s> are unapproved.' % include.group(1))
    if Match('\\s*#', line) and (not Match('\\s*#\\s*define\\b', line)):
        return
    for top_name in ('alignment_of', 'aligned_union'):
        if Search('\\bstd::%s\\b' % top_name, line):
            error(filename, linenum, 'build/c++11', 5, 'std::%s is an unapproved C++11 class or function.  Send c-style an example of where it would make your code more readable, and they may let you use it.' % top_name)

def FlagCxx14Features(filename, clean_lines, linenum, error):
    if False:
        while True:
            i = 10
    'Flag those C++14 features that we restrict.\n\n  Args:\n    filename: The name of the current file.\n    clean_lines: A CleansedLines instance containing the file.\n    linenum: The number of the line to check.\n    error: The function to call with any errors found.\n  '
    line = clean_lines.elided[linenum]
    include = Match('\\s*#\\s*include\\s+[<"]([^<"]+)[">]', line)
    if include and include.group(1) in ('scoped_allocator', 'shared_mutex'):
        error(filename, linenum, 'build/c++14', 5, '<%s> is an unapproved C++14 header.' % include.group(1))

def ProcessFileData(filename, file_extension, lines, error, extra_check_functions=[]):
    if False:
        while True:
            i = 10
    'Performs lint checks and reports any errors to the given error function.\n\n  Args:\n    filename: Filename of the file that is being processed.\n    file_extension: The extension (dot not included) of the file.\n    lines: An array of strings, each representing a line of the file, with the\n           last element being empty if the file is terminated with a newline.\n    error: A callable to which errors are reported, which takes 4 arguments:\n           filename, line number, error level, and message\n    extra_check_functions: An array of additional check functions that will be\n                           run on each source line. Each function takes 4\n                           arguments: filename, clean_lines, line, error\n  '
    lines = ['// marker so line numbers and indices both start at 1'] + lines + ['// marker so line numbers end in a known way']
    include_state = _IncludeState()
    function_state = _FunctionState()
    nesting_state = NestingState()
    ResetNolintSuppressions()
    CheckForCopyright(filename, lines, error)
    ProcessGlobalSuppresions(lines)
    RemoveMultiLineComments(filename, lines, error)
    clean_lines = CleansedLines(lines)
    if IsHeaderExtension(file_extension):
        CheckForHeaderGuard(filename, clean_lines, error)
    for line in range(clean_lines.NumLines()):
        ProcessLine(filename, file_extension, clean_lines, line, include_state, function_state, nesting_state, error, extra_check_functions)
        FlagCxx11Features(filename, clean_lines, line, error)
    nesting_state.CheckCompletedBlocks(filename, error)
    CheckForIncludeWhatYouUse(filename, clean_lines, include_state, error)
    if _IsSourceExtension(file_extension):
        CheckHeaderFileIncluded(filename, include_state, error)
    CheckForBadCharacters(filename, lines, error)
    CheckForNewlineAtEOF(filename, lines, error)

def ProcessConfigOverrides(filename):
    if False:
        for i in range(10):
            print('nop')
    ' Loads the configuration files and processes the config overrides.\n\n  Args:\n    filename: The name of the file being processed by the linter.\n\n  Returns:\n    False if the current |filename| should not be processed further.\n  '
    abs_filename = os.path.abspath(filename)
    cfg_filters = []
    keep_looking = True
    while keep_looking:
        (abs_path, base_name) = os.path.split(abs_filename)
        if not base_name:
            break
        cfg_file = os.path.join(abs_path, 'CPPLINT.cfg')
        abs_filename = abs_path
        if not os.path.isfile(cfg_file):
            continue
        try:
            with open(cfg_file) as file_handle:
                for line in file_handle:
                    (line, _, _) = line.partition('#')
                    if not line.strip():
                        continue
                    (name, _, val) = line.partition('=')
                    name = name.strip()
                    val = val.strip()
                    if name == 'set noparent':
                        keep_looking = False
                    elif name == 'filter':
                        cfg_filters.append(val)
                    elif name == 'exclude_files':
                        if base_name:
                            pattern = re.compile(val)
                            if pattern.match(base_name):
                                if _cpplint_state.quiet:
                                    return False
                                sys.stderr.write('Ignoring "%s": file excluded by "%s". File path component "%s" matches pattern "%s"\n' % (filename, cfg_file, base_name, val))
                                return False
                    elif name == 'linelength':
                        global _line_length
                        try:
                            _line_length = int(val)
                        except ValueError:
                            sys.stderr.write('Line length must be numeric.')
                    elif name == 'root':
                        global _root
                        _root = os.path.join(os.path.dirname(cfg_file), val)
                    elif name == 'headers':
                        ProcessHppHeadersOption(val)
                    else:
                        sys.stderr.write('Invalid configuration option (%s) in file %s\n' % (name, cfg_file))
        except IOError:
            sys.stderr.write("Skipping config file '%s': Can't open for reading\n" % cfg_file)
            keep_looking = False
    for filter in reversed(cfg_filters):
        _AddFilters(filter)
    return True

def ProcessFile(filename, vlevel, extra_check_functions=[]):
    if False:
        return 10
    'Does google-lint on a single file.\n\n  Args:\n    filename: The name of the file to parse.\n\n    vlevel: The level of errors to report.  Every error of confidence\n    >= verbose_level will be reported.  0 is a good default.\n\n    extra_check_functions: An array of additional check functions that will be\n                           run on each source line. Each function takes 4\n                           arguments: filename, clean_lines, line, error\n  '
    _SetVerboseLevel(vlevel)
    _BackupFilters()
    old_errors = _cpplint_state.error_count
    if not ProcessConfigOverrides(filename):
        _RestoreFilters()
        return
    lf_lines = []
    crlf_lines = []
    try:
        if filename == '-':
            lines = codecs.StreamReaderWriter(sys.stdin, codecs.getreader('utf8'), codecs.getwriter('utf8'), 'replace').read().split('\n')
        else:
            lines = codecs.open(filename, 'r', 'utf8', 'replace').read().split('\n')
        for linenum in range(len(lines) - 1):
            if lines[linenum].endswith('\r'):
                lines[linenum] = lines[linenum].rstrip('\r')
                crlf_lines.append(linenum + 1)
            else:
                lf_lines.append(linenum + 1)
    except IOError:
        sys.stderr.write("Skipping input '%s': Can't open for reading\n" % filename)
        _RestoreFilters()
        return
    file_extension = filename[filename.rfind('.') + 1:]
    if filename != '-' and file_extension not in _valid_extensions:
        sys.stderr.write('Ignoring %s; not a valid file name (%s)\n' % (filename, ', '.join(_valid_extensions)))
    else:
        ProcessFileData(filename, file_extension, lines, Error, extra_check_functions)
        if lf_lines and crlf_lines:
            for linenum in crlf_lines:
                Error(filename, linenum, 'whitespace/newline', 1, 'Unexpected \\r (^M) found; better to use only \\n')
    if not _cpplint_state.quiet or old_errors != _cpplint_state.error_count:
        sys.stdout.write('Done processing %s\n' % filename)
    _RestoreFilters()

def PrintUsage(message):
    if False:
        return 10
    'Prints a brief usage string and exits, optionally with an error message.\n\n  Args:\n    message: The optional error message.\n  '
    sys.stderr.write(_USAGE)
    if message:
        sys.exit('\nFATAL ERROR: ' + message)
    else:
        sys.exit(1)

def PrintCategories():
    if False:
        i = 10
        return i + 15
    'Prints a list of all the error-categories used by error messages.\n\n  These are the categories used to filter messages via --filter.\n  '
    sys.stderr.write(''.join(('  %s\n' % cat for cat in _ERROR_CATEGORIES)))
    sys.exit(0)

def ParseArguments(args):
    if False:
        while True:
            i = 10
    'Parses the command line arguments.\n\n  This may set the output format and verbosity level as side-effects.\n\n  Args:\n    args: The command line arguments:\n\n  Returns:\n    The list of filenames to lint.\n  '
    try:
        (opts, filenames) = getopt.getopt(args, '', ['help', 'output=', 'verbose=', 'counting=', 'filter=', 'root=', 'linelength=', 'extensions=', 'headers=', 'quiet'])
    except getopt.GetoptError:
        PrintUsage('Invalid arguments.')
    verbosity = _VerboseLevel()
    output_format = _OutputFormat()
    filters = ''
    quiet = _Quiet()
    counting_style = ''
    for (opt, val) in opts:
        if opt == '--help':
            PrintUsage(None)
        elif opt == '--output':
            if val not in ('emacs', 'vs7', 'eclipse'):
                PrintUsage('The only allowed output formats are emacs, vs7 and eclipse.')
            output_format = val
        elif opt == '--quiet':
            quiet = True
        elif opt == '--verbose':
            verbosity = int(val)
        elif opt == '--filter':
            filters = val
            if not filters:
                PrintCategories()
        elif opt == '--counting':
            if val not in ('total', 'toplevel', 'detailed'):
                PrintUsage('Valid counting options are total, toplevel, and detailed')
            counting_style = val
        elif opt == '--root':
            global _root
            _root = val
        elif opt == '--linelength':
            global _line_length
            try:
                _line_length = int(val)
            except ValueError:
                PrintUsage('Line length must be digits.')
        elif opt == '--extensions':
            global _valid_extensions
            try:
                _valid_extensions = set(val.split(','))
            except ValueError:
                PrintUsage('Extensions must be comma seperated list.')
        elif opt == '--headers':
            ProcessHppHeadersOption(val)
    if not filenames:
        PrintUsage('No files were specified.')
    _SetOutputFormat(output_format)
    _SetQuiet(quiet)
    _SetVerboseLevel(verbosity)
    _SetFilters(filters)
    _SetCountingStyle(counting_style)
    return filenames

def main():
    if False:
        return 10
    filenames = ParseArguments(sys.argv[1:])
    if python2_version:
        sys.stderr = codecs.StreamReaderWriter(sys.stderr, codecs.getreader('utf8'), codecs.getwriter('utf8'), 'replace')
    _cpplint_state.ResetErrorCounts()
    for filename in filenames:
        ProcessFile(filename, _cpplint_state.verbose_level)
    if not _cpplint_state.quiet or _cpplint_state.error_count > 0:
        _cpplint_state.PrintErrorCounts()
    sys.exit(_cpplint_state.error_count > 0)
if __name__ == '__main__':
    main()