__doc__ = '\npyparsing module - Classes and methods to define and execute parsing grammars\n\nThe pyparsing module is an alternative approach to creating and executing simple grammars,\nvs. the traditional lex/yacc approach, or the use of regular expressions.  With pyparsing, you\ndon\'t need to learn a new syntax for defining grammars or matching expressions - the parsing module\nprovides a library of classes that you use to construct the grammar directly in Python.\n\nHere is a program to parse "Hello, World!" (or any greeting of the form \nC{"<salutation>, <addressee>!"}), built up using L{Word}, L{Literal}, and L{And} elements \n(L{\'+\'<ParserElement.__add__>} operator gives L{And} expressions, strings are auto-converted to\nL{Literal} expressions)::\n\n    from pyparsing import Word, alphas\n\n    # define grammar of a greeting\n    greet = Word(alphas) + "," + Word(alphas) + "!"\n\n    hello = "Hello, World!"\n    print (hello, "->", greet.parseString(hello))\n\nThe program outputs the following::\n\n    Hello, World! -> [\'Hello\', \',\', \'World\', \'!\']\n\nThe Python representation of the grammar is quite readable, owing to the self-explanatory\nclass names, and the use of \'+\', \'|\' and \'^\' operators.\n\nThe L{ParseResults} object returned from L{ParserElement.parseString<ParserElement.parseString>} can be accessed as a nested list, a dictionary, or an\nobject with named attributes.\n\nThe pyparsing module handles some of the problems that are typically vexing when writing text parsers:\n - extra or missing whitespace (the above program will also handle "Hello,World!", "Hello  ,  World  !", etc.)\n - quoted strings\n - embedded comments\n'
__version__ = '2.2.0'
__versionTime__ = '06 Mar 2017 02:06 UTC'
__author__ = 'Paul McGuire <ptmcg@users.sourceforge.net>'
import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
try:
    from _thread import RLock
except ImportError:
    from threading import RLock
try:
    from collections import OrderedDict as _OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict as _OrderedDict
    except ImportError:
        _OrderedDict = None
__all__ = ['And', 'CaselessKeyword', 'CaselessLiteral', 'CharsNotIn', 'Combine', 'Dict', 'Each', 'Empty', 'FollowedBy', 'Forward', 'GoToColumn', 'Group', 'Keyword', 'LineEnd', 'LineStart', 'Literal', 'MatchFirst', 'NoMatch', 'NotAny', 'OneOrMore', 'OnlyOnce', 'Optional', 'Or', 'ParseBaseException', 'ParseElementEnhance', 'ParseException', 'ParseExpression', 'ParseFatalException', 'ParseResults', 'ParseSyntaxException', 'ParserElement', 'QuotedString', 'RecursiveGrammarException', 'Regex', 'SkipTo', 'StringEnd', 'StringStart', 'Suppress', 'Token', 'TokenConverter', 'White', 'Word', 'WordEnd', 'WordStart', 'ZeroOrMore', 'alphanums', 'alphas', 'alphas8bit', 'anyCloseTag', 'anyOpenTag', 'cStyleComment', 'col', 'commaSeparatedList', 'commonHTMLEntity', 'countedArray', 'cppStyleComment', 'dblQuotedString', 'dblSlashComment', 'delimitedList', 'dictOf', 'downcaseTokens', 'empty', 'hexnums', 'htmlComment', 'javaStyleComment', 'line', 'lineEnd', 'lineStart', 'lineno', 'makeHTMLTags', 'makeXMLTags', 'matchOnlyAtCol', 'matchPreviousExpr', 'matchPreviousLiteral', 'nestedExpr', 'nullDebugAction', 'nums', 'oneOf', 'opAssoc', 'operatorPrecedence', 'printables', 'punc8bit', 'pythonStyleComment', 'quotedString', 'removeQuotes', 'replaceHTMLEntity', 'replaceWith', 'restOfLine', 'sglQuotedString', 'srange', 'stringEnd', 'stringStart', 'traceParseAction', 'unicodeString', 'upcaseTokens', 'withAttribute', 'indentedBlock', 'originalTextFor', 'ungroup', 'infixNotation', 'locatedExpr', 'withClass', 'CloseMatch', 'tokenMap', 'pyparsing_common']
system_version = tuple(sys.version_info)[:3]
PY_3 = system_version[0] == 3
if PY_3:
    _MAX_INT = sys.maxsize
    basestring = str
    unichr = chr
    _ustr = str
    singleArgBuiltins = [sum, len, sorted, reversed, list, tuple, set, any, all, min, max]
else:
    _MAX_INT = sys.maxint
    range = xrange

    def _ustr(obj):
        if False:
            for i in range(10):
                print('nop')
        'Drop-in replacement for str(obj) that tries to be Unicode friendly. It first tries\n           str(obj). If that fails with a UnicodeEncodeError, then it tries unicode(obj). It\n           then < returns the unicode object | encodes it with the default encoding | ... >.\n        '
        if isinstance(obj, unicode):
            return obj
        try:
            return str(obj)
        except UnicodeEncodeError:
            ret = unicode(obj).encode(sys.getdefaultencoding(), 'xmlcharrefreplace')
            xmlcharref = Regex('&#\\d+;')
            xmlcharref.setParseAction(lambda t: '\\u' + hex(int(t[0][2:-1]))[2:])
            return xmlcharref.transformString(ret)
    singleArgBuiltins = []
    import __builtin__
    for fname in 'sum len sorted reversed list tuple set any all min max'.split():
        try:
            singleArgBuiltins.append(getattr(__builtin__, fname))
        except AttributeError:
            continue
_generatorType = type((y for y in range(1)))

def _xml_escape(data):
    if False:
        while True:
            i = 10
    'Escape &, <, >, ", \', etc. in a string of data.'
    from_symbols = '&><"\''
    to_symbols = ('&' + s + ';' for s in 'amp gt lt quot apos'.split())
    for (from_, to_) in zip(from_symbols, to_symbols):
        data = data.replace(from_, to_)
    return data

class _Constants(object):
    pass
alphas = string.ascii_uppercase + string.ascii_lowercase
nums = '0123456789'
hexnums = nums + 'ABCDEFabcdef'
alphanums = alphas + nums
_bslash = chr(92)
printables = ''.join((c for c in string.printable if c not in string.whitespace))

class ParseBaseException(Exception):
    """base exception class for all parsing runtime exceptions"""

    def __init__(self, pstr, loc=0, msg=None, elem=None):
        if False:
            for i in range(10):
                print('nop')
        self.loc = loc
        if msg is None:
            self.msg = pstr
            self.pstr = ''
        else:
            self.msg = msg
            self.pstr = pstr
        self.parserElement = elem
        self.args = (pstr, loc, msg)

    @classmethod
    def _from_exception(cls, pe):
        if False:
            for i in range(10):
                print('nop')
        '\n        internal factory method to simplify creating one type of ParseException \n        from another - avoids having __init__ signature conflicts among subclasses\n        '
        return cls(pe.pstr, pe.loc, pe.msg, pe.parserElement)

    def __getattr__(self, aname):
        if False:
            return 10
        'supported attributes by name are:\n            - lineno - returns the line number of the exception text\n            - col - returns the column number of the exception text\n            - line - returns the line containing the exception text\n        '
        if aname == 'lineno':
            return lineno(self.loc, self.pstr)
        elif aname in ('col', 'column'):
            return col(self.loc, self.pstr)
        elif aname == 'line':
            return line(self.loc, self.pstr)
        else:
            raise AttributeError(aname)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s (at char %d), (line:%d, col:%d)' % (self.msg, self.loc, self.lineno, self.column)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return _ustr(self)

    def markInputline(self, markerString='>!<'):
        if False:
            for i in range(10):
                print('nop')
        'Extracts the exception line from the input string, and marks\n           the location of the exception with a special symbol.\n        '
        line_str = self.line
        line_column = self.column - 1
        if markerString:
            line_str = ''.join((line_str[:line_column], markerString, line_str[line_column:]))
        return line_str.strip()

    def __dir__(self):
        if False:
            print('Hello World!')
        return 'lineno col line'.split() + dir(type(self))

class ParseException(ParseBaseException):
    """
    Exception thrown when parse expressions don't match class;
    supported attributes by name are:
     - lineno - returns the line number of the exception text
     - col - returns the column number of the exception text
     - line - returns the line containing the exception text
        
    Example::
        try:
            Word(nums).setName("integer").parseString("ABC")
        except ParseException as pe:
            print(pe)
            print("column: {}".format(pe.col))
            
    prints::
       Expected integer (at char 0), (line:1, col:1)
        column: 1
    """
    pass

class ParseFatalException(ParseBaseException):
    """user-throwable exception thrown when inconsistent parse content
       is found; stops all parsing immediately"""
    pass

class ParseSyntaxException(ParseFatalException):
    """just like L{ParseFatalException}, but thrown internally when an
       L{ErrorStop<And._ErrorStop>} ('-' operator) indicates that parsing is to stop 
       immediately because an unbacktrackable syntax error has been found"""
    pass

class RecursiveGrammarException(Exception):
    """exception thrown by L{ParserElement.validate} if the grammar could be improperly recursive"""

    def __init__(self, parseElementList):
        if False:
            i = 10
            return i + 15
        self.parseElementTrace = parseElementList

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'RecursiveGrammarException: %s' % self.parseElementTrace

class _ParseResultsWithOffset(object):

    def __init__(self, p1, p2):
        if False:
            for i in range(10):
                print('nop')
        self.tup = (p1, p2)

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        return self.tup[i]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.tup[0])

    def setOffset(self, i):
        if False:
            for i in range(10):
                print('nop')
        self.tup = (self.tup[0], i)

class ParseResults(object):
    """
    Structured parse results, to provide multiple means of access to the parsed data:
       - as a list (C{len(results)})
       - by list index (C{results[0], results[1]}, etc.)
       - by attribute (C{results.<resultsName>} - see L{ParserElement.setResultsName})

    Example::
        integer = Word(nums)
        date_str = (integer.setResultsName("year") + '/' 
                        + integer.setResultsName("month") + '/' 
                        + integer.setResultsName("day"))
        # equivalent form:
        # date_str = integer("year") + '/' + integer("month") + '/' + integer("day")

        # parseString returns a ParseResults object
        result = date_str.parseString("1999/12/31")

        def test(s, fn=repr):
            print("%s -> %s" % (s, fn(eval(s))))
        test("list(result)")
        test("result[0]")
        test("result['month']")
        test("result.day")
        test("'month' in result")
        test("'minutes' in result")
        test("result.dump()", str)
    prints::
        list(result) -> ['1999', '/', '12', '/', '31']
        result[0] -> '1999'
        result['month'] -> '12'
        result.day -> '31'
        'month' in result -> True
        'minutes' in result -> False
        result.dump() -> ['1999', '/', '12', '/', '31']
        - day: 31
        - month: 12
        - year: 1999
    """

    def __new__(cls, toklist=None, name=None, asList=True, modal=True):
        if False:
            return 10
        if isinstance(toklist, cls):
            return toklist
        retobj = object.__new__(cls)
        retobj.__doinit = True
        return retobj

    def __init__(self, toklist=None, name=None, asList=True, modal=True, isinstance=isinstance):
        if False:
            print('Hello World!')
        if self.__doinit:
            self.__doinit = False
            self.__name = None
            self.__parent = None
            self.__accumNames = {}
            self.__asList = asList
            self.__modal = modal
            if toklist is None:
                toklist = []
            if isinstance(toklist, list):
                self.__toklist = toklist[:]
            elif isinstance(toklist, _generatorType):
                self.__toklist = list(toklist)
            else:
                self.__toklist = [toklist]
            self.__tokdict = dict()
        if name is not None and name:
            if not modal:
                self.__accumNames[name] = 0
            if isinstance(name, int):
                name = _ustr(name)
            self.__name = name
            if not (isinstance(toklist, (type(None), basestring, list)) and toklist in (None, '', [])):
                if isinstance(toklist, basestring):
                    toklist = [toklist]
                if asList:
                    if isinstance(toklist, ParseResults):
                        self[name] = _ParseResultsWithOffset(toklist.copy(), 0)
                    else:
                        self[name] = _ParseResultsWithOffset(ParseResults(toklist[0]), 0)
                    self[name].__name = name
                else:
                    try:
                        self[name] = toklist[0]
                    except (KeyError, TypeError, IndexError):
                        self[name] = toklist

    def __getitem__(self, i):
        if False:
            return 10
        if isinstance(i, (int, slice)):
            return self.__toklist[i]
        elif i not in self.__accumNames:
            return self.__tokdict[i][-1][0]
        else:
            return ParseResults([v[0] for v in self.__tokdict[i]])

    def __setitem__(self, k, v, isinstance=isinstance):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(v, _ParseResultsWithOffset):
            self.__tokdict[k] = self.__tokdict.get(k, list()) + [v]
            sub = v[0]
        elif isinstance(k, (int, slice)):
            self.__toklist[k] = v
            sub = v
        else:
            self.__tokdict[k] = self.__tokdict.get(k, list()) + [_ParseResultsWithOffset(v, 0)]
            sub = v
        if isinstance(sub, ParseResults):
            sub.__parent = wkref(self)

    def __delitem__(self, i):
        if False:
            while True:
                i = 10
        if isinstance(i, (int, slice)):
            mylen = len(self.__toklist)
            del self.__toklist[i]
            if isinstance(i, int):
                if i < 0:
                    i += mylen
                i = slice(i, i + 1)
            removed = list(range(*i.indices(mylen)))
            removed.reverse()
            for (name, occurrences) in self.__tokdict.items():
                for j in removed:
                    for (k, (value, position)) in enumerate(occurrences):
                        occurrences[k] = _ParseResultsWithOffset(value, position - (position > j))
        else:
            del self.__tokdict[i]

    def __contains__(self, k):
        if False:
            i = 10
            return i + 15
        return k in self.__tokdict

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.__toklist)

    def __bool__(self):
        if False:
            print('Hello World!')
        return not not self.__toklist
    __nonzero__ = __bool__

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.__toklist)

    def __reversed__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.__toklist[::-1])

    def _iterkeys(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self.__tokdict, 'iterkeys'):
            return self.__tokdict.iterkeys()
        else:
            return iter(self.__tokdict)

    def _itervalues(self):
        if False:
            while True:
                i = 10
        return (self[k] for k in self._iterkeys())

    def _iteritems(self):
        if False:
            print('Hello World!')
        return ((k, self[k]) for k in self._iterkeys())
    if PY_3:
        keys = _iterkeys
        'Returns an iterator of all named result keys (Python 3.x only).'
        values = _itervalues
        'Returns an iterator of all named result values (Python 3.x only).'
        items = _iteritems
        'Returns an iterator of all named result key-value tuples (Python 3.x only).'
    else:
        iterkeys = _iterkeys
        'Returns an iterator of all named result keys (Python 2.x only).'
        itervalues = _itervalues
        'Returns an iterator of all named result values (Python 2.x only).'
        iteritems = _iteritems
        'Returns an iterator of all named result key-value tuples (Python 2.x only).'

        def keys(self):
            if False:
                i = 10
                return i + 15
            'Returns all named result keys (as a list in Python 2.x, as an iterator in Python 3.x).'
            return list(self.iterkeys())

        def values(self):
            if False:
                print('Hello World!')
            'Returns all named result values (as a list in Python 2.x, as an iterator in Python 3.x).'
            return list(self.itervalues())

        def items(self):
            if False:
                print('Hello World!')
            'Returns all named result key-values (as a list of tuples in Python 2.x, as an iterator in Python 3.x).'
            return list(self.iteritems())

    def haskeys(self):
        if False:
            while True:
                i = 10
        'Since keys() returns an iterator, this method is helpful in bypassing\n           code that looks for the existence of any defined results names.'
        return bool(self.__tokdict)

    def pop(self, *args, **kwargs):
        if False:
            return 10
        '\n        Removes and returns item at specified index (default=C{last}).\n        Supports both C{list} and C{dict} semantics for C{pop()}. If passed no\n        argument or an integer argument, it will use C{list} semantics\n        and pop tokens from the list of parsed tokens. If passed a \n        non-integer argument (most likely a string), it will use C{dict}\n        semantics and pop the corresponding value from any defined \n        results names. A second default return value argument is \n        supported, just as in C{dict.pop()}.\n\n        Example::\n            def remove_first(tokens):\n                tokens.pop(0)\n            print(OneOrMore(Word(nums)).parseString("0 123 321")) # -> [\'0\', \'123\', \'321\']\n            print(OneOrMore(Word(nums)).addParseAction(remove_first).parseString("0 123 321")) # -> [\'123\', \'321\']\n\n            label = Word(alphas)\n            patt = label("LABEL") + OneOrMore(Word(nums))\n            print(patt.parseString("AAB 123 321").dump())\n\n            # Use pop() in a parse action to remove named result (note that corresponding value is not\n            # removed from list form of results)\n            def remove_LABEL(tokens):\n                tokens.pop("LABEL")\n                return tokens\n            patt.addParseAction(remove_LABEL)\n            print(patt.parseString("AAB 123 321").dump())\n        prints::\n            [\'AAB\', \'123\', \'321\']\n            - LABEL: AAB\n\n            [\'AAB\', \'123\', \'321\']\n        '
        if not args:
            args = [-1]
        for (k, v) in kwargs.items():
            if k == 'default':
                args = (args[0], v)
            else:
                raise TypeError("pop() got an unexpected keyword argument '%s'" % k)
        if isinstance(args[0], int) or len(args) == 1 or args[0] in self:
            index = args[0]
            ret = self[index]
            del self[index]
            return ret
        else:
            defaultvalue = args[1]
            return defaultvalue

    def get(self, key, defaultValue=None):
        if False:
            return 10
        '\n        Returns named result matching the given key, or if there is no\n        such name, then returns the given C{defaultValue} or C{None} if no\n        C{defaultValue} is specified.\n\n        Similar to C{dict.get()}.\n        \n        Example::\n            integer = Word(nums)\n            date_str = integer("year") + \'/\' + integer("month") + \'/\' + integer("day")           \n\n            result = date_str.parseString("1999/12/31")\n            print(result.get("year")) # -> \'1999\'\n            print(result.get("hour", "not specified")) # -> \'not specified\'\n            print(result.get("hour")) # -> None\n        '
        if key in self:
            return self[key]
        else:
            return defaultValue

    def insert(self, index, insStr):
        if False:
            return 10
        '\n        Inserts new element at location index in the list of parsed tokens.\n        \n        Similar to C{list.insert()}.\n\n        Example::\n            print(OneOrMore(Word(nums)).parseString("0 123 321")) # -> [\'0\', \'123\', \'321\']\n\n            # use a parse action to insert the parse location in the front of the parsed results\n            def insert_locn(locn, tokens):\n                tokens.insert(0, locn)\n            print(OneOrMore(Word(nums)).addParseAction(insert_locn).parseString("0 123 321")) # -> [0, \'0\', \'123\', \'321\']\n        '
        self.__toklist.insert(index, insStr)
        for (name, occurrences) in self.__tokdict.items():
            for (k, (value, position)) in enumerate(occurrences):
                occurrences[k] = _ParseResultsWithOffset(value, position + (position > index))

    def append(self, item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add single element to end of ParseResults list of elements.\n\n        Example::\n            print(OneOrMore(Word(nums)).parseString("0 123 321")) # -> [\'0\', \'123\', \'321\']\n            \n            # use a parse action to compute the sum of the parsed integers, and add it to the end\n            def append_sum(tokens):\n                tokens.append(sum(map(int, tokens)))\n            print(OneOrMore(Word(nums)).addParseAction(append_sum).parseString("0 123 321")) # -> [\'0\', \'123\', \'321\', 444]\n        '
        self.__toklist.append(item)

    def extend(self, itemseq):
        if False:
            while True:
                i = 10
        '\n        Add sequence of elements to end of ParseResults list of elements.\n\n        Example::\n            patt = OneOrMore(Word(alphas))\n            \n            # use a parse action to append the reverse of the matched strings, to make a palindrome\n            def make_palindrome(tokens):\n                tokens.extend(reversed([t[::-1] for t in tokens]))\n                return \'\'.join(tokens)\n            print(patt.addParseAction(make_palindrome).parseString("lskdj sdlkjf lksd")) # -> \'lskdjsdlkjflksddsklfjkldsjdksl\'\n        '
        if isinstance(itemseq, ParseResults):
            self += itemseq
        else:
            self.__toklist.extend(itemseq)

    def clear(self):
        if False:
            while True:
                i = 10
        '\n        Clear all elements and results names.\n        '
        del self.__toklist[:]
        self.__tokdict.clear()

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        try:
            return self[name]
        except KeyError:
            return ''
        if name in self.__tokdict:
            if name not in self.__accumNames:
                return self.__tokdict[name][-1][0]
            else:
                return ParseResults([v[0] for v in self.__tokdict[name]])
        else:
            return ''

    def __add__(self, other):
        if False:
            print('Hello World!')
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other):
        if False:
            print('Hello World!')
        if other.__tokdict:
            offset = len(self.__toklist)
            addoffset = lambda a: offset if a < 0 else a + offset
            otheritems = other.__tokdict.items()
            otherdictitems = [(k, _ParseResultsWithOffset(v[0], addoffset(v[1]))) for (k, vlist) in otheritems for v in vlist]
            for (k, v) in otherdictitems:
                self[k] = v
                if isinstance(v[0], ParseResults):
                    v[0].__parent = wkref(self)
        self.__toklist += other.__toklist
        self.__accumNames.update(other.__accumNames)
        return self

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, int) and other == 0:
            return self.copy()
        else:
            return other + self

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '(%s, %s)' % (repr(self.__toklist), repr(self.__tokdict))

    def __str__(self):
        if False:
            return 10
        return '[' + ', '.join((_ustr(i) if isinstance(i, ParseResults) else repr(i) for i in self.__toklist)) + ']'

    def _asStringList(self, sep=''):
        if False:
            print('Hello World!')
        out = []
        for item in self.__toklist:
            if out and sep:
                out.append(sep)
            if isinstance(item, ParseResults):
                out += item._asStringList()
            else:
                out.append(_ustr(item))
        return out

    def asList(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the parse results as a nested list of matching tokens, all converted to strings.\n\n        Example::\n            patt = OneOrMore(Word(alphas))\n            result = patt.parseString("sldkj lsdkj sldkj")\n            # even though the result prints in string-like form, it is actually a pyparsing ParseResults\n            print(type(result), result) # -> <class \'pyparsing.ParseResults\'> [\'sldkj\', \'lsdkj\', \'sldkj\']\n            \n            # Use asList() to create an actual list\n            result_list = result.asList()\n            print(type(result_list), result_list) # -> <class \'list\'> [\'sldkj\', \'lsdkj\', \'sldkj\']\n        '
        return [res.asList() if isinstance(res, ParseResults) else res for res in self.__toklist]

    def asDict(self):
        if False:
            return 10
        '\n        Returns the named parse results as a nested dictionary.\n\n        Example::\n            integer = Word(nums)\n            date_str = integer("year") + \'/\' + integer("month") + \'/\' + integer("day")\n            \n            result = date_str.parseString(\'12/31/1999\')\n            print(type(result), repr(result)) # -> <class \'pyparsing.ParseResults\'> ([\'12\', \'/\', \'31\', \'/\', \'1999\'], {\'day\': [(\'1999\', 4)], \'year\': [(\'12\', 0)], \'month\': [(\'31\', 2)]})\n            \n            result_dict = result.asDict()\n            print(type(result_dict), repr(result_dict)) # -> <class \'dict\'> {\'day\': \'1999\', \'year\': \'12\', \'month\': \'31\'}\n\n            # even though a ParseResults supports dict-like access, sometime you just need to have a dict\n            import json\n            print(json.dumps(result)) # -> Exception: TypeError: ... is not JSON serializable\n            print(json.dumps(result.asDict())) # -> {"month": "31", "day": "1999", "year": "12"}\n        '
        if PY_3:
            item_fn = self.items
        else:
            item_fn = self.iteritems

        def toItem(obj):
            if False:
                while True:
                    i = 10
            if isinstance(obj, ParseResults):
                if obj.haskeys():
                    return obj.asDict()
                else:
                    return [toItem(v) for v in obj]
            else:
                return obj
        return dict(((k, toItem(v)) for (k, v) in item_fn()))

    def copy(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a new copy of a C{ParseResults} object.\n        '
        ret = ParseResults(self.__toklist)
        ret.__tokdict = self.__tokdict.copy()
        ret.__parent = self.__parent
        ret.__accumNames.update(self.__accumNames)
        ret.__name = self.__name
        return ret

    def asXML(self, doctag=None, namedItemsOnly=False, indent='', formatted=True):
        if False:
            print('Hello World!')
        '\n        (Deprecated) Returns the parse results as XML. Tags are created for tokens and lists that have defined results names.\n        '
        nl = '\n'
        out = []
        namedItems = dict(((v[1], k) for (k, vlist) in self.__tokdict.items() for v in vlist))
        nextLevelIndent = indent + '  '
        if not formatted:
            indent = ''
            nextLevelIndent = ''
            nl = ''
        selfTag = None
        if doctag is not None:
            selfTag = doctag
        elif self.__name:
            selfTag = self.__name
        if not selfTag:
            if namedItemsOnly:
                return ''
            else:
                selfTag = 'ITEM'
        out += [nl, indent, '<', selfTag, '>']
        for (i, res) in enumerate(self.__toklist):
            if isinstance(res, ParseResults):
                if i in namedItems:
                    out += [res.asXML(namedItems[i], namedItemsOnly and doctag is None, nextLevelIndent, formatted)]
                else:
                    out += [res.asXML(None, namedItemsOnly and doctag is None, nextLevelIndent, formatted)]
            else:
                resTag = None
                if i in namedItems:
                    resTag = namedItems[i]
                if not resTag:
                    if namedItemsOnly:
                        continue
                    else:
                        resTag = 'ITEM'
                xmlBodyText = _xml_escape(_ustr(res))
                out += [nl, nextLevelIndent, '<', resTag, '>', xmlBodyText, '</', resTag, '>']
        out += [nl, indent, '</', selfTag, '>']
        return ''.join(out)

    def __lookup(self, sub):
        if False:
            print('Hello World!')
        for (k, vlist) in self.__tokdict.items():
            for (v, loc) in vlist:
                if sub is v:
                    return k
        return None

    def getName(self):
        if False:
            return 10
        '\n        Returns the results name for this token expression. Useful when several \n        different expressions might match at a particular location.\n\n        Example::\n            integer = Word(nums)\n            ssn_expr = Regex(r"\\d\\d\\d-\\d\\d-\\d\\d\\d\\d")\n            house_number_expr = Suppress(\'#\') + Word(nums, alphanums)\n            user_data = (Group(house_number_expr)("house_number") \n                        | Group(ssn_expr)("ssn")\n                        | Group(integer)("age"))\n            user_info = OneOrMore(user_data)\n            \n            result = user_info.parseString("22 111-22-3333 #221B")\n            for item in result:\n                print(item.getName(), \':\', item[0])\n        prints::\n            age : 22\n            ssn : 111-22-3333\n            house_number : 221B\n        '
        if self.__name:
            return self.__name
        elif self.__parent:
            par = self.__parent()
            if par:
                return par.__lookup(self)
            else:
                return None
        elif len(self) == 1 and len(self.__tokdict) == 1 and (next(iter(self.__tokdict.values()))[0][1] in (0, -1)):
            return next(iter(self.__tokdict.keys()))
        else:
            return None

    def dump(self, indent='', depth=0, full=True):
        if False:
            return 10
        '\n        Diagnostic method for listing out the contents of a C{ParseResults}.\n        Accepts an optional C{indent} argument so that this string can be embedded\n        in a nested display of other data.\n\n        Example::\n            integer = Word(nums)\n            date_str = integer("year") + \'/\' + integer("month") + \'/\' + integer("day")\n            \n            result = date_str.parseString(\'12/31/1999\')\n            print(result.dump())\n        prints::\n            [\'12\', \'/\', \'31\', \'/\', \'1999\']\n            - day: 1999\n            - month: 31\n            - year: 12\n        '
        out = []
        NL = '\n'
        out.append(indent + _ustr(self.asList()))
        if full:
            if self.haskeys():
                items = sorted(((str(k), v) for (k, v) in self.items()))
                for (k, v) in items:
                    if out:
                        out.append(NL)
                    out.append('%s%s- %s: ' % (indent, '  ' * depth, k))
                    if isinstance(v, ParseResults):
                        if v:
                            out.append(v.dump(indent, depth + 1))
                        else:
                            out.append(_ustr(v))
                    else:
                        out.append(repr(v))
            elif any((isinstance(vv, ParseResults) for vv in self)):
                v = self
                for (i, vv) in enumerate(v):
                    if isinstance(vv, ParseResults):
                        out.append('\n%s%s[%d]:\n%s%s%s' % (indent, '  ' * depth, i, indent, '  ' * (depth + 1), vv.dump(indent, depth + 1)))
                    else:
                        out.append('\n%s%s[%d]:\n%s%s%s' % (indent, '  ' * depth, i, indent, '  ' * (depth + 1), _ustr(vv)))
        return ''.join(out)

    def pprint(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Pretty-printer for parsed results as a list, using the C{pprint} module.\n        Accepts additional positional or keyword args as defined for the \n        C{pprint.pprint} method. (U{https://docs.python.org/3/library/pprint.html#pprint.pprint})\n\n        Example::\n            ident = Word(alphas, alphanums)\n            num = Word(nums)\n            func = Forward()\n            term = ident | num | Group(\'(\' + func + \')\')\n            func <<= ident + Group(Optional(delimitedList(term)))\n            result = func.parseString("fna a,b,(fnb c,d,200),100")\n            result.pprint(width=40)\n        prints::\n            [\'fna\',\n             [\'a\',\n              \'b\',\n              [\'(\', \'fnb\', [\'c\', \'d\', \'200\'], \')\'],\n              \'100\']]\n        '
        pprint.pprint(self.asList(), *args, **kwargs)

    def __getstate__(self):
        if False:
            print('Hello World!')
        return (self.__toklist, (self.__tokdict.copy(), self.__parent is not None and self.__parent() or None, self.__accumNames, self.__name))

    def __setstate__(self, state):
        if False:
            return 10
        self.__toklist = state[0]
        (self.__tokdict, par, inAccumNames, self.__name) = state[1]
        self.__accumNames = {}
        self.__accumNames.update(inAccumNames)
        if par is not None:
            self.__parent = wkref(par)
        else:
            self.__parent = None

    def __getnewargs__(self):
        if False:
            i = 10
            return i + 15
        return (self.__toklist, self.__name, self.__asList, self.__modal)

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        return dir(type(self)) + list(self.keys())
collections.MutableMapping.register(ParseResults)

def col(loc, strg):
    if False:
        print('Hello World!')
    'Returns current column within a string, counting newlines as line separators.\n   The first column is number 1.\n\n   Note: the default parsing behavior is to expand tabs in the input string\n   before starting the parsing process.  See L{I{ParserElement.parseString}<ParserElement.parseString>} for more information\n   on parsing strings containing C{<TAB>}s, and suggested methods to maintain a\n   consistent view of the parsed string, the parse location, and line and column\n   positions within the parsed string.\n   '
    s = strg
    return 1 if 0 < loc < len(s) and s[loc - 1] == '\n' else loc - s.rfind('\n', 0, loc)

def lineno(loc, strg):
    if False:
        i = 10
        return i + 15
    'Returns current line number within a string, counting newlines as line separators.\n   The first line is number 1.\n\n   Note: the default parsing behavior is to expand tabs in the input string\n   before starting the parsing process.  See L{I{ParserElement.parseString}<ParserElement.parseString>} for more information\n   on parsing strings containing C{<TAB>}s, and suggested methods to maintain a\n   consistent view of the parsed string, the parse location, and line and column\n   positions within the parsed string.\n   '
    return strg.count('\n', 0, loc) + 1

def line(loc, strg):
    if False:
        for i in range(10):
            print('nop')
    'Returns the line of text containing loc within a string, counting newlines as line separators.\n       '
    lastCR = strg.rfind('\n', 0, loc)
    nextCR = strg.find('\n', loc)
    if nextCR >= 0:
        return strg[lastCR + 1:nextCR]
    else:
        return strg[lastCR + 1:]

def _defaultStartDebugAction(instring, loc, expr):
    if False:
        print('Hello World!')
    print('Match ' + _ustr(expr) + ' at loc ' + _ustr(loc) + '(%d,%d)' % (lineno(loc, instring), col(loc, instring)))

def _defaultSuccessDebugAction(instring, startloc, endloc, expr, toks):
    if False:
        i = 10
        return i + 15
    print('Matched ' + _ustr(expr) + ' -> ' + str(toks.asList()))

def _defaultExceptionDebugAction(instring, loc, expr, exc):
    if False:
        return 10
    print('Exception raised:' + _ustr(exc))

def nullDebugAction(*args):
    if False:
        return 10
    "'Do-nothing' debug action, to suppress debugging output during parsing."
    pass
'decorator to trim function calls to match the arity of the target'

def _trim_arity(func, maxargs=2):
    if False:
        print('Hello World!')
    if func in singleArgBuiltins:
        return lambda s, l, t: func(t)
    limit = [0]
    foundArity = [False]

    def extract_stack(limit=0):
        if False:
            i = 10
            return i + 15
        offset = -2
        frame_summary = traceback.extract_stack(limit=-offset + limit - 1)[offset]
        return [(frame_summary.filename, frame_summary.lineno)]

    def extract_tb(tb, limit=0):
        if False:
            i = 10
            return i + 15
        frames = traceback.extract_tb(tb, limit=limit)
        frame_summary = frames[-1]
        return [(frame_summary.filename, frame_summary.lineno)]
    LINE_DIFF = 6
    this_line = extract_stack(limit=2)[-1]
    pa_call_line_synth = (this_line[0], this_line[1] + LINE_DIFF)

    def wrapper(*args):
        if False:
            while True:
                i = 10
        while 1:
            try:
                ret = func(*args[limit[0]:])
                foundArity[0] = True
                return ret
            except TypeError:
                if foundArity[0]:
                    raise
                else:
                    try:
                        tb = sys.exc_info()[-1]
                        if not extract_tb(tb, limit=2)[-1][:2] == pa_call_line_synth:
                            raise
                    finally:
                        del tb
                if limit[0] <= maxargs:
                    limit[0] += 1
                    continue
                raise
    func_name = '<parse action>'
    try:
        func_name = getattr(func, '__name__', getattr(func, '__class__').__name__)
    except Exception:
        func_name = str(func)
    wrapper.__name__ = func_name
    return wrapper

class ParserElement(object):
    """Abstract base level parser element class."""
    DEFAULT_WHITE_CHARS = ' \n\t\r'
    verbose_stacktrace = False

    @staticmethod
    def setDefaultWhitespaceChars(chars):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overrides the default whitespace chars\n\n        Example::\n            # default whitespace chars are space, <TAB> and newline\n            OneOrMore(Word(alphas)).parseString("abc def\\nghi jkl")  # -> [\'abc\', \'def\', \'ghi\', \'jkl\']\n            \n            # change to just treat newline as significant\n            ParserElement.setDefaultWhitespaceChars(" \\t")\n            OneOrMore(Word(alphas)).parseString("abc def\\nghi jkl")  # -> [\'abc\', \'def\']\n        '
        ParserElement.DEFAULT_WHITE_CHARS = chars

    @staticmethod
    def inlineLiteralsUsing(cls):
        if False:
            print('Hello World!')
        '\n        Set class to be used for inclusion of string literals into a parser.\n        \n        Example::\n            # default literal class used is Literal\n            integer = Word(nums)\n            date_str = integer("year") + \'/\' + integer("month") + \'/\' + integer("day")           \n\n            date_str.parseString("1999/12/31")  # -> [\'1999\', \'/\', \'12\', \'/\', \'31\']\n\n\n            # change to Suppress\n            ParserElement.inlineLiteralsUsing(Suppress)\n            date_str = integer("year") + \'/\' + integer("month") + \'/\' + integer("day")           \n\n            date_str.parseString("1999/12/31")  # -> [\'1999\', \'12\', \'31\']\n        '
        ParserElement._literalStringClass = cls

    def __init__(self, savelist=False):
        if False:
            return 10
        self.parseAction = list()
        self.failAction = None
        self.strRepr = None
        self.resultsName = None
        self.saveAsList = savelist
        self.skipWhitespace = True
        self.whiteChars = ParserElement.DEFAULT_WHITE_CHARS
        self.copyDefaultWhiteChars = True
        self.mayReturnEmpty = False
        self.keepTabs = False
        self.ignoreExprs = list()
        self.debug = False
        self.streamlined = False
        self.mayIndexError = True
        self.errmsg = ''
        self.modalResults = True
        self.debugActions = (None, None, None)
        self.re = None
        self.callPreparse = True
        self.callDuringTry = False

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Make a copy of this C{ParserElement}.  Useful for defining different parse actions\n        for the same parsing pattern, using copies of the original parse element.\n        \n        Example::\n            integer = Word(nums).setParseAction(lambda toks: int(toks[0]))\n            integerK = integer.copy().addParseAction(lambda toks: toks[0]*1024) + Suppress("K")\n            integerM = integer.copy().addParseAction(lambda toks: toks[0]*1024*1024) + Suppress("M")\n            \n            print(OneOrMore(integerK | integerM | integer).parseString("5K 100 640K 256M"))\n        prints::\n            [5120, 100, 655360, 268435456]\n        Equivalent form of C{expr.copy()} is just C{expr()}::\n            integerM = integer().addParseAction(lambda toks: toks[0]*1024*1024) + Suppress("M")\n        '
        cpy = copy.copy(self)
        cpy.parseAction = self.parseAction[:]
        cpy.ignoreExprs = self.ignoreExprs[:]
        if self.copyDefaultWhiteChars:
            cpy.whiteChars = ParserElement.DEFAULT_WHITE_CHARS
        return cpy

    def setName(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Define name for this expression, makes debugging and exception messages clearer.\n        \n        Example::\n            Word(nums).parseString("ABC")  # -> Exception: Expected W:(0123...) (at char 0), (line:1, col:1)\n            Word(nums).setName("integer").parseString("ABC")  # -> Exception: Expected integer (at char 0), (line:1, col:1)\n        '
        self.name = name
        self.errmsg = 'Expected ' + self.name
        if hasattr(self, 'exception'):
            self.exception.msg = self.errmsg
        return self

    def setResultsName(self, name, listAllMatches=False):
        if False:
            return 10
        '\n        Define name for referencing matching tokens as a nested attribute\n        of the returned parse results.\n        NOTE: this returns a *copy* of the original C{ParserElement} object;\n        this is so that the client can define a basic element, such as an\n        integer, and reference it in multiple places with different names.\n\n        You can also set results names using the abbreviated syntax,\n        C{expr("name")} in place of C{expr.setResultsName("name")} - \n        see L{I{__call__}<__call__>}.\n\n        Example::\n            date_str = (integer.setResultsName("year") + \'/\' \n                        + integer.setResultsName("month") + \'/\' \n                        + integer.setResultsName("day"))\n\n            # equivalent form:\n            date_str = integer("year") + \'/\' + integer("month") + \'/\' + integer("day")\n        '
        newself = self.copy()
        if name.endswith('*'):
            name = name[:-1]
            listAllMatches = True
        newself.resultsName = name
        newself.modalResults = not listAllMatches
        return newself

    def setBreak(self, breakFlag=True):
        if False:
            print('Hello World!')
        'Method to invoke the Python pdb debugger when this element is\n           about to be parsed. Set C{breakFlag} to True to enable, False to\n           disable.\n        '
        if breakFlag:
            _parseMethod = self._parse

            def breaker(instring, loc, doActions=True, callPreParse=True):
                if False:
                    return 10
                import pdb
                pdb.set_trace()
                return _parseMethod(instring, loc, doActions, callPreParse)
            breaker._originalParseMethod = _parseMethod
            self._parse = breaker
        elif hasattr(self._parse, '_originalParseMethod'):
            self._parse = self._parse._originalParseMethod
        return self

    def setParseAction(self, *fns, **kwargs):
        if False:
            return 10
        '\n        Define one or more actions to perform when successfully matching parse element definition.\n        Parse action fn is a callable method with 0-3 arguments, called as C{fn(s,loc,toks)},\n        C{fn(loc,toks)}, C{fn(toks)}, or just C{fn()}, where:\n         - s   = the original string being parsed (see note below)\n         - loc = the location of the matching substring\n         - toks = a list of the matched tokens, packaged as a C{L{ParseResults}} object\n        If the functions in fns modify the tokens, they can return them as the return\n        value from fn, and the modified list of tokens will replace the original.\n        Otherwise, fn does not need to return any value.\n\n        Optional keyword arguments:\n         - callDuringTry = (default=C{False}) indicate if parse action should be run during lookaheads and alternate testing\n\n        Note: the default parsing behavior is to expand tabs in the input string\n        before starting the parsing process.  See L{I{parseString}<parseString>} for more information\n        on parsing strings containing C{<TAB>}s, and suggested methods to maintain a\n        consistent view of the parsed string, the parse location, and line and column\n        positions within the parsed string.\n        \n        Example::\n            integer = Word(nums)\n            date_str = integer + \'/\' + integer + \'/\' + integer\n\n            date_str.parseString("1999/12/31")  # -> [\'1999\', \'/\', \'12\', \'/\', \'31\']\n\n            # use parse action to convert to ints at parse time\n            integer = Word(nums).setParseAction(lambda toks: int(toks[0]))\n            date_str = integer + \'/\' + integer + \'/\' + integer\n\n            # note that integer fields are now ints, not strings\n            date_str.parseString("1999/12/31")  # -> [1999, \'/\', 12, \'/\', 31]\n        '
        self.parseAction = list(map(_trim_arity, list(fns)))
        self.callDuringTry = kwargs.get('callDuringTry', False)
        return self

    def addParseAction(self, *fns, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Add one or more parse actions to expression's list of parse actions. See L{I{setParseAction}<setParseAction>}.\n        \n        See examples in L{I{copy}<copy>}.\n        "
        self.parseAction += list(map(_trim_arity, list(fns)))
        self.callDuringTry = self.callDuringTry or kwargs.get('callDuringTry', False)
        return self

    def addCondition(self, *fns, **kwargs):
        if False:
            while True:
                i = 10
        'Add a boolean predicate function to expression\'s list of parse actions. See \n        L{I{setParseAction}<setParseAction>} for function call signatures. Unlike C{setParseAction}, \n        functions passed to C{addCondition} need to return boolean success/fail of the condition.\n\n        Optional keyword arguments:\n         - message = define a custom message to be used in the raised exception\n         - fatal   = if True, will raise ParseFatalException to stop parsing immediately; otherwise will raise ParseException\n         \n        Example::\n            integer = Word(nums).setParseAction(lambda toks: int(toks[0]))\n            year_int = integer.copy()\n            year_int.addCondition(lambda toks: toks[0] >= 2000, message="Only support years 2000 and later")\n            date_str = year_int + \'/\' + integer + \'/\' + integer\n\n            result = date_str.parseString("1999/12/31")  # -> Exception: Only support years 2000 and later (at char 0), (line:1, col:1)\n        '
        msg = kwargs.get('message', 'failed user-defined condition')
        exc_type = ParseFatalException if kwargs.get('fatal', False) else ParseException
        for fn in fns:

            def pa(s, l, t):
                if False:
                    i = 10
                    return i + 15
                if not bool(_trim_arity(fn)(s, l, t)):
                    raise exc_type(s, l, msg)
            self.parseAction.append(pa)
        self.callDuringTry = self.callDuringTry or kwargs.get('callDuringTry', False)
        return self

    def setFailAction(self, fn):
        if False:
            print('Hello World!')
        'Define action to perform if parsing fails at this expression.\n           Fail action fn is a callable function that takes the arguments\n           C{fn(s,loc,expr,err)} where:\n            - s = string being parsed\n            - loc = location where expression match was attempted and failed\n            - expr = the parse expression that failed\n            - err = the exception thrown\n           The function returns no value.  It may throw C{L{ParseFatalException}}\n           if it is desired to stop parsing immediately.'
        self.failAction = fn
        return self

    def _skipIgnorables(self, instring, loc):
        if False:
            i = 10
            return i + 15
        exprsFound = True
        while exprsFound:
            exprsFound = False
            for e in self.ignoreExprs:
                try:
                    while 1:
                        (loc, dummy) = e._parse(instring, loc)
                        exprsFound = True
                except ParseException:
                    pass
        return loc

    def preParse(self, instring, loc):
        if False:
            return 10
        if self.ignoreExprs:
            loc = self._skipIgnorables(instring, loc)
        if self.skipWhitespace:
            wt = self.whiteChars
            instrlen = len(instring)
            while loc < instrlen and instring[loc] in wt:
                loc += 1
        return loc

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        return (loc, [])

    def postParse(self, instring, loc, tokenlist):
        if False:
            i = 10
            return i + 15
        return tokenlist

    def _parseNoCache(self, instring, loc, doActions=True, callPreParse=True):
        if False:
            return 10
        debugging = self.debug
        if debugging or self.failAction:
            if self.debugActions[0]:
                self.debugActions[0](instring, loc, self)
            if callPreParse and self.callPreparse:
                preloc = self.preParse(instring, loc)
            else:
                preloc = loc
            tokensStart = preloc
            try:
                try:
                    (loc, tokens) = self.parseImpl(instring, preloc, doActions)
                except IndexError:
                    raise ParseException(instring, len(instring), self.errmsg, self)
            except ParseBaseException as err:
                if self.debugActions[2]:
                    self.debugActions[2](instring, tokensStart, self, err)
                if self.failAction:
                    self.failAction(instring, tokensStart, self, err)
                raise
        else:
            if callPreParse and self.callPreparse:
                preloc = self.preParse(instring, loc)
            else:
                preloc = loc
            tokensStart = preloc
            if self.mayIndexError or loc >= len(instring):
                try:
                    (loc, tokens) = self.parseImpl(instring, preloc, doActions)
                except IndexError:
                    raise ParseException(instring, len(instring), self.errmsg, self)
            else:
                (loc, tokens) = self.parseImpl(instring, preloc, doActions)
        tokens = self.postParse(instring, loc, tokens)
        retTokens = ParseResults(tokens, self.resultsName, asList=self.saveAsList, modal=self.modalResults)
        if self.parseAction and (doActions or self.callDuringTry):
            if debugging:
                try:
                    for fn in self.parseAction:
                        tokens = fn(instring, tokensStart, retTokens)
                        if tokens is not None:
                            retTokens = ParseResults(tokens, self.resultsName, asList=self.saveAsList and isinstance(tokens, (ParseResults, list)), modal=self.modalResults)
                except ParseBaseException as err:
                    if self.debugActions[2]:
                        self.debugActions[2](instring, tokensStart, self, err)
                    raise
            else:
                for fn in self.parseAction:
                    tokens = fn(instring, tokensStart, retTokens)
                    if tokens is not None:
                        retTokens = ParseResults(tokens, self.resultsName, asList=self.saveAsList and isinstance(tokens, (ParseResults, list)), modal=self.modalResults)
        if debugging:
            if self.debugActions[1]:
                self.debugActions[1](instring, tokensStart, loc, self, retTokens)
        return (loc, retTokens)

    def tryParse(self, instring, loc):
        if False:
            return 10
        try:
            return self._parse(instring, loc, doActions=False)[0]
        except ParseFatalException:
            raise ParseException(instring, loc, self.errmsg, self)

    def canParseNext(self, instring, loc):
        if False:
            i = 10
            return i + 15
        try:
            self.tryParse(instring, loc)
        except (ParseException, IndexError):
            return False
        else:
            return True

    class _UnboundedCache(object):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            cache = {}
            self.not_in_cache = not_in_cache = object()

            def get(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                return cache.get(key, not_in_cache)

            def set(self, key, value):
                if False:
                    i = 10
                    return i + 15
                cache[key] = value

            def clear(self):
                if False:
                    while True:
                        i = 10
                cache.clear()

            def cache_len(self):
                if False:
                    print('Hello World!')
                return len(cache)
            self.get = types.MethodType(get, self)
            self.set = types.MethodType(set, self)
            self.clear = types.MethodType(clear, self)
            self.__len__ = types.MethodType(cache_len, self)
    if _OrderedDict is not None:

        class _FifoCache(object):

            def __init__(self, size):
                if False:
                    while True:
                        i = 10
                self.not_in_cache = not_in_cache = object()
                cache = _OrderedDict()

                def get(self, key):
                    if False:
                        return 10
                    return cache.get(key, not_in_cache)

                def set(self, key, value):
                    if False:
                        while True:
                            i = 10
                    cache[key] = value
                    while len(cache) > size:
                        try:
                            cache.popitem(False)
                        except KeyError:
                            pass

                def clear(self):
                    if False:
                        while True:
                            i = 10
                    cache.clear()

                def cache_len(self):
                    if False:
                        i = 10
                        return i + 15
                    return len(cache)
                self.get = types.MethodType(get, self)
                self.set = types.MethodType(set, self)
                self.clear = types.MethodType(clear, self)
                self.__len__ = types.MethodType(cache_len, self)
    else:

        class _FifoCache(object):

            def __init__(self, size):
                if False:
                    i = 10
                    return i + 15
                self.not_in_cache = not_in_cache = object()
                cache = {}
                key_fifo = collections.deque([], size)

                def get(self, key):
                    if False:
                        print('Hello World!')
                    return cache.get(key, not_in_cache)

                def set(self, key, value):
                    if False:
                        while True:
                            i = 10
                    cache[key] = value
                    while len(key_fifo) > size:
                        cache.pop(key_fifo.popleft(), None)
                    key_fifo.append(key)

                def clear(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    cache.clear()
                    key_fifo.clear()

                def cache_len(self):
                    if False:
                        print('Hello World!')
                    return len(cache)
                self.get = types.MethodType(get, self)
                self.set = types.MethodType(set, self)
                self.clear = types.MethodType(clear, self)
                self.__len__ = types.MethodType(cache_len, self)
    packrat_cache = {}
    packrat_cache_lock = RLock()
    packrat_cache_stats = [0, 0]

    def _parseCache(self, instring, loc, doActions=True, callPreParse=True):
        if False:
            while True:
                i = 10
        (HIT, MISS) = (0, 1)
        lookup = (self, instring, loc, callPreParse, doActions)
        with ParserElement.packrat_cache_lock:
            cache = ParserElement.packrat_cache
            value = cache.get(lookup)
            if value is cache.not_in_cache:
                ParserElement.packrat_cache_stats[MISS] += 1
                try:
                    value = self._parseNoCache(instring, loc, doActions, callPreParse)
                except ParseBaseException as pe:
                    cache.set(lookup, pe.__class__(*pe.args))
                    raise
                else:
                    cache.set(lookup, (value[0], value[1].copy()))
                    return value
            else:
                ParserElement.packrat_cache_stats[HIT] += 1
                if isinstance(value, Exception):
                    raise value
                return (value[0], value[1].copy())
    _parse = _parseNoCache

    @staticmethod
    def resetCache():
        if False:
            return 10
        ParserElement.packrat_cache.clear()
        ParserElement.packrat_cache_stats[:] = [0] * len(ParserElement.packrat_cache_stats)
    _packratEnabled = False

    @staticmethod
    def enablePackrat(cache_size_limit=128):
        if False:
            while True:
                i = 10
        'Enables "packrat" parsing, which adds memoizing to the parsing logic.\n           Repeated parse attempts at the same string location (which happens\n           often in many complex grammars) can immediately return a cached value,\n           instead of re-executing parsing/validating code.  Memoizing is done of\n           both valid results and parsing exceptions.\n           \n           Parameters:\n            - cache_size_limit - (default=C{128}) - if an integer value is provided\n              will limit the size of the packrat cache; if None is passed, then\n              the cache size will be unbounded; if 0 is passed, the cache will\n              be effectively disabled.\n            \n           This speedup may break existing programs that use parse actions that\n           have side-effects.  For this reason, packrat parsing is disabled when\n           you first import pyparsing.  To activate the packrat feature, your\n           program must call the class method C{ParserElement.enablePackrat()}.  If\n           your program uses C{psyco} to "compile as you go", you must call\n           C{enablePackrat} before calling C{psyco.full()}.  If you do not do this,\n           Python will crash.  For best results, call C{enablePackrat()} immediately\n           after importing pyparsing.\n           \n           Example::\n               import pyparsing\n               pyparsing.ParserElement.enablePackrat()\n        '
        if not ParserElement._packratEnabled:
            ParserElement._packratEnabled = True
            if cache_size_limit is None:
                ParserElement.packrat_cache = ParserElement._UnboundedCache()
            else:
                ParserElement.packrat_cache = ParserElement._FifoCache(cache_size_limit)
            ParserElement._parse = ParserElement._parseCache

    def parseString(self, instring, parseAll=False):
        if False:
            return 10
        "\n        Execute the parse expression with the given string.\n        This is the main interface to the client code, once the complete\n        expression has been built.\n\n        If you want the grammar to require that the entire input string be\n        successfully parsed, then set C{parseAll} to True (equivalent to ending\n        the grammar with C{L{StringEnd()}}).\n\n        Note: C{parseString} implicitly calls C{expandtabs()} on the input string,\n        in order to report proper column numbers in parse actions.\n        If the input string contains tabs and\n        the grammar uses parse actions that use the C{loc} argument to index into the\n        string being parsed, you can ensure you have a consistent view of the input\n        string by:\n         - calling C{parseWithTabs} on your grammar before calling C{parseString}\n           (see L{I{parseWithTabs}<parseWithTabs>})\n         - define your parse action using the full C{(s,loc,toks)} signature, and\n           reference the input string using the parse action's C{s} argument\n         - explicitly expand the tabs in your input string before calling\n           C{parseString}\n        \n        Example::\n            Word('a').parseString('aaaaabaaa')  # -> ['aaaaa']\n            Word('a').parseString('aaaaabaaa', parseAll=True)  # -> Exception: Expected end of text\n        "
        ParserElement.resetCache()
        if not self.streamlined:
            self.streamline()
        for e in self.ignoreExprs:
            e.streamline()
        if not self.keepTabs:
            instring = instring.expandtabs()
        try:
            (loc, tokens) = self._parse(instring, 0)
            if parseAll:
                loc = self.preParse(instring, loc)
                se = Empty() + StringEnd()
                se._parse(instring, loc)
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc
        else:
            return tokens

    def scanString(self, instring, maxMatches=_MAX_INT, overlap=False):
        if False:
            i = 10
            return i + 15
        '\n        Scan the input string for expression matches.  Each match will return the\n        matching tokens, start location, and end location.  May be called with optional\n        C{maxMatches} argument, to clip scanning after \'n\' matches are found.  If\n        C{overlap} is specified, then overlapping matches will be reported.\n\n        Note that the start and end locations are reported relative to the string\n        being parsed.  See L{I{parseString}<parseString>} for more information on parsing\n        strings with embedded tabs.\n\n        Example::\n            source = "sldjf123lsdjjkf345sldkjf879lkjsfd987"\n            print(source)\n            for tokens,start,end in Word(alphas).scanString(source):\n                print(\' \'*start + \'^\'*(end-start))\n                print(\' \'*start + tokens[0])\n        \n        prints::\n        \n            sldjf123lsdjjkf345sldkjf879lkjsfd987\n            ^^^^^\n            sldjf\n                    ^^^^^^^\n                    lsdjjkf\n                              ^^^^^^\n                              sldkjf\n                                       ^^^^^^\n                                       lkjsfd\n        '
        if not self.streamlined:
            self.streamline()
        for e in self.ignoreExprs:
            e.streamline()
        if not self.keepTabs:
            instring = _ustr(instring).expandtabs()
        instrlen = len(instring)
        loc = 0
        preparseFn = self.preParse
        parseFn = self._parse
        ParserElement.resetCache()
        matches = 0
        try:
            while loc <= instrlen and matches < maxMatches:
                try:
                    preloc = preparseFn(instring, loc)
                    (nextLoc, tokens) = parseFn(instring, preloc, callPreParse=False)
                except ParseException:
                    loc = preloc + 1
                else:
                    if nextLoc > loc:
                        matches += 1
                        yield (tokens, preloc, nextLoc)
                        if overlap:
                            nextloc = preparseFn(instring, loc)
                            if nextloc > loc:
                                loc = nextLoc
                            else:
                                loc += 1
                        else:
                            loc = nextLoc
                    else:
                        loc = preloc + 1
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def transformString(self, instring):
        if False:
            print('Hello World!')
        '\n        Extension to C{L{scanString}}, to modify matching text with modified tokens that may\n        be returned from a parse action.  To use C{transformString}, define a grammar and\n        attach a parse action to it that modifies the returned token list.\n        Invoking C{transformString()} on a target string will then scan for matches,\n        and replace the matched text patterns according to the logic in the parse\n        action.  C{transformString()} returns the resulting transformed string.\n        \n        Example::\n            wd = Word(alphas)\n            wd.setParseAction(lambda toks: toks[0].title())\n            \n            print(wd.transformString("now is the winter of our discontent made glorious summer by this sun of york."))\n        Prints::\n            Now Is The Winter Of Our Discontent Made Glorious Summer By This Sun Of York.\n        '
        out = []
        lastE = 0
        self.keepTabs = True
        try:
            for (t, s, e) in self.scanString(instring):
                out.append(instring[lastE:s])
                if t:
                    if isinstance(t, ParseResults):
                        out += t.asList()
                    elif isinstance(t, list):
                        out += t
                    else:
                        out.append(t)
                lastE = e
            out.append(instring[lastE:])
            out = [o for o in out if o]
            return ''.join(map(_ustr, _flatten(out)))
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def searchString(self, instring, maxMatches=_MAX_INT):
        if False:
            return 10
        '\n        Another extension to C{L{scanString}}, simplifying the access to the tokens found\n        to match the given parse expression.  May be called with optional\n        C{maxMatches} argument, to clip searching after \'n\' matches are found.\n        \n        Example::\n            # a capitalized word starts with an uppercase letter, followed by zero or more lowercase letters\n            cap_word = Word(alphas.upper(), alphas.lower())\n            \n            print(cap_word.searchString("More than Iron, more than Lead, more than Gold I need Electricity"))\n\n            # the sum() builtin can be used to merge results into a single ParseResults object\n            print(sum(cap_word.searchString("More than Iron, more than Lead, more than Gold I need Electricity")))\n        prints::\n            [[\'More\'], [\'Iron\'], [\'Lead\'], [\'Gold\'], [\'I\'], [\'Electricity\']]\n            [\'More\', \'Iron\', \'Lead\', \'Gold\', \'I\', \'Electricity\']\n        '
        try:
            return ParseResults([t for (t, s, e) in self.scanString(instring, maxMatches)])
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def split(self, instring, maxsplit=_MAX_INT, includeSeparators=False):
        if False:
            print('Hello World!')
        '\n        Generator method to split a string using the given expression as a separator.\n        May be called with optional C{maxsplit} argument, to limit the number of splits;\n        and the optional C{includeSeparators} argument (default=C{False}), if the separating\n        matching text should be included in the split results.\n        \n        Example::        \n            punc = oneOf(list(".,;:/-!?"))\n            print(list(punc.split("This, this?, this sentence, is badly punctuated!")))\n        prints::\n            [\'This\', \' this\', \'\', \' this sentence\', \' is badly punctuated\', \'\']\n        '
        splits = 0
        last = 0
        for (t, s, e) in self.scanString(instring, maxMatches=maxsplit):
            yield instring[last:s]
            if includeSeparators:
                yield t[0]
            last = e
        yield instring[last:]

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Implementation of + operator - returns C{L{And}}. Adding strings to a ParserElement\n        converts them to L{Literal}s by default.\n        \n        Example::\n            greet = Word(alphas) + "," + Word(alphas) + "!"\n            hello = "Hello, World!"\n            print (hello, "->", greet.parseString(hello))\n        Prints::\n            Hello, World! -> [\'Hello\', \',\', \'World\', \'!\']\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return And([self, other])

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implementation of + operator when left operand is not a C{L{ParserElement}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other + self

    def __sub__(self, other):
        if False:
            return 10
        '\n        Implementation of - operator, returns C{L{And}} with error stop\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return self + And._ErrorStop() + other

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implementation of - operator when left operand is not a C{L{ParserElement}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other - self

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        '\n        Implementation of * operator, allows use of C{expr * 3} in place of\n        C{expr + expr + expr}.  Expressions may also me multiplied by a 2-integer\n        tuple, similar to C{{min,max}} multipliers in regular expressions.  Tuples\n        may also include C{None} as in:\n         - C{expr*(n,None)} or C{expr*(n,)} is equivalent\n              to C{expr*n + L{ZeroOrMore}(expr)}\n              (read as "at least n instances of C{expr}")\n         - C{expr*(None,n)} is equivalent to C{expr*(0,n)}\n              (read as "0 to n instances of C{expr}")\n         - C{expr*(None,None)} is equivalent to C{L{ZeroOrMore}(expr)}\n         - C{expr*(1,None)} is equivalent to C{L{OneOrMore}(expr)}\n\n        Note that C{expr*(None,n)} does not raise an exception if\n        more than n exprs exist in the input stream; that is,\n        C{expr*(None,n)} does not enforce a maximum number of expr\n        occurrences.  If this behavior is desired, then write\n        C{expr*(None,n) + ~expr}\n        '
        if isinstance(other, int):
            (minElements, optElements) = (other, 0)
        elif isinstance(other, tuple):
            other = (other + (None, None))[:2]
            if other[0] is None:
                other = (0, other[1])
            if isinstance(other[0], int) and other[1] is None:
                if other[0] == 0:
                    return ZeroOrMore(self)
                if other[0] == 1:
                    return OneOrMore(self)
                else:
                    return self * other[0] + ZeroOrMore(self)
            elif isinstance(other[0], int) and isinstance(other[1], int):
                (minElements, optElements) = other
                optElements -= minElements
            else:
                raise TypeError("cannot multiply 'ParserElement' and ('%s','%s') objects", type(other[0]), type(other[1]))
        else:
            raise TypeError("cannot multiply 'ParserElement' and '%s' objects", type(other))
        if minElements < 0:
            raise ValueError('cannot multiply ParserElement by negative value')
        if optElements < 0:
            raise ValueError('second tuple value must be greater or equal to first tuple value')
        if minElements == optElements == 0:
            raise ValueError('cannot multiply ParserElement by 0 or (0,0)')
        if optElements:

            def makeOptionalList(n):
                if False:
                    while True:
                        i = 10
                if n > 1:
                    return Optional(self + makeOptionalList(n - 1))
                else:
                    return Optional(self)
            if minElements:
                if minElements == 1:
                    ret = self + makeOptionalList(optElements)
                else:
                    ret = And([self] * minElements) + makeOptionalList(optElements)
            else:
                ret = makeOptionalList(optElements)
        elif minElements == 1:
            ret = self
        else:
            ret = And([self] * minElements)
        return ret

    def __rmul__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__mul__(other)

    def __or__(self, other):
        if False:
            return 10
        '\n        Implementation of | operator - returns C{L{MatchFirst}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return MatchFirst([self, other])

    def __ror__(self, other):
        if False:
            print('Hello World!')
        '\n        Implementation of | operator when left operand is not a C{L{ParserElement}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other | self

    def __xor__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implementation of ^ operator - returns C{L{Or}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return Or([self, other])

    def __rxor__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Implementation of ^ operator when left operand is not a C{L{ParserElement}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other ^ self

    def __and__(self, other):
        if False:
            return 10
        '\n        Implementation of & operator - returns C{L{Each}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return Each([self, other])

    def __rand__(self, other):
        if False:
            while True:
                i = 10
        '\n        Implementation of & operator when left operand is not a C{L{ParserElement}}\n        '
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        if not isinstance(other, ParserElement):
            warnings.warn('Cannot combine element of type %s with ParserElement' % type(other), SyntaxWarning, stacklevel=2)
            return None
        return other & self

    def __invert__(self):
        if False:
            while True:
                i = 10
        '\n        Implementation of ~ operator - returns C{L{NotAny}}\n        '
        return NotAny(self)

    def __call__(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shortcut for C{L{setResultsName}}, with C{listAllMatches=False}.\n        \n        If C{name} is given with a trailing C{\'*\'} character, then C{listAllMatches} will be\n        passed as C{True}.\n           \n        If C{name} is omitted, same as calling C{L{copy}}.\n\n        Example::\n            # these are equivalent\n            userdata = Word(alphas).setResultsName("name") + Word(nums+"-").setResultsName("socsecno")\n            userdata = Word(alphas)("name") + Word(nums+"-")("socsecno")             \n        '
        if name is not None:
            return self.setResultsName(name)
        else:
            return self.copy()

    def suppress(self):
        if False:
            print('Hello World!')
        '\n        Suppresses the output of this C{ParserElement}; useful to keep punctuation from\n        cluttering up returned output.\n        '
        return Suppress(self)

    def leaveWhitespace(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Disables the skipping of whitespace before matching the characters in the\n        C{ParserElement}'s defined pattern.  This is normally only used internally by\n        the pyparsing module, but may be needed in some whitespace-sensitive grammars.\n        "
        self.skipWhitespace = False
        return self

    def setWhitespaceChars(self, chars):
        if False:
            return 10
        '\n        Overrides the default whitespace chars\n        '
        self.skipWhitespace = True
        self.whiteChars = chars
        self.copyDefaultWhiteChars = False
        return self

    def parseWithTabs(self):
        if False:
            return 10
        '\n        Overrides default behavior to expand C{<TAB>}s to spaces before parsing the input string.\n        Must be called before C{parseString} when the input grammar contains elements that\n        match C{<TAB>} characters.\n        '
        self.keepTabs = True
        return self

    def ignore(self, other):
        if False:
            return 10
        "\n        Define expression to be ignored (e.g., comments) while doing pattern\n        matching; may be called repeatedly, to define multiple comment or other\n        ignorable patterns.\n        \n        Example::\n            patt = OneOrMore(Word(alphas))\n            patt.parseString('ablaj /* comment */ lskjd') # -> ['ablaj']\n            \n            patt.ignore(cStyleComment)\n            patt.parseString('ablaj /* comment */ lskjd') # -> ['ablaj', 'lskjd']\n        "
        if isinstance(other, basestring):
            other = Suppress(other)
        if isinstance(other, Suppress):
            if other not in self.ignoreExprs:
                self.ignoreExprs.append(other)
        else:
            self.ignoreExprs.append(Suppress(other.copy()))
        return self

    def setDebugActions(self, startAction, successAction, exceptionAction):
        if False:
            while True:
                i = 10
        '\n        Enable display of debugging messages while doing pattern matching.\n        '
        self.debugActions = (startAction or _defaultStartDebugAction, successAction or _defaultSuccessDebugAction, exceptionAction or _defaultExceptionDebugAction)
        self.debug = True
        return self

    def setDebug(self, flag=True):
        if False:
            i = 10
            return i + 15
        '\n        Enable display of debugging messages while doing pattern matching.\n        Set C{flag} to True to enable, False to disable.\n\n        Example::\n            wd = Word(alphas).setName("alphaword")\n            integer = Word(nums).setName("numword")\n            term = wd | integer\n            \n            # turn on debugging for wd\n            wd.setDebug()\n\n            OneOrMore(term).parseString("abc 123 xyz 890")\n        \n        prints::\n            Match alphaword at loc 0(1,1)\n            Matched alphaword -> [\'abc\']\n            Match alphaword at loc 3(1,4)\n            Exception raised:Expected alphaword (at char 4), (line:1, col:5)\n            Match alphaword at loc 7(1,8)\n            Matched alphaword -> [\'xyz\']\n            Match alphaword at loc 11(1,12)\n            Exception raised:Expected alphaword (at char 12), (line:1, col:13)\n            Match alphaword at loc 15(1,16)\n            Exception raised:Expected alphaword (at char 15), (line:1, col:16)\n\n        The output shown is that produced by the default debug actions - custom debug actions can be\n        specified using L{setDebugActions}. Prior to attempting\n        to match the C{wd} expression, the debugging message C{"Match <exprname> at loc <n>(<line>,<col>)"}\n        is shown. Then if the parse succeeds, a C{"Matched"} message is shown, or an C{"Exception raised"}\n        message is shown. Also note the use of L{setName} to assign a human-readable name to the expression,\n        which makes debugging and exception messages easier to understand - for instance, the default\n        name created for the C{Word} expression without calling C{setName} is C{"W:(ABCD...)"}.\n        '
        if flag:
            self.setDebugActions(_defaultStartDebugAction, _defaultSuccessDebugAction, _defaultExceptionDebugAction)
        else:
            self.debug = False
        return self

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.name

    def __repr__(self):
        if False:
            while True:
                i = 10
        return _ustr(self)

    def streamline(self):
        if False:
            i = 10
            return i + 15
        self.streamlined = True
        self.strRepr = None
        return self

    def checkRecursion(self, parseElementList):
        if False:
            i = 10
            return i + 15
        pass

    def validate(self, validateTrace=[]):
        if False:
            return 10
        '\n        Check defined expressions for valid structure, check for infinite recursive definitions.\n        '
        self.checkRecursion([])

    def parseFile(self, file_or_filename, parseAll=False):
        if False:
            i = 10
            return i + 15
        '\n        Execute the parse expression on the given file or filename.\n        If a filename is specified (instead of a file object),\n        the entire file is opened, read, and closed before parsing.\n        '
        try:
            file_contents = file_or_filename.read()
        except AttributeError:
            with open(file_or_filename, 'r') as f:
                file_contents = f.read()
        try:
            return self.parseString(file_contents, parseAll)
        except ParseBaseException as exc:
            if ParserElement.verbose_stacktrace:
                raise
            else:
                raise exc

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, ParserElement):
            return self is other or vars(self) == vars(other)
        elif isinstance(other, basestring):
            return self.matches(other)
        else:
            return super(ParserElement, self) == other

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(id(self))

    def __req__(self, other):
        if False:
            i = 10
            return i + 15
        return self == other

    def __rne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def matches(self, testString, parseAll=True):
        if False:
            while True:
                i = 10
        '\n        Method for quick testing of a parser against a test string. Good for simple \n        inline microtests of sub expressions while building up larger parser.\n           \n        Parameters:\n         - testString - to test against this expression for a match\n         - parseAll - (default=C{True}) - flag to pass to C{L{parseString}} when running tests\n            \n        Example::\n            expr = Word(nums)\n            assert expr.matches("100")\n        '
        try:
            self.parseString(_ustr(testString), parseAll=parseAll)
            return True
        except ParseBaseException:
            return False

    def runTests(self, tests, parseAll=True, comment='#', fullDump=True, printResults=True, failureTests=False):
        if False:
            return 10
        '\n        Execute the parse expression on a series of test strings, showing each\n        test, the parsed results or where the parse failed. Quick and easy way to\n        run a parse expression against a list of sample strings.\n           \n        Parameters:\n         - tests - a list of separate test strings, or a multiline string of test strings\n         - parseAll - (default=C{True}) - flag to pass to C{L{parseString}} when running tests           \n         - comment - (default=C{\'#\'}) - expression for indicating embedded comments in the test \n              string; pass None to disable comment filtering\n         - fullDump - (default=C{True}) - dump results as list followed by results names in nested outline;\n              if False, only dump nested list\n         - printResults - (default=C{True}) prints test output to stdout\n         - failureTests - (default=C{False}) indicates if these tests are expected to fail parsing\n\n        Returns: a (success, results) tuple, where success indicates that all tests succeeded\n        (or failed if C{failureTests} is True), and the results contain a list of lines of each \n        test\'s output\n        \n        Example::\n            number_expr = pyparsing_common.number.copy()\n\n            result = number_expr.runTests(\'\'\'\n                # unsigned integer\n                100\n                # negative integer\n                -100\n                # float with scientific notation\n                6.02e23\n                # integer with scientific notation\n                1e-12\n                \'\'\')\n            print("Success" if result[0] else "Failed!")\n\n            result = number_expr.runTests(\'\'\'\n                # stray character\n                100Z\n                # missing leading digit before \'.\'\n                -.100\n                # too many \'.\'\n                3.14.159\n                \'\'\', failureTests=True)\n            print("Success" if result[0] else "Failed!")\n        prints::\n            # unsigned integer\n            100\n            [100]\n\n            # negative integer\n            -100\n            [-100]\n\n            # float with scientific notation\n            6.02e23\n            [6.02e+23]\n\n            # integer with scientific notation\n            1e-12\n            [1e-12]\n\n            Success\n            \n            # stray character\n            100Z\n               ^\n            FAIL: Expected end of text (at char 3), (line:1, col:4)\n\n            # missing leading digit before \'.\'\n            -.100\n            ^\n            FAIL: Expected {real number with scientific notation | real number | signed integer} (at char 0), (line:1, col:1)\n\n            # too many \'.\'\n            3.14.159\n                ^\n            FAIL: Expected end of text (at char 4), (line:1, col:5)\n\n            Success\n\n        Each test string must be on a single line. If you want to test a string that spans multiple\n        lines, create a test like this::\n\n            expr.runTest(r"this is a test\\n of strings that spans \\n 3 lines")\n        \n        (Note that this is a raw string literal, you must include the leading \'r\'.)\n        '
        if isinstance(tests, basestring):
            tests = list(map(str.strip, tests.rstrip().splitlines()))
        if isinstance(comment, basestring):
            comment = Literal(comment)
        allResults = []
        comments = []
        success = True
        for t in tests:
            if comment is not None and comment.matches(t, False) or (comments and (not t)):
                comments.append(t)
                continue
            if not t:
                continue
            out = ['\n'.join(comments), t]
            comments = []
            try:
                t = t.replace('\\n', '\n')
                result = self.parseString(t, parseAll=parseAll)
                out.append(result.dump(full=fullDump))
                success = success and (not failureTests)
            except ParseBaseException as pe:
                fatal = '(FATAL)' if isinstance(pe, ParseFatalException) else ''
                if '\n' in t:
                    out.append(line(pe.loc, t))
                    out.append(' ' * (col(pe.loc, t) - 1) + '^' + fatal)
                else:
                    out.append(' ' * pe.loc + '^' + fatal)
                out.append('FAIL: ' + str(pe))
                success = success and failureTests
                result = pe
            except Exception as exc:
                out.append('FAIL-EXCEPTION: ' + str(exc))
                success = success and failureTests
                result = exc
            if printResults:
                if fullDump:
                    out.append('')
                print('\n'.join(out))
            allResults.append((t, result))
        return (success, allResults)

class Token(ParserElement):
    """
    Abstract C{ParserElement} subclass, for defining atomic matching patterns.
    """

    def __init__(self):
        if False:
            return 10
        super(Token, self).__init__(savelist=False)

class Empty(Token):
    """
    An empty token, will always match.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Empty, self).__init__()
        self.name = 'Empty'
        self.mayReturnEmpty = True
        self.mayIndexError = False

class NoMatch(Token):
    """
    A token that will never match.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super(NoMatch, self).__init__()
        self.name = 'NoMatch'
        self.mayReturnEmpty = True
        self.mayIndexError = False
        self.errmsg = 'Unmatchable token'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        raise ParseException(instring, loc, self.errmsg, self)

class Literal(Token):
    """
    Token to exactly match a specified string.
    
    Example::
        Literal('blah').parseString('blah')  # -> ['blah']
        Literal('blah').parseString('blahfooblah')  # -> ['blah']
        Literal('blah').parseString('bla')  # -> Exception: Expected "blah"
    
    For case-insensitive matching, use L{CaselessLiteral}.
    
    For keyword matching (force word break before and after the matched string),
    use L{Keyword} or L{CaselessKeyword}.
    """

    def __init__(self, matchString):
        if False:
            print('Hello World!')
        super(Literal, self).__init__()
        self.match = matchString
        self.matchLen = len(matchString)
        try:
            self.firstMatchChar = matchString[0]
        except IndexError:
            warnings.warn('null string passed to Literal; use Empty() instead', SyntaxWarning, stacklevel=2)
            self.__class__ = Empty
        self.name = '"%s"' % _ustr(self.match)
        self.errmsg = 'Expected ' + self.name
        self.mayReturnEmpty = False
        self.mayIndexError = False

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            print('Hello World!')
        if instring[loc] == self.firstMatchChar and (self.matchLen == 1 or instring.startswith(self.match, loc)):
            return (loc + self.matchLen, self.match)
        raise ParseException(instring, loc, self.errmsg, self)
_L = Literal
ParserElement._literalStringClass = Literal

class Keyword(Token):
    """
    Token to exactly match a specified string as a keyword, that is, it must be
    immediately followed by a non-keyword character.  Compare with C{L{Literal}}:
     - C{Literal("if")} will match the leading C{'if'} in C{'ifAndOnlyIf'}.
     - C{Keyword("if")} will not; it will only match the leading C{'if'} in C{'if x=1'}, or C{'if(y==2)'}
    Accepts two optional constructor arguments in addition to the keyword string:
     - C{identChars} is a string of characters that would be valid identifier characters,
          defaulting to all alphanumerics + "_" and "$"
     - C{caseless} allows case-insensitive matching, default is C{False}.
       
    Example::
        Keyword("start").parseString("start")  # -> ['start']
        Keyword("start").parseString("starting")  # -> Exception

    For case-insensitive matching, use L{CaselessKeyword}.
    """
    DEFAULT_KEYWORD_CHARS = alphanums + '_$'

    def __init__(self, matchString, identChars=None, caseless=False):
        if False:
            print('Hello World!')
        super(Keyword, self).__init__()
        if identChars is None:
            identChars = Keyword.DEFAULT_KEYWORD_CHARS
        self.match = matchString
        self.matchLen = len(matchString)
        try:
            self.firstMatchChar = matchString[0]
        except IndexError:
            warnings.warn('null string passed to Keyword; use Empty() instead', SyntaxWarning, stacklevel=2)
        self.name = '"%s"' % self.match
        self.errmsg = 'Expected ' + self.name
        self.mayReturnEmpty = False
        self.mayIndexError = False
        self.caseless = caseless
        if caseless:
            self.caselessmatch = matchString.upper()
            identChars = identChars.upper()
        self.identChars = set(identChars)

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            for i in range(10):
                print('nop')
        if self.caseless:
            if instring[loc:loc + self.matchLen].upper() == self.caselessmatch and (loc >= len(instring) - self.matchLen or instring[loc + self.matchLen].upper() not in self.identChars) and (loc == 0 or instring[loc - 1].upper() not in self.identChars):
                return (loc + self.matchLen, self.match)
        elif instring[loc] == self.firstMatchChar and (self.matchLen == 1 or instring.startswith(self.match, loc)) and (loc >= len(instring) - self.matchLen or instring[loc + self.matchLen] not in self.identChars) and (loc == 0 or instring[loc - 1] not in self.identChars):
            return (loc + self.matchLen, self.match)
        raise ParseException(instring, loc, self.errmsg, self)

    def copy(self):
        if False:
            while True:
                i = 10
        c = super(Keyword, self).copy()
        c.identChars = Keyword.DEFAULT_KEYWORD_CHARS
        return c

    @staticmethod
    def setDefaultKeywordChars(chars):
        if False:
            for i in range(10):
                print('nop')
        'Overrides the default Keyword chars\n        '
        Keyword.DEFAULT_KEYWORD_CHARS = chars

class CaselessLiteral(Literal):
    """
    Token to match a specified string, ignoring case of letters.
    Note: the matched results will always be in the case of the given
    match string, NOT the case of the input text.

    Example::
        OneOrMore(CaselessLiteral("CMD")).parseString("cmd CMD Cmd10") # -> ['CMD', 'CMD', 'CMD']
        
    (Contrast with example for L{CaselessKeyword}.)
    """

    def __init__(self, matchString):
        if False:
            i = 10
            return i + 15
        super(CaselessLiteral, self).__init__(matchString.upper())
        self.returnString = matchString
        self.name = "'%s'" % self.returnString
        self.errmsg = 'Expected ' + self.name

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        if instring[loc:loc + self.matchLen].upper() == self.match:
            return (loc + self.matchLen, self.returnString)
        raise ParseException(instring, loc, self.errmsg, self)

class CaselessKeyword(Keyword):
    """
    Caseless version of L{Keyword}.

    Example::
        OneOrMore(CaselessKeyword("CMD")).parseString("cmd CMD Cmd10") # -> ['CMD', 'CMD']
        
    (Contrast with example for L{CaselessLiteral}.)
    """

    def __init__(self, matchString, identChars=None):
        if False:
            return 10
        super(CaselessKeyword, self).__init__(matchString, identChars, caseless=True)

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        if instring[loc:loc + self.matchLen].upper() == self.caselessmatch and (loc >= len(instring) - self.matchLen or instring[loc + self.matchLen].upper() not in self.identChars):
            return (loc + self.matchLen, self.match)
        raise ParseException(instring, loc, self.errmsg, self)

class CloseMatch(Token):
    """
    A variation on L{Literal} which matches "close" matches, that is, 
    strings with at most 'n' mismatching characters. C{CloseMatch} takes parameters:
     - C{match_string} - string to be matched
     - C{maxMismatches} - (C{default=1}) maximum number of mismatches allowed to count as a match
    
    The results from a successful parse will contain the matched text from the input string and the following named results:
     - C{mismatches} - a list of the positions within the match_string where mismatches were found
     - C{original} - the original match_string used to compare against the input string
    
    If C{mismatches} is an empty list, then the match was an exact match.
    
    Example::
        patt = CloseMatch("ATCATCGAATGGA")
        patt.parseString("ATCATCGAAXGGA") # -> (['ATCATCGAAXGGA'], {'mismatches': [[9]], 'original': ['ATCATCGAATGGA']})
        patt.parseString("ATCAXCGAAXGGA") # -> Exception: Expected 'ATCATCGAATGGA' (with up to 1 mismatches) (at char 0), (line:1, col:1)

        # exact match
        patt.parseString("ATCATCGAATGGA") # -> (['ATCATCGAATGGA'], {'mismatches': [[]], 'original': ['ATCATCGAATGGA']})

        # close match allowing up to 2 mismatches
        patt = CloseMatch("ATCATCGAATGGA", maxMismatches=2)
        patt.parseString("ATCAXCGAAXGGA") # -> (['ATCAXCGAAXGGA'], {'mismatches': [[4, 9]], 'original': ['ATCATCGAATGGA']})
    """

    def __init__(self, match_string, maxMismatches=1):
        if False:
            i = 10
            return i + 15
        super(CloseMatch, self).__init__()
        self.name = match_string
        self.match_string = match_string
        self.maxMismatches = maxMismatches
        self.errmsg = 'Expected %r (with up to %d mismatches)' % (self.match_string, self.maxMismatches)
        self.mayIndexError = False
        self.mayReturnEmpty = False

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        start = loc
        instrlen = len(instring)
        maxloc = start + len(self.match_string)
        if maxloc <= instrlen:
            match_string = self.match_string
            match_stringloc = 0
            mismatches = []
            maxMismatches = self.maxMismatches
            for (match_stringloc, s_m) in enumerate(zip(instring[loc:maxloc], self.match_string)):
                (src, mat) = s_m
                if src != mat:
                    mismatches.append(match_stringloc)
                    if len(mismatches) > maxMismatches:
                        break
            else:
                loc = match_stringloc + 1
                results = ParseResults([instring[start:loc]])
                results['original'] = self.match_string
                results['mismatches'] = mismatches
                return (loc, results)
        raise ParseException(instring, loc, self.errmsg, self)

class Word(Token):
    """
    Token for matching words composed of allowed character sets.
    Defined with string containing all allowed initial characters,
    an optional string containing allowed body characters (if omitted,
    defaults to the initial character set), and an optional minimum,
    maximum, and/or exact length.  The default value for C{min} is 1 (a
    minimum value < 1 is not valid); the default values for C{max} and C{exact}
    are 0, meaning no maximum or exact length restriction. An optional
    C{excludeChars} parameter can list characters that might be found in 
    the input C{bodyChars} string; useful to define a word of all printables
    except for one or two characters, for instance.
    
    L{srange} is useful for defining custom character set strings for defining 
    C{Word} expressions, using range notation from regular expression character sets.
    
    A common mistake is to use C{Word} to match a specific literal string, as in 
    C{Word("Address")}. Remember that C{Word} uses the string argument to define
    I{sets} of matchable characters. This expression would match "Add", "AAA",
    "dAred", or any other word made up of the characters 'A', 'd', 'r', 'e', and 's'.
    To match an exact literal string, use L{Literal} or L{Keyword}.

    pyparsing includes helper strings for building Words:
     - L{alphas}
     - L{nums}
     - L{alphanums}
     - L{hexnums}
     - L{alphas8bit} (alphabetic characters in ASCII range 128-255 - accented, tilded, umlauted, etc.)
     - L{punc8bit} (non-alphabetic characters in ASCII range 128-255 - currency, symbols, superscripts, diacriticals, etc.)
     - L{printables} (any non-whitespace character)

    Example::
        # a word composed of digits
        integer = Word(nums) # equivalent to Word("0123456789") or Word(srange("0-9"))
        
        # a word with a leading capital, and zero or more lowercase
        capital_word = Word(alphas.upper(), alphas.lower())

        # hostnames are alphanumeric, with leading alpha, and '-'
        hostname = Word(alphas, alphanums+'-')
        
        # roman numeral (not a strict parser, accepts invalid mix of characters)
        roman = Word("IVXLCDM")
        
        # any string of non-whitespace characters, except for ','
        csv_value = Word(printables, excludeChars=",")
    """

    def __init__(self, initChars, bodyChars=None, min=1, max=0, exact=0, asKeyword=False, excludeChars=None):
        if False:
            i = 10
            return i + 15
        super(Word, self).__init__()
        if excludeChars:
            initChars = ''.join((c for c in initChars if c not in excludeChars))
            if bodyChars:
                bodyChars = ''.join((c for c in bodyChars if c not in excludeChars))
        self.initCharsOrig = initChars
        self.initChars = set(initChars)
        if bodyChars:
            self.bodyCharsOrig = bodyChars
            self.bodyChars = set(bodyChars)
        else:
            self.bodyCharsOrig = initChars
            self.bodyChars = set(initChars)
        self.maxSpecified = max > 0
        if min < 1:
            raise ValueError('cannot specify a minimum length < 1; use Optional(Word()) if zero-length word is permitted')
        self.minLen = min
        if max > 0:
            self.maxLen = max
        else:
            self.maxLen = _MAX_INT
        if exact > 0:
            self.maxLen = exact
            self.minLen = exact
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayIndexError = False
        self.asKeyword = asKeyword
        if ' ' not in self.initCharsOrig + self.bodyCharsOrig and (min == 1 and max == 0 and (exact == 0)):
            if self.bodyCharsOrig == self.initCharsOrig:
                self.reString = '[%s]+' % _escapeRegexRangeChars(self.initCharsOrig)
            elif len(self.initCharsOrig) == 1:
                self.reString = '%s[%s]*' % (re.escape(self.initCharsOrig), _escapeRegexRangeChars(self.bodyCharsOrig))
            else:
                self.reString = '[%s][%s]*' % (_escapeRegexRangeChars(self.initCharsOrig), _escapeRegexRangeChars(self.bodyCharsOrig))
            if self.asKeyword:
                self.reString = '\\b' + self.reString + '\\b'
            try:
                self.re = re.compile(self.reString)
            except Exception:
                self.re = None

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        if self.re:
            result = self.re.match(instring, loc)
            if not result:
                raise ParseException(instring, loc, self.errmsg, self)
            loc = result.end()
            return (loc, result.group())
        if not instring[loc] in self.initChars:
            raise ParseException(instring, loc, self.errmsg, self)
        start = loc
        loc += 1
        instrlen = len(instring)
        bodychars = self.bodyChars
        maxloc = start + self.maxLen
        maxloc = min(maxloc, instrlen)
        while loc < maxloc and instring[loc] in bodychars:
            loc += 1
        throwException = False
        if loc - start < self.minLen:
            throwException = True
        if self.maxSpecified and loc < instrlen and (instring[loc] in bodychars):
            throwException = True
        if self.asKeyword:
            if start > 0 and instring[start - 1] in bodychars or (loc < instrlen and instring[loc] in bodychars):
                throwException = True
        if throwException:
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, instring[start:loc])

    def __str__(self):
        if False:
            return 10
        try:
            return super(Word, self).__str__()
        except Exception:
            pass
        if self.strRepr is None:

            def charsAsStr(s):
                if False:
                    i = 10
                    return i + 15
                if len(s) > 4:
                    return s[:4] + '...'
                else:
                    return s
            if self.initCharsOrig != self.bodyCharsOrig:
                self.strRepr = 'W:(%s,%s)' % (charsAsStr(self.initCharsOrig), charsAsStr(self.bodyCharsOrig))
            else:
                self.strRepr = 'W:(%s)' % charsAsStr(self.initCharsOrig)
        return self.strRepr

class Regex(Token):
    """
    Token for matching strings that match a given regular expression.
    Defined with string specifying the regular expression in a form recognized by the inbuilt Python re module.
    If the given regex contains named groups (defined using C{(?P<name>...)}), these will be preserved as 
    named parse results.

    Example::
        realnum = Regex(r"[+-]?\\d+\\.\\d*")
        date = Regex(r'(?P<year>\\d{4})-(?P<month>\\d\\d?)-(?P<day>\\d\\d?)')
        # ref: https://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression
        roman = Regex(r"M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
    """
    compiledREtype = type(re.compile('[A-Z]'))

    def __init__(self, pattern, flags=0):
        if False:
            i = 10
            return i + 15
        'The parameters C{pattern} and C{flags} are passed to the C{re.compile()} function as-is. See the Python C{re} module for an explanation of the acceptable patterns and flags.'
        super(Regex, self).__init__()
        if isinstance(pattern, basestring):
            if not pattern:
                warnings.warn('null string passed to Regex; use Empty() instead', SyntaxWarning, stacklevel=2)
            self.pattern = pattern
            self.flags = flags
            try:
                self.re = re.compile(self.pattern, self.flags)
                self.reString = self.pattern
            except sre_constants.error:
                warnings.warn('invalid pattern (%s) passed to Regex' % pattern, SyntaxWarning, stacklevel=2)
                raise
        elif isinstance(pattern, Regex.compiledREtype):
            self.re = pattern
            self.pattern = self.reString = str(pattern)
            self.flags = flags
        else:
            raise ValueError('Regex may only be constructed with a string or a compiled RE object')
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayIndexError = False
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            i = 10
            return i + 15
        result = self.re.match(instring, loc)
        if not result:
            raise ParseException(instring, loc, self.errmsg, self)
        loc = result.end()
        d = result.groupdict()
        ret = ParseResults(result.group())
        if d:
            for k in d:
                ret[k] = d[k]
        return (loc, ret)

    def __str__(self):
        if False:
            while True:
                i = 10
        try:
            return super(Regex, self).__str__()
        except Exception:
            pass
        if self.strRepr is None:
            self.strRepr = 'Re:(%s)' % repr(self.pattern)
        return self.strRepr

class QuotedString(Token):
    """
    Token for matching strings that are delimited by quoting characters.
    
    Defined with the following parameters:
        - quoteChar - string of one or more characters defining the quote delimiting string
        - escChar - character to escape quotes, typically backslash (default=C{None})
        - escQuote - special quote sequence to escape an embedded quote string (such as SQL's "" to escape an embedded ") (default=C{None})
        - multiline - boolean indicating whether quotes can span multiple lines (default=C{False})
        - unquoteResults - boolean indicating whether the matched text should be unquoted (default=C{True})
        - endQuoteChar - string of one or more characters defining the end of the quote delimited string (default=C{None} => same as quoteChar)
        - convertWhitespaceEscapes - convert escaped whitespace (C{'\\t'}, C{'\\n'}, etc.) to actual whitespace (default=C{True})

    Example::
        qs = QuotedString('"')
        print(qs.searchString('lsjdf "This is the quote" sldjf'))
        complex_qs = QuotedString('{{', endQuoteChar='}}')
        print(complex_qs.searchString('lsjdf {{This is the "quote"}} sldjf'))
        sql_qs = QuotedString('"', escQuote='""')
        print(sql_qs.searchString('lsjdf "This is the quote with ""embedded"" quotes" sldjf'))
    prints::
        [['This is the quote']]
        [['This is the "quote"']]
        [['This is the quote with "embedded" quotes']]
    """

    def __init__(self, quoteChar, escChar=None, escQuote=None, multiline=False, unquoteResults=True, endQuoteChar=None, convertWhitespaceEscapes=True):
        if False:
            print('Hello World!')
        super(QuotedString, self).__init__()
        quoteChar = quoteChar.strip()
        if not quoteChar:
            warnings.warn('quoteChar cannot be the empty string', SyntaxWarning, stacklevel=2)
            raise SyntaxError()
        if endQuoteChar is None:
            endQuoteChar = quoteChar
        else:
            endQuoteChar = endQuoteChar.strip()
            if not endQuoteChar:
                warnings.warn('endQuoteChar cannot be the empty string', SyntaxWarning, stacklevel=2)
                raise SyntaxError()
        self.quoteChar = quoteChar
        self.quoteCharLen = len(quoteChar)
        self.firstQuoteChar = quoteChar[0]
        self.endQuoteChar = endQuoteChar
        self.endQuoteCharLen = len(endQuoteChar)
        self.escChar = escChar
        self.escQuote = escQuote
        self.unquoteResults = unquoteResults
        self.convertWhitespaceEscapes = convertWhitespaceEscapes
        if multiline:
            self.flags = re.MULTILINE | re.DOTALL
            self.pattern = '%s(?:[^%s%s]' % (re.escape(self.quoteChar), _escapeRegexRangeChars(self.endQuoteChar[0]), escChar is not None and _escapeRegexRangeChars(escChar) or '')
        else:
            self.flags = 0
            self.pattern = '%s(?:[^%s\\n\\r%s]' % (re.escape(self.quoteChar), _escapeRegexRangeChars(self.endQuoteChar[0]), escChar is not None and _escapeRegexRangeChars(escChar) or '')
        if len(self.endQuoteChar) > 1:
            self.pattern += '|(?:' + ')|(?:'.join(('%s[^%s]' % (re.escape(self.endQuoteChar[:i]), _escapeRegexRangeChars(self.endQuoteChar[i])) for i in range(len(self.endQuoteChar) - 1, 0, -1))) + ')'
        if escQuote:
            self.pattern += '|(?:%s)' % re.escape(escQuote)
        if escChar:
            self.pattern += '|(?:%s.)' % re.escape(escChar)
            self.escCharReplacePattern = re.escape(self.escChar) + '(.)'
        self.pattern += ')*%s' % re.escape(self.endQuoteChar)
        try:
            self.re = re.compile(self.pattern, self.flags)
            self.reString = self.pattern
        except sre_constants.error:
            warnings.warn('invalid pattern (%s) passed to Regex' % self.pattern, SyntaxWarning, stacklevel=2)
            raise
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayIndexError = False
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        result = instring[loc] == self.firstQuoteChar and self.re.match(instring, loc) or None
        if not result:
            raise ParseException(instring, loc, self.errmsg, self)
        loc = result.end()
        ret = result.group()
        if self.unquoteResults:
            ret = ret[self.quoteCharLen:-self.endQuoteCharLen]
            if isinstance(ret, basestring):
                if '\\' in ret and self.convertWhitespaceEscapes:
                    ws_map = {'\\t': '\t', '\\n': '\n', '\\f': '\x0c', '\\r': '\r'}
                    for (wslit, wschar) in ws_map.items():
                        ret = ret.replace(wslit, wschar)
                if self.escChar:
                    ret = re.sub(self.escCharReplacePattern, '\\g<1>', ret)
                if self.escQuote:
                    ret = ret.replace(self.escQuote, self.endQuoteChar)
        return (loc, ret)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return super(QuotedString, self).__str__()
        except Exception:
            pass
        if self.strRepr is None:
            self.strRepr = 'quoted string, starting with %s ending with %s' % (self.quoteChar, self.endQuoteChar)
        return self.strRepr

class CharsNotIn(Token):
    """
    Token for matching words composed of characters I{not} in a given set (will
    include whitespace in matched characters if not listed in the provided exclusion set - see example).
    Defined with string containing all disallowed characters, and an optional
    minimum, maximum, and/or exact length.  The default value for C{min} is 1 (a
    minimum value < 1 is not valid); the default values for C{max} and C{exact}
    are 0, meaning no maximum or exact length restriction.

    Example::
        # define a comma-separated-value as anything that is not a ','
        csv_value = CharsNotIn(',')
        print(delimitedList(csv_value).parseString("dkls,lsdkjf,s12 34,@!#,213"))
    prints::
        ['dkls', 'lsdkjf', 's12 34', '@!#', '213']
    """

    def __init__(self, notChars, min=1, max=0, exact=0):
        if False:
            return 10
        super(CharsNotIn, self).__init__()
        self.skipWhitespace = False
        self.notChars = notChars
        if min < 1:
            raise ValueError('cannot specify a minimum length < 1; use Optional(CharsNotIn()) if zero-length char group is permitted')
        self.minLen = min
        if max > 0:
            self.maxLen = max
        else:
            self.maxLen = _MAX_INT
        if exact > 0:
            self.maxLen = exact
            self.minLen = exact
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayReturnEmpty = self.minLen == 0
        self.mayIndexError = False

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        if instring[loc] in self.notChars:
            raise ParseException(instring, loc, self.errmsg, self)
        start = loc
        loc += 1
        notchars = self.notChars
        maxlen = min(start + self.maxLen, len(instring))
        while loc < maxlen and instring[loc] not in notchars:
            loc += 1
        if loc - start < self.minLen:
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, instring[start:loc])

    def __str__(self):
        if False:
            i = 10
            return i + 15
        try:
            return super(CharsNotIn, self).__str__()
        except Exception:
            pass
        if self.strRepr is None:
            if len(self.notChars) > 4:
                self.strRepr = '!W:(%s...)' % self.notChars[:4]
            else:
                self.strRepr = '!W:(%s)' % self.notChars
        return self.strRepr

class White(Token):
    """
    Special matching class for matching whitespace.  Normally, whitespace is ignored
    by pyparsing grammars.  This class is included when some whitespace structures
    are significant.  Define with a string containing the whitespace characters to be
    matched; default is C{" \\t\\r\\n"}.  Also takes optional C{min}, C{max}, and C{exact} arguments,
    as defined for the C{L{Word}} class.
    """
    whiteStrs = {' ': '<SPC>', '\t': '<TAB>', '\n': '<LF>', '\r': '<CR>', '\x0c': '<FF>'}

    def __init__(self, ws=' \t\r\n', min=1, max=0, exact=0):
        if False:
            i = 10
            return i + 15
        super(White, self).__init__()
        self.matchWhite = ws
        self.setWhitespaceChars(''.join((c for c in self.whiteChars if c not in self.matchWhite)))
        self.name = ''.join((White.whiteStrs[c] for c in self.matchWhite))
        self.mayReturnEmpty = True
        self.errmsg = 'Expected ' + self.name
        self.minLen = min
        if max > 0:
            self.maxLen = max
        else:
            self.maxLen = _MAX_INT
        if exact > 0:
            self.maxLen = exact
            self.minLen = exact

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        if not instring[loc] in self.matchWhite:
            raise ParseException(instring, loc, self.errmsg, self)
        start = loc
        loc += 1
        maxloc = start + self.maxLen
        maxloc = min(maxloc, len(instring))
        while loc < maxloc and instring[loc] in self.matchWhite:
            loc += 1
        if loc - start < self.minLen:
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, instring[start:loc])

class _PositionToken(Token):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(_PositionToken, self).__init__()
        self.name = self.__class__.__name__
        self.mayReturnEmpty = True
        self.mayIndexError = False

class GoToColumn(_PositionToken):
    """
    Token to advance to a specific column of input text; useful for tabular report scraping.
    """

    def __init__(self, colno):
        if False:
            print('Hello World!')
        super(GoToColumn, self).__init__()
        self.col = colno

    def preParse(self, instring, loc):
        if False:
            print('Hello World!')
        if col(loc, instring) != self.col:
            instrlen = len(instring)
            if self.ignoreExprs:
                loc = self._skipIgnorables(instring, loc)
            while loc < instrlen and instring[loc].isspace() and (col(loc, instring) != self.col):
                loc += 1
        return loc

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        thiscol = col(loc, instring)
        if thiscol > self.col:
            raise ParseException(instring, loc, 'Text not in expected column', self)
        newloc = loc + self.col - thiscol
        ret = instring[loc:newloc]
        return (newloc, ret)

class LineStart(_PositionToken):
    """
    Matches if current position is at the beginning of a line within the parse string
    
    Example::
    
        test = '''        AAA this line
        AAA and this line
          AAA but not this one
        B AAA and definitely not this one
        '''

        for t in (LineStart() + 'AAA' + restOfLine).searchString(test):
            print(t)
    
    Prints::
        ['AAA', ' this line']
        ['AAA', ' and this line']    

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(LineStart, self).__init__()
        self.errmsg = 'Expected start of line'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        if col(loc, instring) == 1:
            return (loc, [])
        raise ParseException(instring, loc, self.errmsg, self)

class LineEnd(_PositionToken):
    """
    Matches if current position is at the end of a line within the parse string
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(LineEnd, self).__init__()
        self.setWhitespaceChars(ParserElement.DEFAULT_WHITE_CHARS.replace('\n', ''))
        self.errmsg = 'Expected end of line'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        if loc < len(instring):
            if instring[loc] == '\n':
                return (loc + 1, '\n')
            else:
                raise ParseException(instring, loc, self.errmsg, self)
        elif loc == len(instring):
            return (loc + 1, [])
        else:
            raise ParseException(instring, loc, self.errmsg, self)

class StringStart(_PositionToken):
    """
    Matches if current position is at the beginning of the parse string
    """

    def __init__(self):
        if False:
            return 10
        super(StringStart, self).__init__()
        self.errmsg = 'Expected start of text'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            print('Hello World!')
        if loc != 0:
            if loc != self.preParse(instring, 0):
                raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])

class StringEnd(_PositionToken):
    """
    Matches if current position is at the end of the parse string
    """

    def __init__(self):
        if False:
            return 10
        super(StringEnd, self).__init__()
        self.errmsg = 'Expected end of text'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        if loc < len(instring):
            raise ParseException(instring, loc, self.errmsg, self)
        elif loc == len(instring):
            return (loc + 1, [])
        elif loc > len(instring):
            return (loc, [])
        else:
            raise ParseException(instring, loc, self.errmsg, self)

class WordStart(_PositionToken):
    """
    Matches if the current position is at the beginning of a Word, and
    is not preceded by any character in a given set of C{wordChars}
    (default=C{printables}). To emulate the C{\x08} behavior of regular expressions,
    use C{WordStart(alphanums)}. C{WordStart} will also match at the beginning of
    the string being parsed, or at the beginning of a line.
    """

    def __init__(self, wordChars=printables):
        if False:
            while True:
                i = 10
        super(WordStart, self).__init__()
        self.wordChars = set(wordChars)
        self.errmsg = 'Not at the start of a word'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            for i in range(10):
                print('nop')
        if loc != 0:
            if instring[loc - 1] in self.wordChars or instring[loc] not in self.wordChars:
                raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])

class WordEnd(_PositionToken):
    """
    Matches if the current position is at the end of a Word, and
    is not followed by any character in a given set of C{wordChars}
    (default=C{printables}). To emulate the C{\x08} behavior of regular expressions,
    use C{WordEnd(alphanums)}. C{WordEnd} will also match at the end of
    the string being parsed, or at the end of a line.
    """

    def __init__(self, wordChars=printables):
        if False:
            for i in range(10):
                print('nop')
        super(WordEnd, self).__init__()
        self.wordChars = set(wordChars)
        self.skipWhitespace = False
        self.errmsg = 'Not at the end of a word'

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            for i in range(10):
                print('nop')
        instrlen = len(instring)
        if instrlen > 0 and loc < instrlen:
            if instring[loc] in self.wordChars or instring[loc - 1] not in self.wordChars:
                raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])

class ParseExpression(ParserElement):
    """
    Abstract subclass of ParserElement, for combining and post-processing parsed tokens.
    """

    def __init__(self, exprs, savelist=False):
        if False:
            i = 10
            return i + 15
        super(ParseExpression, self).__init__(savelist)
        if isinstance(exprs, _generatorType):
            exprs = list(exprs)
        if isinstance(exprs, basestring):
            self.exprs = [ParserElement._literalStringClass(exprs)]
        elif isinstance(exprs, collections.Iterable):
            exprs = list(exprs)
            if all((isinstance(expr, basestring) for expr in exprs)):
                exprs = map(ParserElement._literalStringClass, exprs)
            self.exprs = list(exprs)
        else:
            try:
                self.exprs = list(exprs)
            except TypeError:
                self.exprs = [exprs]
        self.callPreparse = False

    def __getitem__(self, i):
        if False:
            while True:
                i = 10
        return self.exprs[i]

    def append(self, other):
        if False:
            for i in range(10):
                print('nop')
        self.exprs.append(other)
        self.strRepr = None
        return self

    def leaveWhitespace(self):
        if False:
            return 10
        'Extends C{leaveWhitespace} defined in base class, and also invokes C{leaveWhitespace} on\n           all contained expressions.'
        self.skipWhitespace = False
        self.exprs = [e.copy() for e in self.exprs]
        for e in self.exprs:
            e.leaveWhitespace()
        return self

    def ignore(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, Suppress):
            if other not in self.ignoreExprs:
                super(ParseExpression, self).ignore(other)
                for e in self.exprs:
                    e.ignore(self.ignoreExprs[-1])
        else:
            super(ParseExpression, self).ignore(other)
            for e in self.exprs:
                e.ignore(self.ignoreExprs[-1])
        return self

    def __str__(self):
        if False:
            print('Hello World!')
        try:
            return super(ParseExpression, self).__str__()
        except Exception:
            pass
        if self.strRepr is None:
            self.strRepr = '%s:(%s)' % (self.__class__.__name__, _ustr(self.exprs))
        return self.strRepr

    def streamline(self):
        if False:
            print('Hello World!')
        super(ParseExpression, self).streamline()
        for e in self.exprs:
            e.streamline()
        if len(self.exprs) == 2:
            other = self.exprs[0]
            if isinstance(other, self.__class__) and (not other.parseAction) and (other.resultsName is None) and (not other.debug):
                self.exprs = other.exprs[:] + [self.exprs[1]]
                self.strRepr = None
                self.mayReturnEmpty |= other.mayReturnEmpty
                self.mayIndexError |= other.mayIndexError
            other = self.exprs[-1]
            if isinstance(other, self.__class__) and (not other.parseAction) and (other.resultsName is None) and (not other.debug):
                self.exprs = self.exprs[:-1] + other.exprs[:]
                self.strRepr = None
                self.mayReturnEmpty |= other.mayReturnEmpty
                self.mayIndexError |= other.mayIndexError
        self.errmsg = 'Expected ' + _ustr(self)
        return self

    def setResultsName(self, name, listAllMatches=False):
        if False:
            for i in range(10):
                print('nop')
        ret = super(ParseExpression, self).setResultsName(name, listAllMatches)
        return ret

    def validate(self, validateTrace=[]):
        if False:
            print('Hello World!')
        tmp = validateTrace[:] + [self]
        for e in self.exprs:
            e.validate(tmp)
        self.checkRecursion([])

    def copy(self):
        if False:
            while True:
                i = 10
        ret = super(ParseExpression, self).copy()
        ret.exprs = [e.copy() for e in self.exprs]
        return ret

class And(ParseExpression):
    """
    Requires all given C{ParseExpression}s to be found in the given order.
    Expressions may be separated by whitespace.
    May be constructed using the C{'+'} operator.
    May also be constructed using the C{'-'} operator, which will suppress backtracking.

    Example::
        integer = Word(nums)
        name_expr = OneOrMore(Word(alphas))

        expr = And([integer("id"),name_expr("name"),integer("age")])
        # more easily written as:
        expr = integer("id") + name_expr("name") + integer("age")
    """

    class _ErrorStop(Empty):

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super(And._ErrorStop, self).__init__(*args, **kwargs)
            self.name = '-'
            self.leaveWhitespace()

    def __init__(self, exprs, savelist=True):
        if False:
            print('Hello World!')
        super(And, self).__init__(exprs, savelist)
        self.mayReturnEmpty = all((e.mayReturnEmpty for e in self.exprs))
        self.setWhitespaceChars(self.exprs[0].whiteChars)
        self.skipWhitespace = self.exprs[0].skipWhitespace
        self.callPreparse = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            i = 10
            return i + 15
        (loc, resultlist) = self.exprs[0]._parse(instring, loc, doActions, callPreParse=False)
        errorStop = False
        for e in self.exprs[1:]:
            if isinstance(e, And._ErrorStop):
                errorStop = True
                continue
            if errorStop:
                try:
                    (loc, exprtokens) = e._parse(instring, loc, doActions)
                except ParseSyntaxException:
                    raise
                except ParseBaseException as pe:
                    pe.__traceback__ = None
                    raise ParseSyntaxException._from_exception(pe)
                except IndexError:
                    raise ParseSyntaxException(instring, len(instring), self.errmsg, self)
            else:
                (loc, exprtokens) = e._parse(instring, loc, doActions)
            if exprtokens or exprtokens.haskeys():
                resultlist += exprtokens
        return (loc, resultlist)

    def __iadd__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        return self.append(other)

    def checkRecursion(self, parseElementList):
        if False:
            return 10
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)
            if not e.mayReturnEmpty:
                break

    def __str__(self):
        if False:
            return 10
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr

class Or(ParseExpression):
    """
    Requires that at least one C{ParseExpression} is found.
    If two expressions match, the expression that matches the longest string will be used.
    May be constructed using the C{'^'} operator.

    Example::
        # construct Or using '^' operator
        
        number = Word(nums) ^ Combine(Word(nums) + '.' + Word(nums))
        print(number.searchString("123 3.1416 789"))
    prints::
        [['123'], ['3.1416'], ['789']]
    """

    def __init__(self, exprs, savelist=False):
        if False:
            i = 10
            return i + 15
        super(Or, self).__init__(exprs, savelist)
        if self.exprs:
            self.mayReturnEmpty = any((e.mayReturnEmpty for e in self.exprs))
        else:
            self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            print('Hello World!')
        maxExcLoc = -1
        maxException = None
        matches = []
        for e in self.exprs:
            try:
                loc2 = e.tryParse(instring, loc)
            except ParseException as err:
                err.__traceback__ = None
                if err.loc > maxExcLoc:
                    maxException = err
                    maxExcLoc = err.loc
            except IndexError:
                if len(instring) > maxExcLoc:
                    maxException = ParseException(instring, len(instring), e.errmsg, self)
                    maxExcLoc = len(instring)
            else:
                matches.append((loc2, e))
        if matches:
            matches.sort(key=lambda x: -x[0])
            for (_, e) in matches:
                try:
                    return e._parse(instring, loc, doActions)
                except ParseException as err:
                    err.__traceback__ = None
                    if err.loc > maxExcLoc:
                        maxException = err
                        maxExcLoc = err.loc
        if maxException is not None:
            maxException.msg = self.errmsg
            raise maxException
        else:
            raise ParseException(instring, loc, 'no defined alternatives to match', self)

    def __ixor__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        return self.append(other)

    def __str__(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' ^ '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr

    def checkRecursion(self, parseElementList):
        if False:
            return 10
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)

class MatchFirst(ParseExpression):
    """
    Requires that at least one C{ParseExpression} is found.
    If two expressions match, the first one listed is the one that will match.
    May be constructed using the C{'|'} operator.

    Example::
        # construct MatchFirst using '|' operator
        
        # watch the order of expressions to match
        number = Word(nums) | Combine(Word(nums) + '.' + Word(nums))
        print(number.searchString("123 3.1416 789")) #  Fail! -> [['123'], ['3'], ['1416'], ['789']]

        # put more selective expression first
        number = Combine(Word(nums) + '.' + Word(nums)) | Word(nums)
        print(number.searchString("123 3.1416 789")) #  Better -> [['123'], ['3.1416'], ['789']]
    """

    def __init__(self, exprs, savelist=False):
        if False:
            for i in range(10):
                print('nop')
        super(MatchFirst, self).__init__(exprs, savelist)
        if self.exprs:
            self.mayReturnEmpty = any((e.mayReturnEmpty for e in self.exprs))
        else:
            self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            print('Hello World!')
        maxExcLoc = -1
        maxException = None
        for e in self.exprs:
            try:
                ret = e._parse(instring, loc, doActions)
                return ret
            except ParseException as err:
                if err.loc > maxExcLoc:
                    maxException = err
                    maxExcLoc = err.loc
            except IndexError:
                if len(instring) > maxExcLoc:
                    maxException = ParseException(instring, len(instring), e.errmsg, self)
                    maxExcLoc = len(instring)
        else:
            if maxException is not None:
                maxException.msg = self.errmsg
                raise maxException
            else:
                raise ParseException(instring, loc, 'no defined alternatives to match', self)

    def __ior__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        return self.append(other)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' | '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr

    def checkRecursion(self, parseElementList):
        if False:
            for i in range(10):
                print('nop')
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)

class Each(ParseExpression):
    """
    Requires all given C{ParseExpression}s to be found, but in any order.
    Expressions may be separated by whitespace.
    May be constructed using the C{'&'} operator.

    Example::
        color = oneOf("RED ORANGE YELLOW GREEN BLUE PURPLE BLACK WHITE BROWN")
        shape_type = oneOf("SQUARE CIRCLE TRIANGLE STAR HEXAGON OCTAGON")
        integer = Word(nums)
        shape_attr = "shape:" + shape_type("shape")
        posn_attr = "posn:" + Group(integer("x") + ',' + integer("y"))("posn")
        color_attr = "color:" + color("color")
        size_attr = "size:" + integer("size")

        # use Each (using operator '&') to accept attributes in any order 
        # (shape and posn are required, color and size are optional)
        shape_spec = shape_attr & posn_attr & Optional(color_attr) & Optional(size_attr)

        shape_spec.runTests('''
            shape: SQUARE color: BLACK posn: 100, 120
            shape: CIRCLE size: 50 color: BLUE posn: 50,80
            color:GREEN size:20 shape:TRIANGLE posn:20,40
            '''
            )
    prints::
        shape: SQUARE color: BLACK posn: 100, 120
        ['shape:', 'SQUARE', 'color:', 'BLACK', 'posn:', ['100', ',', '120']]
        - color: BLACK
        - posn: ['100', ',', '120']
          - x: 100
          - y: 120
        - shape: SQUARE


        shape: CIRCLE size: 50 color: BLUE posn: 50,80
        ['shape:', 'CIRCLE', 'size:', '50', 'color:', 'BLUE', 'posn:', ['50', ',', '80']]
        - color: BLUE
        - posn: ['50', ',', '80']
          - x: 50
          - y: 80
        - shape: CIRCLE
        - size: 50


        color: GREEN size: 20 shape: TRIANGLE posn: 20,40
        ['color:', 'GREEN', 'size:', '20', 'shape:', 'TRIANGLE', 'posn:', ['20', ',', '40']]
        - color: GREEN
        - posn: ['20', ',', '40']
          - x: 20
          - y: 40
        - shape: TRIANGLE
        - size: 20
    """

    def __init__(self, exprs, savelist=True):
        if False:
            for i in range(10):
                print('nop')
        super(Each, self).__init__(exprs, savelist)
        self.mayReturnEmpty = all((e.mayReturnEmpty for e in self.exprs))
        self.skipWhitespace = True
        self.initExprGroups = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            i = 10
            return i + 15
        if self.initExprGroups:
            self.opt1map = dict(((id(e.expr), e) for e in self.exprs if isinstance(e, Optional)))
            opt1 = [e.expr for e in self.exprs if isinstance(e, Optional)]
            opt2 = [e for e in self.exprs if e.mayReturnEmpty and (not isinstance(e, Optional))]
            self.optionals = opt1 + opt2
            self.multioptionals = [e.expr for e in self.exprs if isinstance(e, ZeroOrMore)]
            self.multirequired = [e.expr for e in self.exprs if isinstance(e, OneOrMore)]
            self.required = [e for e in self.exprs if not isinstance(e, (Optional, ZeroOrMore, OneOrMore))]
            self.required += self.multirequired
            self.initExprGroups = False
        tmpLoc = loc
        tmpReqd = self.required[:]
        tmpOpt = self.optionals[:]
        matchOrder = []
        keepMatching = True
        while keepMatching:
            tmpExprs = tmpReqd + tmpOpt + self.multioptionals + self.multirequired
            failed = []
            for e in tmpExprs:
                try:
                    tmpLoc = e.tryParse(instring, tmpLoc)
                except ParseException:
                    failed.append(e)
                else:
                    matchOrder.append(self.opt1map.get(id(e), e))
                    if e in tmpReqd:
                        tmpReqd.remove(e)
                    elif e in tmpOpt:
                        tmpOpt.remove(e)
            if len(failed) == len(tmpExprs):
                keepMatching = False
        if tmpReqd:
            missing = ', '.join((_ustr(e) for e in tmpReqd))
            raise ParseException(instring, loc, 'Missing one or more required elements (%s)' % missing)
        matchOrder += [e for e in self.exprs if isinstance(e, Optional) and e.expr in tmpOpt]
        resultlist = []
        for e in matchOrder:
            (loc, results) = e._parse(instring, loc, doActions)
            resultlist.append(results)
        finalResults = sum(resultlist, ParseResults([]))
        return (loc, finalResults)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' & '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr

    def checkRecursion(self, parseElementList):
        if False:
            i = 10
            return i + 15
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)

class ParseElementEnhance(ParserElement):
    """
    Abstract subclass of C{ParserElement}, for combining and post-processing parsed tokens.
    """

    def __init__(self, expr, savelist=False):
        if False:
            for i in range(10):
                print('nop')
        super(ParseElementEnhance, self).__init__(savelist)
        if isinstance(expr, basestring):
            if issubclass(ParserElement._literalStringClass, Token):
                expr = ParserElement._literalStringClass(expr)
            else:
                expr = ParserElement._literalStringClass(Literal(expr))
        self.expr = expr
        self.strRepr = None
        if expr is not None:
            self.mayIndexError = expr.mayIndexError
            self.mayReturnEmpty = expr.mayReturnEmpty
            self.setWhitespaceChars(expr.whiteChars)
            self.skipWhitespace = expr.skipWhitespace
            self.saveAsList = expr.saveAsList
            self.callPreparse = expr.callPreparse
            self.ignoreExprs.extend(expr.ignoreExprs)

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            i = 10
            return i + 15
        if self.expr is not None:
            return self.expr._parse(instring, loc, doActions, callPreParse=False)
        else:
            raise ParseException('', loc, self.errmsg, self)

    def leaveWhitespace(self):
        if False:
            i = 10
            return i + 15
        self.skipWhitespace = False
        self.expr = self.expr.copy()
        if self.expr is not None:
            self.expr.leaveWhitespace()
        return self

    def ignore(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, Suppress):
            if other not in self.ignoreExprs:
                super(ParseElementEnhance, self).ignore(other)
                if self.expr is not None:
                    self.expr.ignore(self.ignoreExprs[-1])
        else:
            super(ParseElementEnhance, self).ignore(other)
            if self.expr is not None:
                self.expr.ignore(self.ignoreExprs[-1])
        return self

    def streamline(self):
        if False:
            for i in range(10):
                print('nop')
        super(ParseElementEnhance, self).streamline()
        if self.expr is not None:
            self.expr.streamline()
        return self

    def checkRecursion(self, parseElementList):
        if False:
            while True:
                i = 10
        if self in parseElementList:
            raise RecursiveGrammarException(parseElementList + [self])
        subRecCheckList = parseElementList[:] + [self]
        if self.expr is not None:
            self.expr.checkRecursion(subRecCheckList)

    def validate(self, validateTrace=[]):
        if False:
            while True:
                i = 10
        tmp = validateTrace[:] + [self]
        if self.expr is not None:
            self.expr.validate(tmp)
        self.checkRecursion([])

    def __str__(self):
        if False:
            print('Hello World!')
        try:
            return super(ParseElementEnhance, self).__str__()
        except Exception:
            pass
        if self.strRepr is None and self.expr is not None:
            self.strRepr = '%s:(%s)' % (self.__class__.__name__, _ustr(self.expr))
        return self.strRepr

class FollowedBy(ParseElementEnhance):
    """
    Lookahead matching of the given parse expression.  C{FollowedBy}
    does I{not} advance the parsing position within the input string, it only
    verifies that the specified parse expression matches at the current
    position.  C{FollowedBy} always returns a null token list.

    Example::
        # use FollowedBy to match a label only if it is followed by a ':'
        data_word = Word(alphas)
        label = data_word + FollowedBy(':')
        attr_expr = Group(label + Suppress(':') + OneOrMore(data_word, stopOn=label).setParseAction(' '.join))
        
        OneOrMore(attr_expr).parseString("shape: SQUARE color: BLACK posn: upper left").pprint()
    prints::
        [['shape', 'SQUARE'], ['color', 'BLACK'], ['posn', 'upper left']]
    """

    def __init__(self, expr):
        if False:
            for i in range(10):
                print('nop')
        super(FollowedBy, self).__init__(expr)
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            while True:
                i = 10
        self.expr.tryParse(instring, loc)
        return (loc, [])

class NotAny(ParseElementEnhance):
    """
    Lookahead to disallow matching with the given parse expression.  C{NotAny}
    does I{not} advance the parsing position within the input string, it only
    verifies that the specified parse expression does I{not} match at the current
    position.  Also, C{NotAny} does I{not} skip over leading whitespace. C{NotAny}
    always returns a null token list.  May be constructed using the '~' operator.

    Example::
        
    """

    def __init__(self, expr):
        if False:
            print('Hello World!')
        super(NotAny, self).__init__(expr)
        self.skipWhitespace = False
        self.mayReturnEmpty = True
        self.errmsg = 'Found unwanted token, ' + _ustr(self.expr)

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            for i in range(10):
                print('nop')
        if self.expr.canParseNext(instring, loc):
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])

    def __str__(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '~{' + _ustr(self.expr) + '}'
        return self.strRepr

class _MultipleMatch(ParseElementEnhance):

    def __init__(self, expr, stopOn=None):
        if False:
            while True:
                i = 10
        super(_MultipleMatch, self).__init__(expr)
        self.saveAsList = True
        ender = stopOn
        if isinstance(ender, basestring):
            ender = ParserElement._literalStringClass(ender)
        self.not_ender = ~ender if ender is not None else None

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        self_expr_parse = self.expr._parse
        self_skip_ignorables = self._skipIgnorables
        check_ender = self.not_ender is not None
        if check_ender:
            try_not_ender = self.not_ender.tryParse
        if check_ender:
            try_not_ender(instring, loc)
        (loc, tokens) = self_expr_parse(instring, loc, doActions, callPreParse=False)
        try:
            hasIgnoreExprs = bool(self.ignoreExprs)
            while 1:
                if check_ender:
                    try_not_ender(instring, loc)
                if hasIgnoreExprs:
                    preloc = self_skip_ignorables(instring, loc)
                else:
                    preloc = loc
                (loc, tmptokens) = self_expr_parse(instring, preloc, doActions)
                if tmptokens or tmptokens.haskeys():
                    tokens += tmptokens
        except (ParseException, IndexError):
            pass
        return (loc, tokens)

class OneOrMore(_MultipleMatch):
    """
    Repetition of one or more of the given expression.
    
    Parameters:
     - expr - expression that must match one or more times
     - stopOn - (default=C{None}) - expression for a terminating sentinel
          (only required if the sentinel would ordinarily match the repetition 
          expression)          

    Example::
        data_word = Word(alphas)
        label = data_word + FollowedBy(':')
        attr_expr = Group(label + Suppress(':') + OneOrMore(data_word).setParseAction(' '.join))

        text = "shape: SQUARE posn: upper left color: BLACK"
        OneOrMore(attr_expr).parseString(text).pprint()  # Fail! read 'color' as data instead of next label -> [['shape', 'SQUARE color']]

        # use stopOn attribute for OneOrMore to avoid reading label string as part of the data
        attr_expr = Group(label + Suppress(':') + OneOrMore(data_word, stopOn=label).setParseAction(' '.join))
        OneOrMore(attr_expr).parseString(text).pprint() # Better -> [['shape', 'SQUARE'], ['posn', 'upper left'], ['color', 'BLACK']]
        
        # could also be written as
        (attr_expr * (1,)).parseString(text).pprint()
    """

    def __str__(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + _ustr(self.expr) + '}...'
        return self.strRepr

class ZeroOrMore(_MultipleMatch):
    """
    Optional repetition of zero or more of the given expression.
    
    Parameters:
     - expr - expression that must match zero or more times
     - stopOn - (default=C{None}) - expression for a terminating sentinel
          (only required if the sentinel would ordinarily match the repetition 
          expression)          

    Example: similar to L{OneOrMore}
    """

    def __init__(self, expr, stopOn=None):
        if False:
            for i in range(10):
                print('nop')
        super(ZeroOrMore, self).__init__(expr, stopOn=stopOn)
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            return 10
        try:
            return super(ZeroOrMore, self).parseImpl(instring, loc, doActions)
        except (ParseException, IndexError):
            return (loc, [])

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '[' + _ustr(self.expr) + ']...'
        return self.strRepr

class _NullToken(object):

    def __bool__(self):
        if False:
            return 10
        return False
    __nonzero__ = __bool__

    def __str__(self):
        if False:
            while True:
                i = 10
        return ''
_optionalNotMatched = _NullToken()

class Optional(ParseElementEnhance):
    """
    Optional matching of the given expression.

    Parameters:
     - expr - expression that must match zero or more times
     - default (optional) - value to be returned if the optional expression is not found.

    Example::
        # US postal code can be a 5-digit zip, plus optional 4-digit qualifier
        zip = Combine(Word(nums, exact=5) + Optional('-' + Word(nums, exact=4)))
        zip.runTests('''
            # traditional ZIP code
            12345
            
            # ZIP+4 form
            12101-0001
            
            # invalid ZIP
            98765-
            ''')
    prints::
        # traditional ZIP code
        12345
        ['12345']

        # ZIP+4 form
        12101-0001
        ['12101-0001']

        # invalid ZIP
        98765-
             ^
        FAIL: Expected end of text (at char 5), (line:1, col:6)
    """

    def __init__(self, expr, default=_optionalNotMatched):
        if False:
            i = 10
            return i + 15
        super(Optional, self).__init__(expr, savelist=False)
        self.saveAsList = self.expr.saveAsList
        self.defaultValue = default
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            i = 10
            return i + 15
        try:
            (loc, tokens) = self.expr._parse(instring, loc, doActions, callPreParse=False)
        except (ParseException, IndexError):
            if self.defaultValue is not _optionalNotMatched:
                if self.expr.resultsName:
                    tokens = ParseResults([self.defaultValue])
                    tokens[self.expr.resultsName] = self.defaultValue
                else:
                    tokens = [self.defaultValue]
            else:
                tokens = []
        return (loc, tokens)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '[' + _ustr(self.expr) + ']'
        return self.strRepr

class SkipTo(ParseElementEnhance):
    """
    Token for skipping over all undefined text until the matched expression is found.

    Parameters:
     - expr - target expression marking the end of the data to be skipped
     - include - (default=C{False}) if True, the target expression is also parsed 
          (the skipped text and target expression are returned as a 2-element list).
     - ignore - (default=C{None}) used to define grammars (typically quoted strings and 
          comments) that might contain false matches to the target expression
     - failOn - (default=C{None}) define expressions that are not allowed to be 
          included in the skipped test; if found before the target expression is found, 
          the SkipTo is not a match

    Example::
        report = '''
            Outstanding Issues Report - 1 Jan 2000

               # | Severity | Description                               |  Days Open
            -----+----------+-------------------------------------------+-----------
             101 | Critical | Intermittent system crash                 |          6
              94 | Cosmetic | Spelling error on Login ('log|n')         |         14
              79 | Minor    | System slow when running too many reports |         47
            '''
        integer = Word(nums)
        SEP = Suppress('|')
        # use SkipTo to simply match everything up until the next SEP
        # - ignore quoted strings, so that a '|' character inside a quoted string does not match
        # - parse action will call token.strip() for each matched token, i.e., the description body
        string_data = SkipTo(SEP, ignore=quotedString)
        string_data.setParseAction(tokenMap(str.strip))
        ticket_expr = (integer("issue_num") + SEP 
                      + string_data("sev") + SEP 
                      + string_data("desc") + SEP 
                      + integer("days_open"))
        
        for tkt in ticket_expr.searchString(report):
            print tkt.dump()
    prints::
        ['101', 'Critical', 'Intermittent system crash', '6']
        - days_open: 6
        - desc: Intermittent system crash
        - issue_num: 101
        - sev: Critical
        ['94', 'Cosmetic', "Spelling error on Login ('log|n')", '14']
        - days_open: 14
        - desc: Spelling error on Login ('log|n')
        - issue_num: 94
        - sev: Cosmetic
        ['79', 'Minor', 'System slow when running too many reports', '47']
        - days_open: 47
        - desc: System slow when running too many reports
        - issue_num: 79
        - sev: Minor
    """

    def __init__(self, other, include=False, ignore=None, failOn=None):
        if False:
            return 10
        super(SkipTo, self).__init__(other)
        self.ignoreExpr = ignore
        self.mayReturnEmpty = True
        self.mayIndexError = False
        self.includeMatch = include
        self.asList = False
        if isinstance(failOn, basestring):
            self.failOn = ParserElement._literalStringClass(failOn)
        else:
            self.failOn = failOn
        self.errmsg = 'No match found for ' + _ustr(self.expr)

    def parseImpl(self, instring, loc, doActions=True):
        if False:
            i = 10
            return i + 15
        startloc = loc
        instrlen = len(instring)
        expr = self.expr
        expr_parse = self.expr._parse
        self_failOn_canParseNext = self.failOn.canParseNext if self.failOn is not None else None
        self_ignoreExpr_tryParse = self.ignoreExpr.tryParse if self.ignoreExpr is not None else None
        tmploc = loc
        while tmploc <= instrlen:
            if self_failOn_canParseNext is not None:
                if self_failOn_canParseNext(instring, tmploc):
                    break
            if self_ignoreExpr_tryParse is not None:
                while 1:
                    try:
                        tmploc = self_ignoreExpr_tryParse(instring, tmploc)
                    except ParseBaseException:
                        break
            try:
                expr_parse(instring, tmploc, doActions=False, callPreParse=False)
            except (ParseException, IndexError):
                tmploc += 1
            else:
                break
        else:
            raise ParseException(instring, loc, self.errmsg, self)
        loc = tmploc
        skiptext = instring[startloc:loc]
        skipresult = ParseResults(skiptext)
        if self.includeMatch:
            (loc, mat) = expr_parse(instring, loc, doActions, callPreParse=False)
            skipresult += mat
        return (loc, skipresult)

class Forward(ParseElementEnhance):
    """
    Forward declaration of an expression to be defined later -
    used for recursive grammars, such as algebraic infix notation.
    When the expression is known, it is assigned to the C{Forward} variable using the '<<' operator.

    Note: take care when assigning to C{Forward} not to overlook precedence of operators.
    Specifically, '|' has a lower precedence than '<<', so that::
        fwdExpr << a | b | c
    will actually be evaluated as::
        (fwdExpr << a) | b | c
    thereby leaving b and c out as parseable alternatives.  It is recommended that you
    explicitly group the values inserted into the C{Forward}::
        fwdExpr << (a | b | c)
    Converting to use the '<<=' operator instead will avoid this problem.

    See L{ParseResults.pprint} for an example of a recursive parser created using
    C{Forward}.
    """

    def __init__(self, other=None):
        if False:
            for i in range(10):
                print('nop')
        super(Forward, self).__init__(other, savelist=False)

    def __lshift__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, basestring):
            other = ParserElement._literalStringClass(other)
        self.expr = other
        self.strRepr = None
        self.mayIndexError = self.expr.mayIndexError
        self.mayReturnEmpty = self.expr.mayReturnEmpty
        self.setWhitespaceChars(self.expr.whiteChars)
        self.skipWhitespace = self.expr.skipWhitespace
        self.saveAsList = self.expr.saveAsList
        self.ignoreExprs.extend(self.expr.ignoreExprs)
        return self

    def __ilshift__(self, other):
        if False:
            while True:
                i = 10
        return self << other

    def leaveWhitespace(self):
        if False:
            i = 10
            return i + 15
        self.skipWhitespace = False
        return self

    def streamline(self):
        if False:
            i = 10
            return i + 15
        if not self.streamlined:
            self.streamlined = True
            if self.expr is not None:
                self.expr.streamline()
        return self

    def validate(self, validateTrace=[]):
        if False:
            for i in range(10):
                print('nop')
        if self not in validateTrace:
            tmp = validateTrace[:] + [self]
            if self.expr is not None:
                self.expr.validate(tmp)
        self.checkRecursion([])

    def __str__(self):
        if False:
            return 10
        if hasattr(self, 'name'):
            return self.name
        return self.__class__.__name__ + ': ...'
        self._revertClass = self.__class__
        self.__class__ = _ForwardNoRecurse
        try:
            if self.expr is not None:
                retString = _ustr(self.expr)
            else:
                retString = 'None'
        finally:
            self.__class__ = self._revertClass
        return self.__class__.__name__ + ': ' + retString

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        if self.expr is not None:
            return super(Forward, self).copy()
        else:
            ret = Forward()
            ret <<= self
            return ret

class _ForwardNoRecurse(Forward):

    def __str__(self):
        if False:
            print('Hello World!')
        return '...'

class TokenConverter(ParseElementEnhance):
    """
    Abstract subclass of C{ParseExpression}, for converting parsed results.
    """

    def __init__(self, expr, savelist=False):
        if False:
            return 10
        super(TokenConverter, self).__init__(expr)
        self.saveAsList = False

class Combine(TokenConverter):
    """
    Converter to concatenate all matching tokens to a single string.
    By default, the matching patterns must also be contiguous in the input string;
    this can be disabled by specifying C{'adjacent=False'} in the constructor.

    Example::
        real = Word(nums) + '.' + Word(nums)
        print(real.parseString('3.1416')) # -> ['3', '.', '1416']
        # will also erroneously match the following
        print(real.parseString('3. 1416')) # -> ['3', '.', '1416']

        real = Combine(Word(nums) + '.' + Word(nums))
        print(real.parseString('3.1416')) # -> ['3.1416']
        # no match when there are internal spaces
        print(real.parseString('3. 1416')) # -> Exception: Expected W:(0123...)
    """

    def __init__(self, expr, joinString='', adjacent=True):
        if False:
            print('Hello World!')
        super(Combine, self).__init__(expr)
        if adjacent:
            self.leaveWhitespace()
        self.adjacent = adjacent
        self.skipWhitespace = True
        self.joinString = joinString
        self.callPreparse = True

    def ignore(self, other):
        if False:
            i = 10
            return i + 15
        if self.adjacent:
            ParserElement.ignore(self, other)
        else:
            super(Combine, self).ignore(other)
        return self

    def postParse(self, instring, loc, tokenlist):
        if False:
            i = 10
            return i + 15
        retToks = tokenlist.copy()
        del retToks[:]
        retToks += ParseResults([''.join(tokenlist._asStringList(self.joinString))], modal=self.modalResults)
        if self.resultsName and retToks.haskeys():
            return [retToks]
        else:
            return retToks

class Group(TokenConverter):
    """
    Converter to return the matched tokens as a list - useful for returning tokens of C{L{ZeroOrMore}} and C{L{OneOrMore}} expressions.

    Example::
        ident = Word(alphas)
        num = Word(nums)
        term = ident | num
        func = ident + Optional(delimitedList(term))
        print(func.parseString("fn a,b,100"))  # -> ['fn', 'a', 'b', '100']

        func = ident + Group(Optional(delimitedList(term)))
        print(func.parseString("fn a,b,100"))  # -> ['fn', ['a', 'b', '100']]
    """

    def __init__(self, expr):
        if False:
            while True:
                i = 10
        super(Group, self).__init__(expr)
        self.saveAsList = True

    def postParse(self, instring, loc, tokenlist):
        if False:
            for i in range(10):
                print('nop')
        return [tokenlist]

class Dict(TokenConverter):
    """
    Converter to return a repetitive expression as a list, but also as a dictionary.
    Each element can also be referenced using the first token in the expression as its key.
    Useful for tabular report scraping when the first column can be used as a item key.

    Example::
        data_word = Word(alphas)
        label = data_word + FollowedBy(':')
        attr_expr = Group(label + Suppress(':') + OneOrMore(data_word).setParseAction(' '.join))

        text = "shape: SQUARE posn: upper left color: light blue texture: burlap"
        attr_expr = (label + Suppress(':') + OneOrMore(data_word, stopOn=label).setParseAction(' '.join))
        
        # print attributes as plain groups
        print(OneOrMore(attr_expr).parseString(text).dump())
        
        # instead of OneOrMore(expr), parse using Dict(OneOrMore(Group(expr))) - Dict will auto-assign names
        result = Dict(OneOrMore(Group(attr_expr))).parseString(text)
        print(result.dump())
        
        # access named fields as dict entries, or output as dict
        print(result['shape'])        
        print(result.asDict())
    prints::
        ['shape', 'SQUARE', 'posn', 'upper left', 'color', 'light blue', 'texture', 'burlap']

        [['shape', 'SQUARE'], ['posn', 'upper left'], ['color', 'light blue'], ['texture', 'burlap']]
        - color: light blue
        - posn: upper left
        - shape: SQUARE
        - texture: burlap
        SQUARE
        {'color': 'light blue', 'posn': 'upper left', 'texture': 'burlap', 'shape': 'SQUARE'}
    See more examples at L{ParseResults} of accessing fields by results name.
    """

    def __init__(self, expr):
        if False:
            print('Hello World!')
        super(Dict, self).__init__(expr)
        self.saveAsList = True

    def postParse(self, instring, loc, tokenlist):
        if False:
            print('Hello World!')
        for (i, tok) in enumerate(tokenlist):
            if len(tok) == 0:
                continue
            ikey = tok[0]
            if isinstance(ikey, int):
                ikey = _ustr(tok[0]).strip()
            if len(tok) == 1:
                tokenlist[ikey] = _ParseResultsWithOffset('', i)
            elif len(tok) == 2 and (not isinstance(tok[1], ParseResults)):
                tokenlist[ikey] = _ParseResultsWithOffset(tok[1], i)
            else:
                dictvalue = tok.copy()
                del dictvalue[0]
                if len(dictvalue) != 1 or (isinstance(dictvalue, ParseResults) and dictvalue.haskeys()):
                    tokenlist[ikey] = _ParseResultsWithOffset(dictvalue, i)
                else:
                    tokenlist[ikey] = _ParseResultsWithOffset(dictvalue[0], i)
        if self.resultsName:
            return [tokenlist]
        else:
            return tokenlist

class Suppress(TokenConverter):
    """
    Converter for ignoring the results of a parsed expression.

    Example::
        source = "a, b, c,d"
        wd = Word(alphas)
        wd_list1 = wd + ZeroOrMore(',' + wd)
        print(wd_list1.parseString(source))

        # often, delimiters that are useful during parsing are just in the
        # way afterward - use Suppress to keep them out of the parsed output
        wd_list2 = wd + ZeroOrMore(Suppress(',') + wd)
        print(wd_list2.parseString(source))
    prints::
        ['a', ',', 'b', ',', 'c', ',', 'd']
        ['a', 'b', 'c', 'd']
    (See also L{delimitedList}.)
    """

    def postParse(self, instring, loc, tokenlist):
        if False:
            for i in range(10):
                print('nop')
        return []

    def suppress(self):
        if False:
            print('Hello World!')
        return self

class OnlyOnce(object):
    """
    Wrapper for parse actions, to ensure they are only called once.
    """

    def __init__(self, methodCall):
        if False:
            print('Hello World!')
        self.callable = _trim_arity(methodCall)
        self.called = False

    def __call__(self, s, l, t):
        if False:
            return 10
        if not self.called:
            results = self.callable(s, l, t)
            self.called = True
            return results
        raise ParseException(s, l, '')

    def reset(self):
        if False:
            while True:
                i = 10
        self.called = False

def traceParseAction(f):
    if False:
        while True:
            i = 10
    '\n    Decorator for debugging parse actions. \n    \n    When the parse action is called, this decorator will print C{">> entering I{method-name}(line:I{current_source_line}, I{parse_location}, I{matched_tokens})".}\n    When the parse action completes, the decorator will print C{"<<"} followed by the returned value, or any exception that the parse action raised.\n\n    Example::\n        wd = Word(alphas)\n\n        @traceParseAction\n        def remove_duplicate_chars(tokens):\n            return \'\'.join(sorted(set(\'\'.join(tokens)))\n\n        wds = OneOrMore(wd).setParseAction(remove_duplicate_chars)\n        print(wds.parseString("slkdjs sld sldd sdlf sdljf"))\n    prints::\n        >>entering remove_duplicate_chars(line: \'slkdjs sld sldd sdlf sdljf\', 0, ([\'slkdjs\', \'sld\', \'sldd\', \'sdlf\', \'sdljf\'], {}))\n        <<leaving remove_duplicate_chars (ret: \'dfjkls\')\n        [\'dfjkls\']\n    '
    f = _trim_arity(f)

    def z(*paArgs):
        if False:
            return 10
        thisFunc = f.__name__
        (s, l, t) = paArgs[-3:]
        if len(paArgs) > 3:
            thisFunc = paArgs[0].__class__.__name__ + '.' + thisFunc
        sys.stderr.write(">>entering %s(line: '%s', %d, %r)\n" % (thisFunc, line(l, s), l, t))
        try:
            ret = f(*paArgs)
        except Exception as exc:
            sys.stderr.write('<<leaving %s (exception: %s)\n' % (thisFunc, exc))
            raise
        sys.stderr.write('<<leaving %s (ret: %r)\n' % (thisFunc, ret))
        return ret
    try:
        z.__name__ = f.__name__
    except AttributeError:
        pass
    return z

def delimitedList(expr, delim=',', combine=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper to define a delimited list of expressions - the delimiter defaults to \',\'.\n    By default, the list elements and delimiters can have intervening whitespace, and\n    comments, but this can be overridden by passing C{combine=True} in the constructor.\n    If C{combine} is set to C{True}, the matching tokens are returned as a single token\n    string, with the delimiters included; otherwise, the matching tokens are returned\n    as a list of tokens, with the delimiters suppressed.\n\n    Example::\n        delimitedList(Word(alphas)).parseString("aa,bb,cc") # -> [\'aa\', \'bb\', \'cc\']\n        delimitedList(Word(hexnums), delim=\':\', combine=True).parseString("AA:BB:CC:DD:EE") # -> [\'AA:BB:CC:DD:EE\']\n    '
    dlName = _ustr(expr) + ' [' + _ustr(delim) + ' ' + _ustr(expr) + ']...'
    if combine:
        return Combine(expr + ZeroOrMore(delim + expr)).setName(dlName)
    else:
        return (expr + ZeroOrMore(Suppress(delim) + expr)).setName(dlName)

def countedArray(expr, intExpr=None):
    if False:
        while True:
            i = 10
    "\n    Helper to define a counted list of expressions.\n    This helper defines a pattern of the form::\n        integer expr expr expr...\n    where the leading integer tells how many expr expressions follow.\n    The matched tokens returns the array of expr tokens as a list - the leading count token is suppressed.\n    \n    If C{intExpr} is specified, it should be a pyparsing expression that produces an integer value.\n\n    Example::\n        countedArray(Word(alphas)).parseString('2 ab cd ef')  # -> ['ab', 'cd']\n\n        # in this parser, the leading integer value is given in binary,\n        # '10' indicating that 2 values are in the array\n        binaryConstant = Word('01').setParseAction(lambda t: int(t[0], 2))\n        countedArray(Word(alphas), intExpr=binaryConstant).parseString('10 ab cd ef')  # -> ['ab', 'cd']\n    "
    arrayExpr = Forward()

    def countFieldParseAction(s, l, t):
        if False:
            while True:
                i = 10
        n = t[0]
        arrayExpr << (n and Group(And([expr] * n)) or Group(empty))
        return []
    if intExpr is None:
        intExpr = Word(nums).setParseAction(lambda t: int(t[0]))
    else:
        intExpr = intExpr.copy()
    intExpr.setName('arrayLen')
    intExpr.addParseAction(countFieldParseAction, callDuringTry=True)
    return (intExpr + arrayExpr).setName('(len) ' + _ustr(expr) + '...')

def _flatten(L):
    if False:
        print('Hello World!')
    ret = []
    for i in L:
        if isinstance(i, list):
            ret.extend(_flatten(i))
        else:
            ret.append(i)
    return ret

def matchPreviousLiteral(expr):
    if False:
        return 10
    '\n    Helper to define an expression that is indirectly defined from\n    the tokens matched in a previous expression, that is, it looks\n    for a \'repeat\' of a previous expression.  For example::\n        first = Word(nums)\n        second = matchPreviousLiteral(first)\n        matchExpr = first + ":" + second\n    will match C{"1:1"}, but not C{"1:2"}.  Because this matches a\n    previous literal, will also match the leading C{"1:1"} in C{"1:10"}.\n    If this is not desired, use C{matchPreviousExpr}.\n    Do I{not} use with packrat parsing enabled.\n    '
    rep = Forward()

    def copyTokenToRepeater(s, l, t):
        if False:
            while True:
                i = 10
        if t:
            if len(t) == 1:
                rep << t[0]
            else:
                tflat = _flatten(t.asList())
                rep << And((Literal(tt) for tt in tflat))
        else:
            rep << Empty()
    expr.addParseAction(copyTokenToRepeater, callDuringTry=True)
    rep.setName('(prev) ' + _ustr(expr))
    return rep

def matchPreviousExpr(expr):
    if False:
        print('Hello World!')
    '\n    Helper to define an expression that is indirectly defined from\n    the tokens matched in a previous expression, that is, it looks\n    for a \'repeat\' of a previous expression.  For example::\n        first = Word(nums)\n        second = matchPreviousExpr(first)\n        matchExpr = first + ":" + second\n    will match C{"1:1"}, but not C{"1:2"}.  Because this matches by\n    expressions, will I{not} match the leading C{"1:1"} in C{"1:10"};\n    the expressions are evaluated first, and then compared, so\n    C{"1"} is compared with C{"10"}.\n    Do I{not} use with packrat parsing enabled.\n    '
    rep = Forward()
    e2 = expr.copy()
    rep <<= e2

    def copyTokenToRepeater(s, l, t):
        if False:
            for i in range(10):
                print('nop')
        matchTokens = _flatten(t.asList())

        def mustMatchTheseTokens(s, l, t):
            if False:
                return 10
            theseTokens = _flatten(t.asList())
            if theseTokens != matchTokens:
                raise ParseException('', 0, '')
        rep.setParseAction(mustMatchTheseTokens, callDuringTry=True)
    expr.addParseAction(copyTokenToRepeater, callDuringTry=True)
    rep.setName('(prev) ' + _ustr(expr))
    return rep

def _escapeRegexRangeChars(s):
    if False:
        return 10
    for c in '\\^-]':
        s = s.replace(c, _bslash + c)
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    return _ustr(s)

def oneOf(strs, caseless=False, useRegex=True):
    if False:
        while True:
            i = 10
    '\n    Helper to quickly define a set of alternative Literals, and makes sure to do\n    longest-first testing when there is a conflict, regardless of the input order,\n    but returns a C{L{MatchFirst}} for best performance.\n\n    Parameters:\n     - strs - a string of space-delimited literals, or a collection of string literals\n     - caseless - (default=C{False}) - treat all literals as caseless\n     - useRegex - (default=C{True}) - as an optimization, will generate a Regex\n          object; otherwise, will generate a C{MatchFirst} object (if C{caseless=True}, or\n          if creating a C{Regex} raises an exception)\n\n    Example::\n        comp_oper = oneOf("< = > <= >= !=")\n        var = Word(alphas)\n        number = Word(nums)\n        term = var | number\n        comparison_expr = term + comp_oper + term\n        print(comparison_expr.searchString("B = 12  AA=23 B<=AA AA>12"))\n    prints::\n        [[\'B\', \'=\', \'12\'], [\'AA\', \'=\', \'23\'], [\'B\', \'<=\', \'AA\'], [\'AA\', \'>\', \'12\']]\n    '
    if caseless:
        isequal = lambda a, b: a.upper() == b.upper()
        masks = lambda a, b: b.upper().startswith(a.upper())
        parseElementClass = CaselessLiteral
    else:
        isequal = lambda a, b: a == b
        masks = lambda a, b: b.startswith(a)
        parseElementClass = Literal
    symbols = []
    if isinstance(strs, basestring):
        symbols = strs.split()
    elif isinstance(strs, collections.Iterable):
        symbols = list(strs)
    else:
        warnings.warn('Invalid argument to oneOf, expected string or iterable', SyntaxWarning, stacklevel=2)
    if not symbols:
        return NoMatch()
    i = 0
    while i < len(symbols) - 1:
        cur = symbols[i]
        for (j, other) in enumerate(symbols[i + 1:]):
            if isequal(other, cur):
                del symbols[i + j + 1]
                break
            elif masks(cur, other):
                del symbols[i + j + 1]
                symbols.insert(i, other)
                cur = other
                break
        else:
            i += 1
    if not caseless and useRegex:
        try:
            if len(symbols) == len(''.join(symbols)):
                return Regex('[%s]' % ''.join((_escapeRegexRangeChars(sym) for sym in symbols))).setName(' | '.join(symbols))
            else:
                return Regex('|'.join((re.escape(sym) for sym in symbols))).setName(' | '.join(symbols))
        except Exception:
            warnings.warn('Exception creating Regex for oneOf, building MatchFirst', SyntaxWarning, stacklevel=2)
    return MatchFirst((parseElementClass(sym) for sym in symbols)).setName(' | '.join(symbols))

def dictOf(key, value):
    if False:
        print('Hello World!')
    '\n    Helper to easily and clearly define a dictionary by specifying the respective patterns\n    for the key and value.  Takes care of defining the C{L{Dict}}, C{L{ZeroOrMore}}, and C{L{Group}} tokens\n    in the proper order.  The key pattern can include delimiting markers or punctuation,\n    as long as they are suppressed, thereby leaving the significant key text.  The value\n    pattern can include named results, so that the C{Dict} results can include named token\n    fields.\n\n    Example::\n        text = "shape: SQUARE posn: upper left color: light blue texture: burlap"\n        attr_expr = (label + Suppress(\':\') + OneOrMore(data_word, stopOn=label).setParseAction(\' \'.join))\n        print(OneOrMore(attr_expr).parseString(text).dump())\n        \n        attr_label = label\n        attr_value = Suppress(\':\') + OneOrMore(data_word, stopOn=label).setParseAction(\' \'.join)\n\n        # similar to Dict, but simpler call format\n        result = dictOf(attr_label, attr_value).parseString(text)\n        print(result.dump())\n        print(result[\'shape\'])\n        print(result.shape)  # object attribute access works too\n        print(result.asDict())\n    prints::\n        [[\'shape\', \'SQUARE\'], [\'posn\', \'upper left\'], [\'color\', \'light blue\'], [\'texture\', \'burlap\']]\n        - color: light blue\n        - posn: upper left\n        - shape: SQUARE\n        - texture: burlap\n        SQUARE\n        SQUARE\n        {\'color\': \'light blue\', \'shape\': \'SQUARE\', \'posn\': \'upper left\', \'texture\': \'burlap\'}\n    '
    return Dict(ZeroOrMore(Group(key + value)))

def originalTextFor(expr, asString=True):
    if False:
        while True:
            i = 10
    '\n    Helper to return the original, untokenized text for a given expression.  Useful to\n    restore the parsed fields of an HTML start tag into the raw tag text itself, or to\n    revert separate tokens with intervening whitespace back to the original matching\n    input text. By default, returns astring containing the original parsed text.  \n       \n    If the optional C{asString} argument is passed as C{False}, then the return value is a \n    C{L{ParseResults}} containing any results names that were originally matched, and a \n    single token containing the original matched text from the input string.  So if \n    the expression passed to C{L{originalTextFor}} contains expressions with defined\n    results names, you must set C{asString} to C{False} if you want to preserve those\n    results name values.\n\n    Example::\n        src = "this is test <b> bold <i>text</i> </b> normal text "\n        for tag in ("b","i"):\n            opener,closer = makeHTMLTags(tag)\n            patt = originalTextFor(opener + SkipTo(closer) + closer)\n            print(patt.searchString(src)[0])\n    prints::\n        [\'<b> bold <i>text</i> </b>\']\n        [\'<i>text</i>\']\n    '
    locMarker = Empty().setParseAction(lambda s, loc, t: loc)
    endlocMarker = locMarker.copy()
    endlocMarker.callPreparse = False
    matchExpr = locMarker('_original_start') + expr + endlocMarker('_original_end')
    if asString:
        extractText = lambda s, l, t: s[t._original_start:t._original_end]
    else:

        def extractText(s, l, t):
            if False:
                return 10
            t[:] = [s[t.pop('_original_start'):t.pop('_original_end')]]
    matchExpr.setParseAction(extractText)
    matchExpr.ignoreExprs = expr.ignoreExprs
    return matchExpr

def ungroup(expr):
    if False:
        i = 10
        return i + 15
    "\n    Helper to undo pyparsing's default grouping of And expressions, even\n    if all but one are non-empty.\n    "
    return TokenConverter(expr).setParseAction(lambda t: t[0])

def locatedExpr(expr):
    if False:
        return 10
    '\n    Helper to decorate a returned token with its starting and ending locations in the input string.\n    This helper adds the following results names:\n     - locn_start = location where matched expression begins\n     - locn_end = location where matched expression ends\n     - value = the actual parsed results\n\n    Be careful if the input text contains C{<TAB>} characters, you may want to call\n    C{L{ParserElement.parseWithTabs}}\n\n    Example::\n        wd = Word(alphas)\n        for match in locatedExpr(wd).searchString("ljsdf123lksdjjf123lkkjj1222"):\n            print(match)\n    prints::\n        [[0, \'ljsdf\', 5]]\n        [[8, \'lksdjjf\', 15]]\n        [[18, \'lkkjj\', 23]]\n    '
    locator = Empty().setParseAction(lambda s, l, t: l)
    return Group(locator('locn_start') + expr('value') + locator.copy().leaveWhitespace()('locn_end'))
empty = Empty().setName('empty')
lineStart = LineStart().setName('lineStart')
lineEnd = LineEnd().setName('lineEnd')
stringStart = StringStart().setName('stringStart')
stringEnd = StringEnd().setName('stringEnd')
_escapedPunc = Word(_bslash, '\\[]-*.$+^?()~ ', exact=2).setParseAction(lambda s, l, t: t[0][1])
_escapedHexChar = Regex('\\\\0?[xX][0-9a-fA-F]+').setParseAction(lambda s, l, t: unichr(int(t[0].lstrip('\\0x'), 16)))
_escapedOctChar = Regex('\\\\0[0-7]+').setParseAction(lambda s, l, t: unichr(int(t[0][1:], 8)))
_singleChar = _escapedPunc | _escapedHexChar | _escapedOctChar | Word(printables, excludeChars='\\]', exact=1) | Regex('\\w', re.UNICODE)
_charRange = Group(_singleChar + Suppress('-') + _singleChar)
_reBracketExpr = Literal('[') + Optional('^').setResultsName('negate') + Group(OneOrMore(_charRange | _singleChar)).setResultsName('body') + ']'

def srange(s):
    if False:
        print('Hello World!')
    '\n    Helper to easily define string ranges for use in Word construction.  Borrows\n    syntax from regexp \'[]\' string range definitions::\n        srange("[0-9]")   -> "0123456789"\n        srange("[a-z]")   -> "abcdefghijklmnopqrstuvwxyz"\n        srange("[a-z$_]") -> "abcdefghijklmnopqrstuvwxyz$_"\n    The input string must be enclosed in []\'s, and the returned string is the expanded\n    character set joined into a single string.\n    The values enclosed in the []\'s may be:\n     - a single character\n     - an escaped character with a leading backslash (such as C{\\-} or C{\\]})\n     - an escaped hex character with a leading C{\'\\x\'} (C{\\x21}, which is a C{\'!\'} character) \n         (C{\\0x##} is also supported for backwards compatibility) \n     - an escaped octal character with a leading C{\'\\0\'} (C{\\041}, which is a C{\'!\'} character)\n     - a range of any of the above, separated by a dash (C{\'a-z\'}, etc.)\n     - any combination of the above (C{\'aeiouy\'}, C{\'a-zA-Z0-9_$\'}, etc.)\n    '
    _expanded = lambda p: p if not isinstance(p, ParseResults) else ''.join((unichr(c) for c in range(ord(p[0]), ord(p[1]) + 1)))
    try:
        return ''.join((_expanded(part) for part in _reBracketExpr.parseString(s).body))
    except Exception:
        return ''

def matchOnlyAtCol(n):
    if False:
        while True:
            i = 10
    '\n    Helper method for defining parse actions that require matching at a specific\n    column in the input text.\n    '

    def verifyCol(strg, locn, toks):
        if False:
            while True:
                i = 10
        if col(locn, strg) != n:
            raise ParseException(strg, locn, 'matched token not at column %d' % n)
    return verifyCol

def replaceWith(replStr):
    if False:
        print('Hello World!')
    '\n    Helper method for common parse actions that simply return a literal value.  Especially\n    useful when used with C{L{transformString<ParserElement.transformString>}()}.\n\n    Example::\n        num = Word(nums).setParseAction(lambda toks: int(toks[0]))\n        na = oneOf("N/A NA").setParseAction(replaceWith(math.nan))\n        term = na | num\n        \n        OneOrMore(term).parseString("324 234 N/A 234") # -> [324, 234, nan, 234]\n    '
    return lambda s, l, t: [replStr]

def removeQuotes(s, l, t):
    if False:
        return 10
    '\n    Helper parse action for removing quotation marks from parsed quoted strings.\n\n    Example::\n        # by default, quotation marks are included in parsed results\n        quotedString.parseString("\'Now is the Winter of our Discontent\'") # -> ["\'Now is the Winter of our Discontent\'"]\n\n        # use removeQuotes to strip quotation marks from parsed results\n        quotedString.setParseAction(removeQuotes)\n        quotedString.parseString("\'Now is the Winter of our Discontent\'") # -> ["Now is the Winter of our Discontent"]\n    '
    return t[0][1:-1]

def tokenMap(func, *args):
    if False:
        print('Hello World!')
    "\n    Helper to define a parse action by mapping a function to all elements of a ParseResults list.If any additional \n    args are passed, they are forwarded to the given function as additional arguments after\n    the token, as in C{hex_integer = Word(hexnums).setParseAction(tokenMap(int, 16))}, which will convert the\n    parsed data to an integer using base 16.\n\n    Example (compare the last to example in L{ParserElement.transformString}::\n        hex_ints = OneOrMore(Word(hexnums)).setParseAction(tokenMap(int, 16))\n        hex_ints.runTests('''\n            00 11 22 aa FF 0a 0d 1a\n            ''')\n        \n        upperword = Word(alphas).setParseAction(tokenMap(str.upper))\n        OneOrMore(upperword).runTests('''\n            my kingdom for a horse\n            ''')\n\n        wd = Word(alphas).setParseAction(tokenMap(str.title))\n        OneOrMore(wd).setParseAction(' '.join).runTests('''\n            now is the winter of our discontent made glorious summer by this sun of york\n            ''')\n    prints::\n        00 11 22 aa FF 0a 0d 1a\n        [0, 17, 34, 170, 255, 10, 13, 26]\n\n        my kingdom for a horse\n        ['MY', 'KINGDOM', 'FOR', 'A', 'HORSE']\n\n        now is the winter of our discontent made glorious summer by this sun of york\n        ['Now Is The Winter Of Our Discontent Made Glorious Summer By This Sun Of York']\n    "

    def pa(s, l, t):
        if False:
            for i in range(10):
                print('nop')
        return [func(tokn, *args) for tokn in t]
    try:
        func_name = getattr(func, '__name__', getattr(func, '__class__').__name__)
    except Exception:
        func_name = str(func)
    pa.__name__ = func_name
    return pa
upcaseTokens = tokenMap(lambda t: _ustr(t).upper())
'(Deprecated) Helper parse action to convert tokens to upper case. Deprecated in favor of L{pyparsing_common.upcaseTokens}'
downcaseTokens = tokenMap(lambda t: _ustr(t).lower())
'(Deprecated) Helper parse action to convert tokens to lower case. Deprecated in favor of L{pyparsing_common.downcaseTokens}'

def _makeTags(tagStr, xml):
    if False:
        return 10
    'Internal helper to construct opening and closing tag expressions, given a tag name'
    if isinstance(tagStr, basestring):
        resname = tagStr
        tagStr = Keyword(tagStr, caseless=not xml)
    else:
        resname = tagStr.name
    tagAttrName = Word(alphas, alphanums + '_-:')
    if xml:
        tagAttrValue = dblQuotedString.copy().setParseAction(removeQuotes)
        openTag = Suppress('<') + tagStr('tag') + Dict(ZeroOrMore(Group(tagAttrName + Suppress('=') + tagAttrValue))) + Optional('/', default=[False]).setResultsName('empty').setParseAction(lambda s, l, t: t[0] == '/') + Suppress('>')
    else:
        printablesLessRAbrack = ''.join((c for c in printables if c not in '>'))
        tagAttrValue = quotedString.copy().setParseAction(removeQuotes) | Word(printablesLessRAbrack)
        openTag = Suppress('<') + tagStr('tag') + Dict(ZeroOrMore(Group(tagAttrName.setParseAction(downcaseTokens) + Optional(Suppress('=') + tagAttrValue)))) + Optional('/', default=[False]).setResultsName('empty').setParseAction(lambda s, l, t: t[0] == '/') + Suppress('>')
    closeTag = Combine(_L('</') + tagStr + '>')
    openTag = openTag.setResultsName('start' + ''.join(resname.replace(':', ' ').title().split())).setName('<%s>' % resname)
    closeTag = closeTag.setResultsName('end' + ''.join(resname.replace(':', ' ').title().split())).setName('</%s>' % resname)
    openTag.tag = resname
    closeTag.tag = resname
    return (openTag, closeTag)

def makeHTMLTags(tagStr):
    if False:
        print('Hello World!')
    '\n    Helper to construct opening and closing tag expressions for HTML, given a tag name. Matches\n    tags in either upper or lower case, attributes with namespaces and with quoted or unquoted values.\n\n    Example::\n        text = \'<td>More info at the <a href="http://pyparsing.wikispaces.com">pyparsing</a> wiki page</td>\'\n        # makeHTMLTags returns pyparsing expressions for the opening and closing tags as a 2-tuple\n        a,a_end = makeHTMLTags("A")\n        link_expr = a + SkipTo(a_end)("link_text") + a_end\n        \n        for link in link_expr.searchString(text):\n            # attributes in the <A> tag (like "href" shown here) are also accessible as named results\n            print(link.link_text, \'->\', link.href)\n    prints::\n        pyparsing -> http://pyparsing.wikispaces.com\n    '
    return _makeTags(tagStr, False)

def makeXMLTags(tagStr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper to construct opening and closing tag expressions for XML, given a tag name. Matches\n    tags only in the given upper/lower case.\n\n    Example: similar to L{makeHTMLTags}\n    '
    return _makeTags(tagStr, True)

def withAttribute(*args, **attrDict):
    if False:
        print('Hello World!')
    '\n    Helper to create a validating parse action to be used with start tags created\n    with C{L{makeXMLTags}} or C{L{makeHTMLTags}}. Use C{withAttribute} to qualify a starting tag\n    with a required attribute value, to avoid false matches on common tags such as\n    C{<TD>} or C{<DIV>}.\n\n    Call C{withAttribute} with a series of attribute names and values. Specify the list\n    of filter attributes names and values as:\n     - keyword arguments, as in C{(align="right")}, or\n     - as an explicit dict with C{**} operator, when an attribute name is also a Python\n          reserved word, as in C{**{"class":"Customer", "align":"right"}}\n     - a list of name-value tuples, as in ( ("ns1:class", "Customer"), ("ns2:align","right") )\n    For attribute names with a namespace prefix, you must use the second form.  Attribute\n    names are matched insensitive to upper/lower case.\n       \n    If just testing for C{class} (with or without a namespace), use C{L{withClass}}.\n\n    To verify that the attribute exists, but without specifying a value, pass\n    C{withAttribute.ANY_VALUE} as the value.\n\n    Example::\n        html = \'\'\'\n            <div>\n            Some text\n            <div type="grid">1 4 0 1 0</div>\n            <div type="graph">1,3 2,3 1,1</div>\n            <div>this has no type</div>\n            </div>\n                \n        \'\'\'\n        div,div_end = makeHTMLTags("div")\n\n        # only match div tag having a type attribute with value "grid"\n        div_grid = div().setParseAction(withAttribute(type="grid"))\n        grid_expr = div_grid + SkipTo(div | div_end)("body")\n        for grid_header in grid_expr.searchString(html):\n            print(grid_header.body)\n        \n        # construct a match with any div tag having a type attribute, regardless of the value\n        div_any_type = div().setParseAction(withAttribute(type=withAttribute.ANY_VALUE))\n        div_expr = div_any_type + SkipTo(div | div_end)("body")\n        for div_header in div_expr.searchString(html):\n            print(div_header.body)\n    prints::\n        1 4 0 1 0\n\n        1 4 0 1 0\n        1,3 2,3 1,1\n    '
    if args:
        attrs = args[:]
    else:
        attrs = attrDict.items()
    attrs = [(k, v) for (k, v) in attrs]

    def pa(s, l, tokens):
        if False:
            for i in range(10):
                print('nop')
        for (attrName, attrValue) in attrs:
            if attrName not in tokens:
                raise ParseException(s, l, 'no matching attribute ' + attrName)
            if attrValue != withAttribute.ANY_VALUE and tokens[attrName] != attrValue:
                raise ParseException(s, l, "attribute '%s' has value '%s', must be '%s'" % (attrName, tokens[attrName], attrValue))
    return pa
withAttribute.ANY_VALUE = object()

def withClass(classname, namespace=''):
    if False:
        print('Hello World!')
    '\n    Simplified version of C{L{withAttribute}} when matching on a div class - made\n    difficult because C{class} is a reserved word in Python.\n\n    Example::\n        html = \'\'\'\n            <div>\n            Some text\n            <div class="grid">1 4 0 1 0</div>\n            <div class="graph">1,3 2,3 1,1</div>\n            <div>this &lt;div&gt; has no class</div>\n            </div>\n                \n        \'\'\'\n        div,div_end = makeHTMLTags("div")\n        div_grid = div().setParseAction(withClass("grid"))\n        \n        grid_expr = div_grid + SkipTo(div | div_end)("body")\n        for grid_header in grid_expr.searchString(html):\n            print(grid_header.body)\n        \n        div_any_type = div().setParseAction(withClass(withAttribute.ANY_VALUE))\n        div_expr = div_any_type + SkipTo(div | div_end)("body")\n        for div_header in div_expr.searchString(html):\n            print(div_header.body)\n    prints::\n        1 4 0 1 0\n\n        1 4 0 1 0\n        1,3 2,3 1,1\n    '
    classattr = '%s:class' % namespace if namespace else 'class'
    return withAttribute(**{classattr: classname})
opAssoc = _Constants()
opAssoc.LEFT = object()
opAssoc.RIGHT = object()

def infixNotation(baseExpr, opList, lpar=Suppress('('), rpar=Suppress(')')):
    if False:
        print('Hello World!')
    "\n    Helper method for constructing grammars of expressions made up of\n    operators working in a precedence hierarchy.  Operators may be unary or\n    binary, left- or right-associative.  Parse actions can also be attached\n    to operator expressions. The generated parser will also recognize the use \n    of parentheses to override operator precedences (see example below).\n    \n    Note: if you define a deep operator list, you may see performance issues\n    when using infixNotation. See L{ParserElement.enablePackrat} for a\n    mechanism to potentially improve your parser performance.\n\n    Parameters:\n     - baseExpr - expression representing the most basic element for the nested\n     - opList - list of tuples, one for each operator precedence level in the\n      expression grammar; each tuple is of the form\n      (opExpr, numTerms, rightLeftAssoc, parseAction), where:\n       - opExpr is the pyparsing expression for the operator;\n          may also be a string, which will be converted to a Literal;\n          if numTerms is 3, opExpr is a tuple of two expressions, for the\n          two operators separating the 3 terms\n       - numTerms is the number of terms for this operator (must\n          be 1, 2, or 3)\n       - rightLeftAssoc is the indicator whether the operator is\n          right or left associative, using the pyparsing-defined\n          constants C{opAssoc.RIGHT} and C{opAssoc.LEFT}.\n       - parseAction is the parse action to be associated with\n          expressions matching this operator expression (the\n          parse action tuple member may be omitted); if the parse action\n          is passed a tuple or list of functions, this is equivalent to\n          calling C{setParseAction(*fn)} (L{ParserElement.setParseAction})\n     - lpar - expression for matching left-parentheses (default=C{Suppress('(')})\n     - rpar - expression for matching right-parentheses (default=C{Suppress(')')})\n\n    Example::\n        # simple example of four-function arithmetic with ints and variable names\n        integer = pyparsing_common.signed_integer\n        varname = pyparsing_common.identifier \n        \n        arith_expr = infixNotation(integer | varname,\n            [\n            ('-', 1, opAssoc.RIGHT),\n            (oneOf('* /'), 2, opAssoc.LEFT),\n            (oneOf('+ -'), 2, opAssoc.LEFT),\n            ])\n        \n        arith_expr.runTests('''\n            5+3*6\n            (5+3)*6\n            -2--11\n            ''', fullDump=False)\n    prints::\n        5+3*6\n        [[5, '+', [3, '*', 6]]]\n\n        (5+3)*6\n        [[[5, '+', 3], '*', 6]]\n\n        -2--11\n        [[['-', 2], '-', ['-', 11]]]\n    "
    ret = Forward()
    lastExpr = baseExpr | lpar + ret + rpar
    for (i, operDef) in enumerate(opList):
        (opExpr, arity, rightLeftAssoc, pa) = (operDef + (None,))[:4]
        termName = '%s term' % opExpr if arity < 3 else '%s%s term' % opExpr
        if arity == 3:
            if opExpr is None or len(opExpr) != 2:
                raise ValueError('if numterms=3, opExpr must be a tuple or list of two expressions')
            (opExpr1, opExpr2) = opExpr
        thisExpr = Forward().setName(termName)
        if rightLeftAssoc == opAssoc.LEFT:
            if arity == 1:
                matchExpr = FollowedBy(lastExpr + opExpr) + Group(lastExpr + OneOrMore(opExpr))
            elif arity == 2:
                if opExpr is not None:
                    matchExpr = FollowedBy(lastExpr + opExpr + lastExpr) + Group(lastExpr + OneOrMore(opExpr + lastExpr))
                else:
                    matchExpr = FollowedBy(lastExpr + lastExpr) + Group(lastExpr + OneOrMore(lastExpr))
            elif arity == 3:
                matchExpr = FollowedBy(lastExpr + opExpr1 + lastExpr + opExpr2 + lastExpr) + Group(lastExpr + opExpr1 + lastExpr + opExpr2 + lastExpr)
            else:
                raise ValueError('operator must be unary (1), binary (2), or ternary (3)')
        elif rightLeftAssoc == opAssoc.RIGHT:
            if arity == 1:
                if not isinstance(opExpr, Optional):
                    opExpr = Optional(opExpr)
                matchExpr = FollowedBy(opExpr.expr + thisExpr) + Group(opExpr + thisExpr)
            elif arity == 2:
                if opExpr is not None:
                    matchExpr = FollowedBy(lastExpr + opExpr + thisExpr) + Group(lastExpr + OneOrMore(opExpr + thisExpr))
                else:
                    matchExpr = FollowedBy(lastExpr + thisExpr) + Group(lastExpr + OneOrMore(thisExpr))
            elif arity == 3:
                matchExpr = FollowedBy(lastExpr + opExpr1 + thisExpr + opExpr2 + thisExpr) + Group(lastExpr + opExpr1 + thisExpr + opExpr2 + thisExpr)
            else:
                raise ValueError('operator must be unary (1), binary (2), or ternary (3)')
        else:
            raise ValueError('operator must indicate right or left associativity')
        if pa:
            if isinstance(pa, (tuple, list)):
                matchExpr.setParseAction(*pa)
            else:
                matchExpr.setParseAction(pa)
        thisExpr <<= matchExpr.setName(termName) | lastExpr
        lastExpr = thisExpr
    ret <<= lastExpr
    return ret
operatorPrecedence = infixNotation
'(Deprecated) Former name of C{L{infixNotation}}, will be dropped in a future release.'
dblQuotedString = Combine(Regex('"(?:[^"\\n\\r\\\\]|(?:"")|(?:\\\\(?:[^x]|x[0-9a-fA-F]+)))*') + '"').setName('string enclosed in double quotes')
sglQuotedString = Combine(Regex("'(?:[^'\\n\\r\\\\]|(?:'')|(?:\\\\(?:[^x]|x[0-9a-fA-F]+)))*") + "'").setName('string enclosed in single quotes')
quotedString = Combine(Regex('"(?:[^"\\n\\r\\\\]|(?:"")|(?:\\\\(?:[^x]|x[0-9a-fA-F]+)))*') + '"' | Regex("'(?:[^'\\n\\r\\\\]|(?:'')|(?:\\\\(?:[^x]|x[0-9a-fA-F]+)))*") + "'").setName('quotedString using single or double quotes')
unicodeString = Combine(_L('u') + quotedString.copy()).setName('unicode string literal')

def nestedExpr(opener='(', closer=')', content=None, ignoreExpr=quotedString.copy()):
    if False:
        while True:
            i = 10
    '\n    Helper method for defining nested lists enclosed in opening and closing\n    delimiters ("(" and ")" are the default).\n\n    Parameters:\n     - opener - opening character for a nested list (default=C{"("}); can also be a pyparsing expression\n     - closer - closing character for a nested list (default=C{")"}); can also be a pyparsing expression\n     - content - expression for items within the nested lists (default=C{None})\n     - ignoreExpr - expression for ignoring opening and closing delimiters (default=C{quotedString})\n\n    If an expression is not provided for the content argument, the nested\n    expression will capture all whitespace-delimited content between delimiters\n    as a list of separate values.\n\n    Use the C{ignoreExpr} argument to define expressions that may contain\n    opening or closing characters that should not be treated as opening\n    or closing characters for nesting, such as quotedString or a comment\n    expression.  Specify multiple expressions using an C{L{Or}} or C{L{MatchFirst}}.\n    The default is L{quotedString}, but if no expressions are to be ignored,\n    then pass C{None} for this argument.\n\n    Example::\n        data_type = oneOf("void int short long char float double")\n        decl_data_type = Combine(data_type + Optional(Word(\'*\')))\n        ident = Word(alphas+\'_\', alphanums+\'_\')\n        number = pyparsing_common.number\n        arg = Group(decl_data_type + ident)\n        LPAR,RPAR = map(Suppress, "()")\n\n        code_body = nestedExpr(\'{\', \'}\', ignoreExpr=(quotedString | cStyleComment))\n\n        c_function = (decl_data_type("type") \n                      + ident("name")\n                      + LPAR + Optional(delimitedList(arg), [])("args") + RPAR \n                      + code_body("body"))\n        c_function.ignore(cStyleComment)\n        \n        source_code = \'\'\'\n            int is_odd(int x) { \n                return (x%2); \n            }\n                \n            int dec_to_hex(char hchar) { \n                if (hchar >= \'0\' && hchar <= \'9\') { \n                    return (ord(hchar)-ord(\'0\')); \n                } else { \n                    return (10+ord(hchar)-ord(\'A\'));\n                } \n            }\n        \'\'\'\n        for func in c_function.searchString(source_code):\n            print("%(name)s (%(type)s) args: %(args)s" % func)\n\n    prints::\n        is_odd (int) args: [[\'int\', \'x\']]\n        dec_to_hex (int) args: [[\'char\', \'hchar\']]\n    '
    if opener == closer:
        raise ValueError('opening and closing strings cannot be the same')
    if content is None:
        if isinstance(opener, basestring) and isinstance(closer, basestring):
            if len(opener) == 1 and len(closer) == 1:
                if ignoreExpr is not None:
                    content = Combine(OneOrMore(~ignoreExpr + CharsNotIn(opener + closer + ParserElement.DEFAULT_WHITE_CHARS, exact=1))).setParseAction(lambda t: t[0].strip())
                else:
                    content = empty.copy() + CharsNotIn(opener + closer + ParserElement.DEFAULT_WHITE_CHARS).setParseAction(lambda t: t[0].strip())
            elif ignoreExpr is not None:
                content = Combine(OneOrMore(~ignoreExpr + ~Literal(opener) + ~Literal(closer) + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1))).setParseAction(lambda t: t[0].strip())
            else:
                content = Combine(OneOrMore(~Literal(opener) + ~Literal(closer) + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1))).setParseAction(lambda t: t[0].strip())
        else:
            raise ValueError('opening and closing arguments must be strings if no content expression is given')
    ret = Forward()
    if ignoreExpr is not None:
        ret <<= Group(Suppress(opener) + ZeroOrMore(ignoreExpr | ret | content) + Suppress(closer))
    else:
        ret <<= Group(Suppress(opener) + ZeroOrMore(ret | content) + Suppress(closer))
    ret.setName('nested %s%s expression' % (opener, closer))
    return ret

def indentedBlock(blockStatementExpr, indentStack, indent=True):
    if False:
        i = 10
        return i + 15
    '\n    Helper method for defining space-delimited indentation blocks, such as\n    those used to define block statements in Python source code.\n\n    Parameters:\n     - blockStatementExpr - expression defining syntax of statement that\n            is repeated within the indented block\n     - indentStack - list created by caller to manage indentation stack\n            (multiple statementWithIndentedBlock expressions within a single grammar\n            should share a common indentStack)\n     - indent - boolean indicating whether block must be indented beyond the\n            the current level; set to False for block of left-most statements\n            (default=C{True})\n\n    A valid block must contain at least one C{blockStatement}.\n\n    Example::\n        data = \'\'\'\n        def A(z):\n          A1\n          B = 100\n          G = A2\n          A2\n          A3\n        B\n        def BB(a,b,c):\n          BB1\n          def BBA():\n            bba1\n            bba2\n            bba3\n        C\n        D\n        def spam(x,y):\n             def eggs(z):\n                 pass\n        \'\'\'\n\n\n        indentStack = [1]\n        stmt = Forward()\n\n        identifier = Word(alphas, alphanums)\n        funcDecl = ("def" + identifier + Group( "(" + Optional( delimitedList(identifier) ) + ")" ) + ":")\n        func_body = indentedBlock(stmt, indentStack)\n        funcDef = Group( funcDecl + func_body )\n\n        rvalue = Forward()\n        funcCall = Group(identifier + "(" + Optional(delimitedList(rvalue)) + ")")\n        rvalue << (funcCall | identifier | Word(nums))\n        assignment = Group(identifier + "=" + rvalue)\n        stmt << ( funcDef | assignment | identifier )\n\n        module_body = OneOrMore(stmt)\n\n        parseTree = module_body.parseString(data)\n        parseTree.pprint()\n    prints::\n        [[\'def\',\n          \'A\',\n          [\'(\', \'z\', \')\'],\n          \':\',\n          [[\'A1\'], [[\'B\', \'=\', \'100\']], [[\'G\', \'=\', \'A2\']], [\'A2\'], [\'A3\']]],\n         \'B\',\n         [\'def\',\n          \'BB\',\n          [\'(\', \'a\', \'b\', \'c\', \')\'],\n          \':\',\n          [[\'BB1\'], [[\'def\', \'BBA\', [\'(\', \')\'], \':\', [[\'bba1\'], [\'bba2\'], [\'bba3\']]]]]],\n         \'C\',\n         \'D\',\n         [\'def\',\n          \'spam\',\n          [\'(\', \'x\', \'y\', \')\'],\n          \':\',\n          [[[\'def\', \'eggs\', [\'(\', \'z\', \')\'], \':\', [[\'pass\']]]]]]] \n    '

    def checkPeerIndent(s, l, t):
        if False:
            for i in range(10):
                print('nop')
        if l >= len(s):
            return
        curCol = col(l, s)
        if curCol != indentStack[-1]:
            if curCol > indentStack[-1]:
                raise ParseFatalException(s, l, 'illegal nesting')
            raise ParseException(s, l, 'not a peer entry')

    def checkSubIndent(s, l, t):
        if False:
            for i in range(10):
                print('nop')
        curCol = col(l, s)
        if curCol > indentStack[-1]:
            indentStack.append(curCol)
        else:
            raise ParseException(s, l, 'not a subentry')

    def checkUnindent(s, l, t):
        if False:
            while True:
                i = 10
        if l >= len(s):
            return
        curCol = col(l, s)
        if not (indentStack and curCol < indentStack[-1] and (curCol <= indentStack[-2])):
            raise ParseException(s, l, 'not an unindent')
        indentStack.pop()
    NL = OneOrMore(LineEnd().setWhitespaceChars('\t ').suppress())
    INDENT = (Empty() + Empty().setParseAction(checkSubIndent)).setName('INDENT')
    PEER = Empty().setParseAction(checkPeerIndent).setName('')
    UNDENT = Empty().setParseAction(checkUnindent).setName('UNINDENT')
    if indent:
        smExpr = Group(Optional(NL) + INDENT + OneOrMore(PEER + Group(blockStatementExpr) + Optional(NL)) + UNDENT)
    else:
        smExpr = Group(Optional(NL) + OneOrMore(PEER + Group(blockStatementExpr) + Optional(NL)))
    blockStatementExpr.ignore(_bslash + LineEnd())
    return smExpr.setName('indented block')
alphas8bit = srange('[\\0xc0-\\0xd6\\0xd8-\\0xf6\\0xf8-\\0xff]')
punc8bit = srange('[\\0xa1-\\0xbf\\0xd7\\0xf7]')
(anyOpenTag, anyCloseTag) = makeHTMLTags(Word(alphas, alphanums + '_:').setName('any tag'))
_htmlEntityMap = dict(zip('gt lt amp nbsp quot apos'.split(), '><& "\''))
commonHTMLEntity = Regex('&(?P<entity>' + '|'.join(_htmlEntityMap.keys()) + ');').setName('common HTML entity')

def replaceHTMLEntity(t):
    if False:
        i = 10
        return i + 15
    'Helper parser action to replace common HTML entities with their special characters'
    return _htmlEntityMap.get(t.entity)
cStyleComment = Combine(Regex('/\\*(?:[^*]|\\*(?!/))*') + '*/').setName('C style comment')
'Comment of the form C{/* ... */}'
htmlComment = Regex('<!--[\\s\\S]*?-->').setName('HTML comment')
'Comment of the form C{<!-- ... -->}'
restOfLine = Regex('.*').leaveWhitespace().setName('rest of line')
dblSlashComment = Regex('//(?:\\\\\\n|[^\\n])*').setName('// comment')
'Comment of the form C{// ... (to end of line)}'
cppStyleComment = Combine(Regex('/\\*(?:[^*]|\\*(?!/))*') + '*/' | dblSlashComment).setName('C++ style comment')
'Comment of either form C{L{cStyleComment}} or C{L{dblSlashComment}}'
javaStyleComment = cppStyleComment
'Same as C{L{cppStyleComment}}'
pythonStyleComment = Regex('#.*').setName('Python style comment')
'Comment of the form C{# ... (to end of line)}'
_commasepitem = Combine(OneOrMore(Word(printables, excludeChars=',') + Optional(Word(' \t') + ~Literal(',') + ~LineEnd()))).streamline().setName('commaItem')
commaSeparatedList = delimitedList(Optional(quotedString.copy() | _commasepitem, default='')).setName('commaSeparatedList')
'(Deprecated) Predefined expression of 1 or more printable words or quoted strings, separated by commas.\n   This expression is deprecated in favor of L{pyparsing_common.comma_separated_list}.'

class pyparsing_common:
    """
    Here are some common low-level expressions that may be useful in jump-starting parser development:
     - numeric forms (L{integers<integer>}, L{reals<real>}, L{scientific notation<sci_real>})
     - common L{programming identifiers<identifier>}
     - network addresses (L{MAC<mac_address>}, L{IPv4<ipv4_address>}, L{IPv6<ipv6_address>})
     - ISO8601 L{dates<iso8601_date>} and L{datetime<iso8601_datetime>}
     - L{UUID<uuid>}
     - L{comma-separated list<comma_separated_list>}
    Parse actions:
     - C{L{convertToInteger}}
     - C{L{convertToFloat}}
     - C{L{convertToDate}}
     - C{L{convertToDatetime}}
     - C{L{stripHTMLTags}}
     - C{L{upcaseTokens}}
     - C{L{downcaseTokens}}

    Example::
        pyparsing_common.number.runTests('''
            # any int or real number, returned as the appropriate type
            100
            -100
            +100
            3.14159
            6.02e23
            1e-12
            ''')

        pyparsing_common.fnumber.runTests('''
            # any int or real number, returned as float
            100
            -100
            +100
            3.14159
            6.02e23
            1e-12
            ''')

        pyparsing_common.hex_integer.runTests('''
            # hex numbers
            100
            FF
            ''')

        pyparsing_common.fraction.runTests('''
            # fractions
            1/2
            -3/4
            ''')

        pyparsing_common.mixed_integer.runTests('''
            # mixed fractions
            1
            1/2
            -3/4
            1-3/4
            ''')

        import uuid
        pyparsing_common.uuid.setParseAction(tokenMap(uuid.UUID))
        pyparsing_common.uuid.runTests('''
            # uuid
            12345678-1234-5678-1234-567812345678
            ''')
    prints::
        # any int or real number, returned as the appropriate type
        100
        [100]

        -100
        [-100]

        +100
        [100]

        3.14159
        [3.14159]

        6.02e23
        [6.02e+23]

        1e-12
        [1e-12]

        # any int or real number, returned as float
        100
        [100.0]

        -100
        [-100.0]

        +100
        [100.0]

        3.14159
        [3.14159]

        6.02e23
        [6.02e+23]

        1e-12
        [1e-12]

        # hex numbers
        100
        [256]

        FF
        [255]

        # fractions
        1/2
        [0.5]

        -3/4
        [-0.75]

        # mixed fractions
        1
        [1]

        1/2
        [0.5]

        -3/4
        [-0.75]

        1-3/4
        [1.75]

        # uuid
        12345678-1234-5678-1234-567812345678
        [UUID('12345678-1234-5678-1234-567812345678')]
    """
    convertToInteger = tokenMap(int)
    '\n    Parse action for converting parsed integers to Python int\n    '
    convertToFloat = tokenMap(float)
    '\n    Parse action for converting parsed numbers to Python float\n    '
    integer = Word(nums).setName('integer').setParseAction(convertToInteger)
    'expression that parses an unsigned integer, returns an int'
    hex_integer = Word(hexnums).setName('hex integer').setParseAction(tokenMap(int, 16))
    'expression that parses a hexadecimal integer, returns an int'
    signed_integer = Regex('[+-]?\\d+').setName('signed integer').setParseAction(convertToInteger)
    'expression that parses an integer with optional leading sign, returns an int'
    fraction = (signed_integer().setParseAction(convertToFloat) + '/' + signed_integer().setParseAction(convertToFloat)).setName('fraction')
    'fractional expression of an integer divided by an integer, returns a float'
    fraction.addParseAction(lambda t: t[0] / t[-1])
    mixed_integer = (fraction | signed_integer + Optional(Optional('-').suppress() + fraction)).setName('fraction or mixed integer-fraction')
    "mixed integer of the form 'integer - fraction', with optional leading integer, returns float"
    mixed_integer.addParseAction(sum)
    real = Regex('[+-]?\\d+\\.\\d*').setName('real number').setParseAction(convertToFloat)
    'expression that parses a floating point number and returns a float'
    sci_real = Regex('[+-]?\\d+([eE][+-]?\\d+|\\.\\d*([eE][+-]?\\d+)?)').setName('real number with scientific notation').setParseAction(convertToFloat)
    'expression that parses a floating point number with optional scientific notation and returns a float'
    number = (sci_real | real | signed_integer).streamline()
    'any numeric expression, returns the corresponding Python type'
    fnumber = Regex('[+-]?\\d+\\.?\\d*([eE][+-]?\\d+)?').setName('fnumber').setParseAction(convertToFloat)
    'any int or real number, returned as float'
    identifier = Word(alphas + '_', alphanums + '_').setName('identifier')
    "typical code identifier (leading alpha or '_', followed by 0 or more alphas, nums, or '_')"
    ipv4_address = Regex('(25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})(\\.(25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}').setName('IPv4 address')
    'IPv4 address (C{0.0.0.0 - 255.255.255.255})'
    _ipv6_part = Regex('[0-9a-fA-F]{1,4}').setName('hex_integer')
    _full_ipv6_address = (_ipv6_part + (':' + _ipv6_part) * 7).setName('full IPv6 address')
    _short_ipv6_address = (Optional(_ipv6_part + (':' + _ipv6_part) * (0, 6)) + '::' + Optional(_ipv6_part + (':' + _ipv6_part) * (0, 6))).setName('short IPv6 address')
    _short_ipv6_address.addCondition(lambda t: sum((1 for tt in t if pyparsing_common._ipv6_part.matches(tt))) < 8)
    _mixed_ipv6_address = ('::ffff:' + ipv4_address).setName('mixed IPv6 address')
    ipv6_address = Combine((_full_ipv6_address | _mixed_ipv6_address | _short_ipv6_address).setName('IPv6 address')).setName('IPv6 address')
    'IPv6 address (long, short, or mixed form)'
    mac_address = Regex('[0-9a-fA-F]{2}([:.-])[0-9a-fA-F]{2}(?:\\1[0-9a-fA-F]{2}){4}').setName('MAC address')
    "MAC address xx:xx:xx:xx:xx (may also have '-' or '.' delimiters)"

    @staticmethod
    def convertToDate(fmt='%Y-%m-%d'):
        if False:
            i = 10
            return i + 15
        '\n        Helper to create a parse action for converting parsed date string to Python datetime.date\n\n        Params -\n         - fmt - format to be passed to datetime.strptime (default=C{"%Y-%m-%d"})\n\n        Example::\n            date_expr = pyparsing_common.iso8601_date.copy()\n            date_expr.setParseAction(pyparsing_common.convertToDate())\n            print(date_expr.parseString("1999-12-31"))\n        prints::\n            [datetime.date(1999, 12, 31)]\n        '

        def cvt_fn(s, l, t):
            if False:
                while True:
                    i = 10
            try:
                return datetime.strptime(t[0], fmt).date()
            except ValueError as ve:
                raise ParseException(s, l, str(ve))
        return cvt_fn

    @staticmethod
    def convertToDatetime(fmt='%Y-%m-%dT%H:%M:%S.%f'):
        if False:
            print('Hello World!')
        '\n        Helper to create a parse action for converting parsed datetime string to Python datetime.datetime\n\n        Params -\n         - fmt - format to be passed to datetime.strptime (default=C{"%Y-%m-%dT%H:%M:%S.%f"})\n\n        Example::\n            dt_expr = pyparsing_common.iso8601_datetime.copy()\n            dt_expr.setParseAction(pyparsing_common.convertToDatetime())\n            print(dt_expr.parseString("1999-12-31T23:59:59.999"))\n        prints::\n            [datetime.datetime(1999, 12, 31, 23, 59, 59, 999000)]\n        '

        def cvt_fn(s, l, t):
            if False:
                print('Hello World!')
            try:
                return datetime.strptime(t[0], fmt)
            except ValueError as ve:
                raise ParseException(s, l, str(ve))
        return cvt_fn
    iso8601_date = Regex('(?P<year>\\d{4})(?:-(?P<month>\\d\\d)(?:-(?P<day>\\d\\d))?)?').setName('ISO8601 date')
    'ISO8601 date (C{yyyy-mm-dd})'
    iso8601_datetime = Regex('(?P<year>\\d{4})-(?P<month>\\d\\d)-(?P<day>\\d\\d)[T ](?P<hour>\\d\\d):(?P<minute>\\d\\d)(:(?P<second>\\d\\d(\\.\\d*)?)?)?(?P<tz>Z|[+-]\\d\\d:?\\d\\d)?').setName('ISO8601 datetime')
    "ISO8601 datetime (C{yyyy-mm-ddThh:mm:ss.s(Z|+-00:00)}) - trailing seconds, milliseconds, and timezone optional; accepts separating C{'T'} or C{' '}"
    uuid = Regex('[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}').setName('UUID')
    'UUID (C{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx})'
    _html_stripper = anyOpenTag.suppress() | anyCloseTag.suppress()

    @staticmethod
    def stripHTMLTags(s, l, tokens):
        if False:
            i = 10
            return i + 15
        '\n        Parse action to remove HTML tags from web page HTML source\n\n        Example::\n            # strip HTML links from normal text \n            text = \'<td>More info at the <a href="http://pyparsing.wikispaces.com">pyparsing</a> wiki page</td>\'\n            td,td_end = makeHTMLTags("TD")\n            table_text = td + SkipTo(td_end).setParseAction(pyparsing_common.stripHTMLTags)("body") + td_end\n            \n            print(table_text.parseString(text).body) # -> \'More info at the pyparsing wiki page\'\n        '
        return pyparsing_common._html_stripper.transformString(tokens[0])
    _commasepitem = Combine(OneOrMore(~Literal(',') + ~LineEnd() + Word(printables, excludeChars=',') + Optional(White(' \t')))).streamline().setName('commaItem')
    comma_separated_list = delimitedList(Optional(quotedString.copy() | _commasepitem, default='')).setName('comma separated list')
    'Predefined expression of 1 or more printable words or quoted strings, separated by commas.'
    upcaseTokens = staticmethod(tokenMap(lambda t: _ustr(t).upper()))
    'Parse action to convert tokens to upper case.'
    downcaseTokens = staticmethod(tokenMap(lambda t: _ustr(t).lower()))
    'Parse action to convert tokens to lower case.'
if __name__ == '__main__':
    selectToken = CaselessLiteral('select')
    fromToken = CaselessLiteral('from')
    ident = Word(alphas, alphanums + '_$')
    columnName = delimitedList(ident, '.', combine=True).setParseAction(upcaseTokens)
    columnNameList = Group(delimitedList(columnName)).setName('columns')
    columnSpec = '*' | columnNameList
    tableName = delimitedList(ident, '.', combine=True).setParseAction(upcaseTokens)
    tableNameList = Group(delimitedList(tableName)).setName('tables')
    simpleSQL = selectToken('command') + columnSpec('columns') + fromToken + tableNameList('tables')
    simpleSQL.runTests('\n        # \'*\' as column list and dotted table name\n        select * from SYS.XYZZY\n\n        # caseless match on "SELECT", and casts back to "select"\n        SELECT * from XYZZY, ABC\n\n        # list of column names, and mixed case SELECT keyword\n        Select AA,BB,CC from Sys.dual\n\n        # multiple tables\n        Select A, B, C from Sys.dual, Table2\n\n        # invalid SELECT keyword - should fail\n        Xelect A, B, C from Sys.dual\n\n        # incomplete command - should fail\n        Select\n\n        # invalid column name - should fail\n        Select ^^^ frox Sys.dual\n\n        ')
    pyparsing_common.number.runTests('\n        100\n        -100\n        +100\n        3.14159\n        6.02e23\n        1e-12\n        ')
    pyparsing_common.fnumber.runTests('\n        100\n        -100\n        +100\n        3.14159\n        6.02e23\n        1e-12\n        ')
    pyparsing_common.hex_integer.runTests('\n        100\n        FF\n        ')
    import uuid
    pyparsing_common.uuid.setParseAction(tokenMap(uuid.UUID))
    pyparsing_common.uuid.runTests('\n        12345678-1234-5678-1234-567812345678\n        ')