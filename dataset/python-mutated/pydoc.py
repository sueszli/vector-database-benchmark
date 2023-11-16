"""Generate Python documentation in HTML or text for interactive use.

At the Python interactive prompt, calling help(thing) on a Python object
documents the object, and calling help() starts up an interactive
help session.

Or, at the shell command line outside of Python:

Run "pydoc <name>" to show documentation on something.  <name> may be
the name of a function, module, package, or a dotted reference to a
class or function within a module or module in a package.  If the
argument contains a path segment delimiter (e.g. slash on Unix,
backslash on Windows) it is treated as the path to a Python source file.

Run "pydoc -k <keyword>" to search for a keyword in the synopsis lines
of all available modules.

Run "pydoc -n <hostname>" to start an HTTP server with the given
hostname (default: localhost) on the local machine.

Run "pydoc -p <port>" to start an HTTP server on the given port on the
local machine.  Port number 0 can be used to get an arbitrary unused port.

Run "pydoc -b" to start an HTTP server on an arbitrary unused port and
open a web browser to interactively browse documentation.  Combine with
the -n and -p options to control the hostname and port used.

Run "pydoc -w <name>" to write out the HTML documentation for a module
to a file named "<name>.html".

Module docs for core modules are assumed to be in

    https://docs.python.org/X.Y/library/

This can be overridden by setting the PYTHONDOCS environment variable
to a different URL or to a local directory containing the Library
Reference Manual pages.
"""
__all__ = ['help']
__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__date__ = '26 February 2001'
__credits__ = 'Guido van Rossum, for an excellent programming language.\nTommy Burnette, the original creator of manpy.\nPaul Prescod, for all his work on onlinehelp.\nRichard Chamberlain, for the first implementation of textdoc.\n'
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import types
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only

def pathdirs():
    if False:
        while True:
            i = 10
    'Convert sys.path into a list of absolute, existing, unique paths.'
    dirs = []
    normdirs = []
    for dir in sys.path:
        dir = os.path.abspath(dir or '.')
        normdir = os.path.normcase(dir)
        if normdir not in normdirs and os.path.isdir(dir):
            dirs.append(dir)
            normdirs.append(normdir)
    return dirs

def _isclass(object):
    if False:
        i = 10
        return i + 15
    return inspect.isclass(object) and (not isinstance(object, types.GenericAlias))

def _findclass(func):
    if False:
        for i in range(10):
            print('nop')
    cls = sys.modules.get(func.__module__)
    if cls is None:
        return None
    for name in func.__qualname__.split('.')[:-1]:
        cls = getattr(cls, name)
    if not _isclass(cls):
        return None
    return cls

def _finddoc(obj):
    if False:
        i = 10
        return i + 15
    if inspect.ismethod(obj):
        name = obj.__func__.__name__
        self = obj.__self__
        if _isclass(self) and getattr(getattr(self, name, None), '__func__') is obj.__func__:
            cls = self
        else:
            cls = self.__class__
    elif inspect.isfunction(obj):
        name = obj.__name__
        cls = _findclass(obj)
        if cls is None or getattr(cls, name) is not obj:
            return None
    elif inspect.isbuiltin(obj):
        name = obj.__name__
        self = obj.__self__
        if _isclass(self) and self.__qualname__ + '.' + name == obj.__qualname__:
            cls = self
        else:
            cls = self.__class__
    elif isinstance(obj, property):
        func = obj.fget
        name = func.__name__
        cls = _findclass(func)
        if cls is None or getattr(cls, name) is not obj:
            return None
    elif inspect.ismethoddescriptor(obj) or inspect.isdatadescriptor(obj):
        name = obj.__name__
        cls = obj.__objclass__
        if getattr(cls, name) is not obj:
            return None
        if inspect.ismemberdescriptor(obj):
            slots = getattr(cls, '__slots__', None)
            if isinstance(slots, dict) and name in slots:
                return slots[name]
    else:
        return None
    for base in cls.__mro__:
        try:
            doc = _getowndoc(getattr(base, name))
        except AttributeError:
            continue
        if doc is not None:
            return doc
    return None

def _getowndoc(obj):
    if False:
        i = 10
        return i + 15
    'Get the documentation string for an object if it is not\n    inherited from its class.'
    try:
        doc = object.__getattribute__(obj, '__doc__')
        if doc is None:
            return None
        if obj is not type:
            typedoc = type(obj).__doc__
            if isinstance(typedoc, str) and typedoc == doc:
                return None
        return doc
    except AttributeError:
        return None

def _getdoc(object):
    if False:
        while True:
            i = 10
    'Get the documentation string for an object.\n\n    All tabs are expanded to spaces.  To clean up docstrings that are\n    indented to line up with blocks of code, any whitespace than can be\n    uniformly removed from the second line onwards is removed.'
    doc = _getowndoc(object)
    if doc is None:
        try:
            doc = _finddoc(object)
        except (AttributeError, TypeError):
            return None
    if not isinstance(doc, str):
        return None
    return inspect.cleandoc(doc)

def getdoc(object):
    if False:
        print('Hello World!')
    'Get the doc string or comments for an object.'
    result = _getdoc(object) or inspect.getcomments(object)
    return result and re.sub('^ *\n', '', result.rstrip()) or ''

def splitdoc(doc):
    if False:
        while True:
            i = 10
    'Split a doc string into a synopsis line (if any) and the rest.'
    lines = doc.strip().split('\n')
    if len(lines) == 1:
        return (lines[0], '')
    elif len(lines) >= 2 and (not lines[1].rstrip()):
        return (lines[0], '\n'.join(lines[2:]))
    return ('', '\n'.join(lines))

def classname(object, modname):
    if False:
        while True:
            i = 10
    'Get a class name and qualify it with a module name if necessary.'
    name = object.__name__
    if object.__module__ != modname:
        name = object.__module__ + '.' + name
    return name

def isdata(object):
    if False:
        return 10
    "Check if an object is of a type that probably means it's data."
    return not (inspect.ismodule(object) or _isclass(object) or inspect.isroutine(object) or inspect.isframe(object) or inspect.istraceback(object) or inspect.iscode(object))

def replace(text, *pairs):
    if False:
        while True:
            i = 10
    'Do a series of global replacements on a string.'
    while pairs:
        text = pairs[1].join(text.split(pairs[0]))
        pairs = pairs[2:]
    return text

def cram(text, maxlen):
    if False:
        while True:
            i = 10
    'Omit part of a string if needed to make it fit in a maximum length.'
    if len(text) > maxlen:
        pre = max(0, (maxlen - 3) // 2)
        post = max(0, maxlen - 3 - pre)
        return text[:pre] + '...' + text[len(text) - post:]
    return text
_re_stripid = re.compile(' at 0x[0-9a-f]{6,16}(>+)$', re.IGNORECASE)

def stripid(text):
    if False:
        print('Hello World!')
    'Remove the hexadecimal id from a Python object representation.'
    return _re_stripid.sub('\\1', text)

def _is_bound_method(fn):
    if False:
        while True:
            i = 10
    '\n    Returns True if fn is a bound method, regardless of whether\n    fn was implemented in Python or in C.\n    '
    if inspect.ismethod(fn):
        return True
    if inspect.isbuiltin(fn):
        self = getattr(fn, '__self__', None)
        return not (inspect.ismodule(self) or self is None)
    return False

def allmethods(cl):
    if False:
        while True:
            i = 10
    methods = {}
    for (key, value) in inspect.getmembers(cl, inspect.isroutine):
        methods[key] = 1
    for base in cl.__bases__:
        methods.update(allmethods(base))
    for key in methods.keys():
        methods[key] = getattr(cl, key)
    return methods

def _split_list(s, predicate):
    if False:
        while True:
            i = 10
    'Split sequence s via predicate, and return pair ([true], [false]).\n\n    The return value is a 2-tuple of lists,\n        ([x for x in s if predicate(x)],\n         [x for x in s if not predicate(x)])\n    '
    yes = []
    no = []
    for x in s:
        if predicate(x):
            yes.append(x)
        else:
            no.append(x)
    return (yes, no)

def visiblename(name, all=None, obj=None):
    if False:
        i = 10
        return i + 15
    'Decide whether to show documentation on a variable.'
    if name in {'__author__', '__builtins__', '__cached__', '__credits__', '__date__', '__doc__', '__file__', '__spec__', '__loader__', '__module__', '__name__', '__package__', '__path__', '__qualname__', '__slots__', '__version__'}:
        return 0
    if name.startswith('__') and name.endswith('__'):
        return 1
    if name.startswith('_') and hasattr(obj, '_fields'):
        return True
    if all is not None:
        return name in all
    else:
        return not name.startswith('_')

def classify_class_attrs(object):
    if False:
        i = 10
        return i + 15
    'Wrap inspect.classify_class_attrs, with fixup for data descriptors.'
    results = []
    for (name, kind, cls, value) in inspect.classify_class_attrs(object):
        if inspect.isdatadescriptor(value):
            kind = 'data descriptor'
            if isinstance(value, property) and value.fset is None:
                kind = 'readonly property'
        results.append((name, kind, cls, value))
    return results

def sort_attributes(attrs, object):
    if False:
        return 10
    'Sort the attrs list in-place by _fields and then alphabetically by name'
    fields = getattr(object, '_fields', [])
    try:
        field_order = {name: i - len(fields) for (i, name) in enumerate(fields)}
    except TypeError:
        field_order = {}
    keyfunc = lambda attr: (field_order.get(attr[0], 0), attr[0])
    attrs.sort(key=keyfunc)

def ispackage(path):
    if False:
        return 10
    'Guess whether a path refers to a package directory.'
    if os.path.isdir(path):
        for ext in ('.py', '.pyc'):
            if os.path.isfile(os.path.join(path, '__init__' + ext)):
                return True
    return False

def source_synopsis(file):
    if False:
        while True:
            i = 10
    line = file.readline()
    while line[:1] == '#' or not line.strip():
        line = file.readline()
        if not line:
            break
    line = line.strip()
    if line[:4] == 'r"""':
        line = line[1:]
    if line[:3] == '"""':
        line = line[3:]
        if line[-1:] == '\\':
            line = line[:-1]
        while not line.strip():
            line = file.readline()
            if not line:
                break
        result = line.split('"""')[0].strip()
    else:
        result = None
    return result

def synopsis(filename, cache={}):
    if False:
        i = 10
        return i + 15
    'Get the one-line summary out of a module file.'
    mtime = os.stat(filename).st_mtime
    (lastupdate, result) = cache.get(filename, (None, None))
    if lastupdate is None or lastupdate < mtime:
        if filename.endswith(tuple(importlib.machinery.BYTECODE_SUFFIXES)):
            loader_cls = importlib.machinery.SourcelessFileLoader
        elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            loader_cls = importlib.machinery.ExtensionFileLoader
        else:
            loader_cls = None
        if loader_cls is None:
            try:
                file = tokenize.open(filename)
            except OSError:
                return None
            with file:
                result = source_synopsis(file)
        else:
            loader = loader_cls('__temp__', filename)
            spec = importlib.util.spec_from_file_location('__temp__', filename, loader=loader)
            try:
                module = importlib._bootstrap._load(spec)
            except:
                return None
            del sys.modules['__temp__']
            result = module.__doc__.splitlines()[0] if module.__doc__ else None
        cache[filename] = (mtime, result)
    return result

class ErrorDuringImport(Exception):
    """Errors that occurred while trying to import something to document it."""

    def __init__(self, filename, exc_info):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        (self.exc, self.value, self.tb) = exc_info

    def __str__(self):
        if False:
            while True:
                i = 10
        exc = self.exc.__name__
        return 'problem in %s - %s: %s' % (self.filename, exc, self.value)

def importfile(path):
    if False:
        for i in range(10):
            print('nop')
    'Import a Python source file or compiled file given its path.'
    magic = importlib.util.MAGIC_NUMBER
    with open(path, 'rb') as file:
        is_bytecode = magic == file.read(len(magic))
    filename = os.path.basename(path)
    (name, ext) = os.path.splitext(filename)
    if is_bytecode:
        loader = importlib._bootstrap_external.SourcelessFileLoader(name, path)
    else:
        loader = importlib._bootstrap_external.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    try:
        return importlib._bootstrap._load(spec)
    except:
        raise ErrorDuringImport(path, sys.exc_info())

def safeimport(path, forceload=0, cache={}):
    if False:
        print('Hello World!')
    "Import a module; handle errors; return None if the module isn't found.\n\n    If the module *is* found but an exception occurs, it's wrapped in an\n    ErrorDuringImport exception and reraised.  Unlike __import__, if a\n    package path is specified, the module at the end of the path is returned,\n    not the package at the beginning.  If the optional 'forceload' argument\n    is 1, we reload the module from disk (unless it's a dynamic extension)."
    try:
        if forceload and path in sys.modules:
            if path not in sys.builtin_module_names:
                subs = [m for m in sys.modules if m.startswith(path + '.')]
                for key in [path] + subs:
                    cache[key] = sys.modules[key]
                    del sys.modules[key]
        module = __import__(path)
    except:
        (exc, value, tb) = info = sys.exc_info()
        if path in sys.modules:
            raise ErrorDuringImport(sys.modules[path].__file__, info)
        elif exc is SyntaxError:
            raise ErrorDuringImport(value.filename, info)
        elif issubclass(exc, ImportError) and value.name == path:
            return None
        else:
            raise ErrorDuringImport(path, sys.exc_info())
    for part in path.split('.')[1:]:
        try:
            module = getattr(module, part)
        except AttributeError:
            return None
    return module

class Doc:
    PYTHONDOCS = os.environ.get('PYTHONDOCS', 'https://docs.python.org/%d.%d/library' % sys.version_info[:2])

    def document(self, object, name=None, *args):
        if False:
            for i in range(10):
                print('nop')
        'Generate documentation for an object.'
        args = (object, name) + args
        try:
            if inspect.ismodule(object):
                return self.docmodule(*args)
            if _isclass(object):
                return self.docclass(*args)
            if inspect.isroutine(object):
                return self.docroutine(*args)
        except AttributeError:
            pass
        if inspect.isdatadescriptor(object):
            return self.docdata(*args)
        return self.docother(*args)

    def fail(self, object, name=None, *args):
        if False:
            return 10
        'Raise an exception for unimplemented types.'
        message = "don't know how to document object%s of type %s" % (name and ' ' + repr(name), type(object).__name__)
        raise TypeError(message)
    docmodule = docclass = docroutine = docother = docproperty = docdata = fail

    def getdocloc(self, object, basedir=sysconfig.get_path('stdlib')):
        if False:
            for i in range(10):
                print('nop')
        'Return the location of module docs or None'
        try:
            file = inspect.getabsfile(object)
        except TypeError:
            file = '(built-in)'
        docloc = os.environ.get('PYTHONDOCS', self.PYTHONDOCS)
        basedir = os.path.normcase(basedir)
        if isinstance(object, type(os)) and (object.__name__ in ('errno', 'exceptions', 'gc', 'imp', 'marshal', 'posix', 'signal', 'sys', '_thread', 'zipimport') or (file.startswith(basedir) and (not file.startswith(os.path.join(basedir, 'site-packages'))))) and (object.__name__ not in ('xml.etree', 'test.pydoc_mod')):
            if docloc.startswith(('http://', 'https://')):
                docloc = '{}/{}.html'.format(docloc.rstrip('/'), object.__name__.lower())
            else:
                docloc = os.path.join(docloc, object.__name__.lower() + '.html')
        else:
            docloc = None
        return docloc

class HTMLRepr(Repr):
    """Class for safely making an HTML representation of a Python object."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        Repr.__init__(self)
        self.maxlist = self.maxtuple = 20
        self.maxdict = 10
        self.maxstring = self.maxother = 100

    def escape(self, text):
        if False:
            for i in range(10):
                print('nop')
        return replace(text, '&', '&amp;', '<', '&lt;', '>', '&gt;')

    def repr(self, object):
        if False:
            return 10
        return Repr.repr(self, object)

    def repr1(self, x, level):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(type(x), '__name__'):
            methodname = 'repr_' + '_'.join(type(x).__name__.split())
            if hasattr(self, methodname):
                return getattr(self, methodname)(x, level)
        return self.escape(cram(stripid(repr(x)), self.maxother))

    def repr_string(self, x, level):
        if False:
            for i in range(10):
                print('nop')
        test = cram(x, self.maxstring)
        testrepr = repr(test)
        if '\\' in test and '\\' not in replace(testrepr, '\\\\', ''):
            return 'r' + testrepr[0] + self.escape(test) + testrepr[0]
        return re.sub('((\\\\[\\\\abfnrtv\\\'"]|\\\\[0-9]..|\\\\x..|\\\\u....)+)', '<font color="#c040c0">\\1</font>', self.escape(testrepr))
    repr_str = repr_string

    def repr_instance(self, x, level):
        if False:
            print('Hello World!')
        try:
            return self.escape(cram(stripid(repr(x)), self.maxstring))
        except:
            return self.escape('<%s instance>' % x.__class__.__name__)
    repr_unicode = repr_string

class HTMLDoc(Doc):
    """Formatter class for HTML documentation."""
    _repr_instance = HTMLRepr()
    repr = _repr_instance.repr
    escape = _repr_instance.escape

    def page(self, title, contents):
        if False:
            while True:
                i = 10
        'Format an HTML page.'
        return '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">\n<html><head><title>Python: %s</title>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n</head><body bgcolor="#f0f0f8">\n%s\n</body></html>' % (title, contents)

    def heading(self, title, fgcol, bgcol, extras=''):
        if False:
            print('Hello World!')
        'Format a page heading.'
        return '\n<table width="100%%" cellspacing=0 cellpadding=2 border=0 summary="heading">\n<tr bgcolor="%s">\n<td valign=bottom>&nbsp;<br>\n<font color="%s" face="helvetica, arial">&nbsp;<br>%s</font></td\n><td align=right valign=bottom\n><font color="%s" face="helvetica, arial">%s</font></td></tr></table>\n    ' % (bgcol, fgcol, title, fgcol, extras or '&nbsp;')

    def section(self, title, fgcol, bgcol, contents, width=6, prelude='', marginalia=None, gap='&nbsp;'):
        if False:
            print('Hello World!')
        'Format a section with a heading.'
        if marginalia is None:
            marginalia = '<tt>' + '&nbsp;' * width + '</tt>'
        result = '<p>\n<table width="100%%" cellspacing=0 cellpadding=2 border=0 summary="section">\n<tr bgcolor="%s">\n<td colspan=3 valign=bottom>&nbsp;<br>\n<font color="%s" face="helvetica, arial">%s</font></td></tr>\n    ' % (bgcol, fgcol, title)
        if prelude:
            result = result + '\n<tr bgcolor="%s"><td rowspan=2>%s</td>\n<td colspan=2>%s</td></tr>\n<tr><td>%s</td>' % (bgcol, marginalia, prelude, gap)
        else:
            result = result + '\n<tr><td bgcolor="%s">%s</td><td>%s</td>' % (bgcol, marginalia, gap)
        return result + '\n<td width="100%%">%s</td></tr></table>' % contents

    def bigsection(self, title, *args):
        if False:
            i = 10
            return i + 15
        'Format a section with a big heading.'
        title = '<big><strong>%s</strong></big>' % title
        return self.section(title, *args)

    def preformat(self, text):
        if False:
            print('Hello World!')
        'Format literal preformatted text.'
        text = self.escape(text.expandtabs())
        return replace(text, '\n\n', '\n \n', '\n\n', '\n \n', ' ', '&nbsp;', '\n', '<br>\n')

    def multicolumn(self, list, format, cols=4):
        if False:
            return 10
        'Format a list of items into a multi-column list.'
        result = ''
        rows = (len(list) + cols - 1) // cols
        for col in range(cols):
            result = result + '<td width="%d%%" valign=top>' % (100 // cols)
            for i in range(rows * col, rows * col + rows):
                if i < len(list):
                    result = result + format(list[i]) + '<br>\n'
            result = result + '</td>'
        return '<table width="100%%" summary="list"><tr>%s</tr></table>' % result

    def grey(self, text):
        if False:
            print('Hello World!')
        return '<font color="#909090">%s</font>' % text

    def namelink(self, name, *dicts):
        if False:
            for i in range(10):
                print('nop')
        'Make a link for an identifier, given name-to-URL mappings.'
        for dict in dicts:
            if name in dict:
                return '<a href="%s">%s</a>' % (dict[name], name)
        return name

    def classlink(self, object, modname):
        if False:
            i = 10
            return i + 15
        'Make a link for a class.'
        (name, module) = (object.__name__, sys.modules.get(object.__module__))
        if hasattr(module, name) and getattr(module, name) is object:
            return '<a href="%s.html#%s">%s</a>' % (module.__name__, name, classname(object, modname))
        return classname(object, modname)

    def modulelink(self, object):
        if False:
            i = 10
            return i + 15
        'Make a link for a module.'
        return '<a href="%s.html">%s</a>' % (object.__name__, object.__name__)

    def modpkglink(self, modpkginfo):
        if False:
            return 10
        'Make a link for a module or package to display in an index.'
        (name, path, ispackage, shadowed) = modpkginfo
        if shadowed:
            return self.grey(name)
        if path:
            url = '%s.%s.html' % (path, name)
        else:
            url = '%s.html' % name
        if ispackage:
            text = '<strong>%s</strong>&nbsp;(package)' % name
        else:
            text = name
        return '<a href="%s">%s</a>' % (url, text)

    def filelink(self, url, path):
        if False:
            for i in range(10):
                print('nop')
        'Make a link to source file.'
        return '<a href="file:%s">%s</a>' % (url, path)

    def markup(self, text, escape=None, funcs={}, classes={}, methods={}):
        if False:
            return 10
        'Mark up some plain text, given a context of symbols to look for.\n        Each context dictionary maps object names to anchor names.'
        escape = escape or self.escape
        results = []
        here = 0
        pattern = re.compile('\\b((http|https|ftp)://\\S+[\\w/]|RFC[- ]?(\\d+)|PEP[- ]?(\\d+)|(self\\.)?(\\w+))')
        while True:
            match = pattern.search(text, here)
            if not match:
                break
            (start, end) = match.span()
            results.append(escape(text[here:start]))
            (all, scheme, rfc, pep, selfdot, name) = match.groups()
            if scheme:
                url = escape(all).replace('"', '&quot;')
                results.append('<a href="%s">%s</a>' % (url, url))
            elif rfc:
                url = 'http://www.rfc-editor.org/rfc/rfc%d.txt' % int(rfc)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif pep:
                url = 'https://www.python.org/dev/peps/pep-%04d/' % int(pep)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif selfdot:
                if text[end:end + 1] == '(':
                    results.append('self.' + self.namelink(name, methods))
                else:
                    results.append('self.<strong>%s</strong>' % name)
            elif text[end:end + 1] == '(':
                results.append(self.namelink(name, methods, funcs, classes))
            else:
                results.append(self.namelink(name, classes))
            here = end
        results.append(escape(text[here:]))
        return ''.join(results)

    def formattree(self, tree, modname, parent=None):
        if False:
            for i in range(10):
                print('nop')
        'Produce HTML for a class tree as given by inspect.getclasstree().'
        result = ''
        for entry in tree:
            if type(entry) is type(()):
                (c, bases) = entry
                result = result + '<dt><font face="helvetica, arial">'
                result = result + self.classlink(c, modname)
                if bases and bases != (parent,):
                    parents = []
                    for base in bases:
                        parents.append(self.classlink(base, modname))
                    result = result + '(' + ', '.join(parents) + ')'
                result = result + '\n</font></dt>'
            elif type(entry) is type([]):
                result = result + '<dd>\n%s</dd>\n' % self.formattree(entry, modname, c)
        return '<dl>\n%s</dl>\n' % result

    def docmodule(self, object, name=None, mod=None, *ignored):
        if False:
            print('Hello World!')
        'Produce HTML documentation for a module object.'
        name = object.__name__
        try:
            all = object.__all__
        except AttributeError:
            all = None
        parts = name.split('.')
        links = []
        for i in range(len(parts) - 1):
            links.append('<a href="%s.html"><font color="#ffffff">%s</font></a>' % ('.'.join(parts[:i + 1]), parts[i]))
        linkedname = '.'.join(links + parts[-1:])
        head = '<big><big><strong>%s</strong></big></big>' % linkedname
        try:
            path = inspect.getabsfile(object)
            url = urllib.parse.quote(path)
            filelink = self.filelink(url, path)
        except TypeError:
            filelink = '(built-in)'
        info = []
        if hasattr(object, '__version__'):
            version = str(object.__version__)
            if version[:11] == '$' + 'Revision: ' and version[-1:] == '$':
                version = version[11:-1].strip()
            info.append('version %s' % self.escape(version))
        if hasattr(object, '__date__'):
            info.append(self.escape(str(object.__date__)))
        if info:
            head = head + ' (%s)' % ', '.join(info)
        docloc = self.getdocloc(object)
        if docloc is not None:
            docloc = '<br><a href="%(docloc)s">Module Reference</a>' % locals()
        else:
            docloc = ''
        result = self.heading(head, '#ffffff', '#7799ee', '<a href=".">index</a><br>' + filelink + docloc)
        modules = inspect.getmembers(object, inspect.ismodule)
        (classes, cdict) = ([], {})
        for (key, value) in inspect.getmembers(object, _isclass):
            if all is not None or (inspect.getmodule(value) or object) is object:
                if visiblename(key, all, object):
                    classes.append((key, value))
                    cdict[key] = cdict[value] = '#' + key
        for (key, value) in classes:
            for base in value.__bases__:
                (key, modname) = (base.__name__, base.__module__)
                module = sys.modules.get(modname)
                if modname != name and module and hasattr(module, key):
                    if getattr(module, key) is base:
                        if not key in cdict:
                            cdict[key] = cdict[base] = modname + '.html#' + key
        (funcs, fdict) = ([], {})
        for (key, value) in inspect.getmembers(object, inspect.isroutine):
            if all is not None or inspect.isbuiltin(value) or inspect.getmodule(value) is object:
                if visiblename(key, all, object):
                    funcs.append((key, value))
                    fdict[key] = '#-' + key
                    if inspect.isfunction(value):
                        fdict[value] = fdict[key]
        data = []
        for (key, value) in inspect.getmembers(object, isdata):
            if visiblename(key, all, object):
                data.append((key, value))
        doc = self.markup(getdoc(object), self.preformat, fdict, cdict)
        doc = doc and '<tt>%s</tt>' % doc
        result = result + '<p>%s</p>\n' % doc
        if hasattr(object, '__path__'):
            modpkgs = []
            for (importer, modname, ispkg) in pkgutil.iter_modules(object.__path__):
                modpkgs.append((modname, name, ispkg, 0))
            modpkgs.sort()
            contents = self.multicolumn(modpkgs, self.modpkglink)
            result = result + self.bigsection('Package Contents', '#ffffff', '#aa55cc', contents)
        elif modules:
            contents = self.multicolumn(modules, lambda t: self.modulelink(t[1]))
            result = result + self.bigsection('Modules', '#ffffff', '#aa55cc', contents)
        if classes:
            classlist = [value for (key, value) in classes]
            contents = [self.formattree(inspect.getclasstree(classlist, 1), name)]
            for (key, value) in classes:
                contents.append(self.document(value, key, name, fdict, cdict))
            result = result + self.bigsection('Classes', '#ffffff', '#ee77aa', ' '.join(contents))
        if funcs:
            contents = []
            for (key, value) in funcs:
                contents.append(self.document(value, key, name, fdict, cdict))
            result = result + self.bigsection('Functions', '#ffffff', '#eeaa77', ' '.join(contents))
        if data:
            contents = []
            for (key, value) in data:
                contents.append(self.document(value, key))
            result = result + self.bigsection('Data', '#ffffff', '#55aa55', '<br>\n'.join(contents))
        if hasattr(object, '__author__'):
            contents = self.markup(str(object.__author__), self.preformat)
            result = result + self.bigsection('Author', '#ffffff', '#7799ee', contents)
        if hasattr(object, '__credits__'):
            contents = self.markup(str(object.__credits__), self.preformat)
            result = result + self.bigsection('Credits', '#ffffff', '#7799ee', contents)
        return result

    def docclass(self, object, name=None, mod=None, funcs={}, classes={}, *ignored):
        if False:
            i = 10
            return i + 15
        'Produce HTML documentation for a class object.'
        realname = object.__name__
        name = name or realname
        bases = object.__bases__
        contents = []
        push = contents.append

        class HorizontalRule:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.needone = 0

            def maybe(self):
                if False:
                    while True:
                        i = 10
                if self.needone:
                    push('<hr>\n')
                self.needone = 1
        hr = HorizontalRule()
        mro = deque(inspect.getmro(object))
        if len(mro) > 2:
            hr.maybe()
            push('<dl><dt>Method resolution order:</dt>\n')
            for base in mro:
                push('<dd>%s</dd>\n' % self.classlink(base, object.__module__))
            push('</dl>\n')

        def spill(msg, attrs, predicate):
            if False:
                print('Hello World!')
            (ok, attrs) = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for (name, kind, homecls, value) in ok:
                    try:
                        value = getattr(object, name)
                    except Exception:
                        push(self.docdata(value, name, mod))
                    else:
                        push(self.document(value, name, mod, funcs, classes, mdict, object))
                    push('\n')
            return attrs

        def spilldescriptors(msg, attrs, predicate):
            if False:
                return 10
            (ok, attrs) = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for (name, kind, homecls, value) in ok:
                    push(self.docdata(value, name, mod))
            return attrs

        def spilldata(msg, attrs, predicate):
            if False:
                while True:
                    i = 10
            (ok, attrs) = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for (name, kind, homecls, value) in ok:
                    base = self.docother(getattr(object, name), name, mod)
                    doc = getdoc(value)
                    if not doc:
                        push('<dl><dt>%s</dl>\n' % base)
                    else:
                        doc = self.markup(getdoc(value), self.preformat, funcs, classes, mdict)
                        doc = '<dd><tt>%s</tt>' % doc
                        push('<dl><dt>%s%s</dl>\n' % (base, doc))
                    push('\n')
            return attrs
        attrs = [(name, kind, cls, value) for (name, kind, cls, value) in classify_class_attrs(object) if visiblename(name, obj=object)]
        mdict = {}
        for (key, kind, homecls, value) in attrs:
            mdict[key] = anchor = '#' + name + '-' + key
            try:
                value = getattr(object, name)
            except Exception:
                pass
            try:
                mdict[value] = anchor
            except TypeError:
                pass
        while attrs:
            if mro:
                thisclass = mro.popleft()
            else:
                thisclass = attrs[0][2]
            (attrs, inherited) = _split_list(attrs, lambda t: t[2] is thisclass)
            if object is not builtins.object and thisclass is builtins.object:
                attrs = inherited
                continue
            elif thisclass is object:
                tag = 'defined here'
            else:
                tag = 'inherited from %s' % self.classlink(thisclass, object.__module__)
            tag += ':<br>\n'
            sort_attributes(attrs, object)
            attrs = spill('Methods %s' % tag, attrs, lambda t: t[1] == 'method')
            attrs = spill('Class methods %s' % tag, attrs, lambda t: t[1] == 'class method')
            attrs = spill('Static methods %s' % tag, attrs, lambda t: t[1] == 'static method')
            attrs = spilldescriptors('Readonly properties %s' % tag, attrs, lambda t: t[1] == 'readonly property')
            attrs = spilldescriptors('Data descriptors %s' % tag, attrs, lambda t: t[1] == 'data descriptor')
            attrs = spilldata('Data and other attributes %s' % tag, attrs, lambda t: t[1] == 'data')
            assert attrs == []
            attrs = inherited
        contents = ''.join(contents)
        if name == realname:
            title = '<a name="%s">class <strong>%s</strong></a>' % (name, realname)
        else:
            title = '<strong>%s</strong> = <a name="%s">class %s</a>' % (name, name, realname)
        if bases:
            parents = []
            for base in bases:
                parents.append(self.classlink(base, object.__module__))
            title = title + '(%s)' % ', '.join(parents)
        decl = ''
        try:
            signature = inspect.signature(object)
        except (ValueError, TypeError):
            signature = None
        if signature:
            argspec = str(signature)
            if argspec and argspec != '()':
                decl = name + self.escape(argspec) + '\n\n'
        doc = getdoc(object)
        if decl:
            doc = decl + (doc or '')
        doc = self.markup(doc, self.preformat, funcs, classes, mdict)
        doc = doc and '<tt>%s<br>&nbsp;</tt>' % doc
        return self.section(title, '#000000', '#ffc8d8', contents, 3, doc)

    def formatvalue(self, object):
        if False:
            return 10
        'Format an argument default value as text.'
        return self.grey('=' + self.repr(object))

    def docroutine(self, object, name=None, mod=None, funcs={}, classes={}, methods={}, cl=None):
        if False:
            print('Hello World!')
        'Produce HTML documentation for a function or method object.'
        realname = object.__name__
        name = name or realname
        anchor = (cl and cl.__name__ or '') + '-' + name
        note = ''
        skipdocs = 0
        if _is_bound_method(object):
            imclass = object.__self__.__class__
            if cl:
                if imclass is not cl:
                    note = ' from ' + self.classlink(imclass, mod)
            elif object.__self__ is not None:
                note = ' method of %s instance' % self.classlink(object.__self__.__class__, mod)
            else:
                note = ' unbound %s method' % self.classlink(imclass, mod)
        if inspect.iscoroutinefunction(object) or inspect.isasyncgenfunction(object):
            asyncqualifier = 'async '
        else:
            asyncqualifier = ''
        if name == realname:
            title = '<a name="%s"><strong>%s</strong></a>' % (anchor, realname)
        else:
            if cl and inspect.getattr_static(cl, realname, []) is object:
                reallink = '<a href="#%s">%s</a>' % (cl.__name__ + '-' + realname, realname)
                skipdocs = 1
            else:
                reallink = realname
            title = '<a name="%s"><strong>%s</strong></a> = %s' % (anchor, name, reallink)
        argspec = None
        if inspect.isroutine(object):
            try:
                signature = inspect.signature(object)
            except (ValueError, TypeError):
                signature = None
            if signature:
                argspec = str(signature)
                if realname == '<lambda>':
                    title = '<strong>%s</strong> <em>lambda</em> ' % name
                    argspec = argspec[1:-1]
        if not argspec:
            argspec = '(...)'
        decl = asyncqualifier + title + self.escape(argspec) + (note and self.grey('<font face="helvetica, arial">%s</font>' % note))
        if skipdocs:
            return '<dl><dt>%s</dt></dl>\n' % decl
        else:
            doc = self.markup(getdoc(object), self.preformat, funcs, classes, methods)
            doc = doc and '<dd><tt>%s</tt></dd>' % doc
            return '<dl><dt>%s</dt>%s</dl>\n' % (decl, doc)

    def docdata(self, object, name=None, mod=None, cl=None):
        if False:
            return 10
        'Produce html documentation for a data descriptor.'
        results = []
        push = results.append
        if name:
            push('<dl><dt><strong>%s</strong></dt>\n' % name)
        doc = self.markup(getdoc(object), self.preformat)
        if doc:
            push('<dd><tt>%s</tt></dd>\n' % doc)
        push('</dl>\n')
        return ''.join(results)
    docproperty = docdata

    def docother(self, object, name=None, mod=None, *ignored):
        if False:
            print('Hello World!')
        'Produce HTML documentation for a data object.'
        lhs = name and '<strong>%s</strong> = ' % name or ''
        return lhs + self.repr(object)

    def index(self, dir, shadowed=None):
        if False:
            return 10
        'Generate an HTML index for a directory of modules.'
        modpkgs = []
        if shadowed is None:
            shadowed = {}
        for (importer, name, ispkg) in pkgutil.iter_modules([dir]):
            if any((55296 <= ord(ch) <= 57343 for ch in name)):
                continue
            modpkgs.append((name, '', ispkg, name in shadowed))
            shadowed[name] = 1
        modpkgs.sort()
        contents = self.multicolumn(modpkgs, self.modpkglink)
        return self.bigsection(dir, '#ffffff', '#ee77aa', contents)

class TextRepr(Repr):
    """Class for safely making a text representation of a Python object."""

    def __init__(self):
        if False:
            return 10
        Repr.__init__(self)
        self.maxlist = self.maxtuple = 20
        self.maxdict = 10
        self.maxstring = self.maxother = 100

    def repr1(self, x, level):
        if False:
            return 10
        if hasattr(type(x), '__name__'):
            methodname = 'repr_' + '_'.join(type(x).__name__.split())
            if hasattr(self, methodname):
                return getattr(self, methodname)(x, level)
        return cram(stripid(repr(x)), self.maxother)

    def repr_string(self, x, level):
        if False:
            while True:
                i = 10
        test = cram(x, self.maxstring)
        testrepr = repr(test)
        if '\\' in test and '\\' not in replace(testrepr, '\\\\', ''):
            return 'r' + testrepr[0] + test + testrepr[0]
        return testrepr
    repr_str = repr_string

    def repr_instance(self, x, level):
        if False:
            for i in range(10):
                print('nop')
        try:
            return cram(stripid(repr(x)), self.maxstring)
        except:
            return '<%s instance>' % x.__class__.__name__

class TextDoc(Doc):
    """Formatter class for text documentation."""
    _repr_instance = TextRepr()
    repr = _repr_instance.repr

    def bold(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Format a string in bold by overstriking.'
        return ''.join((ch + '\x08' + ch for ch in text))

    def indent(self, text, prefix='    '):
        if False:
            return 10
        'Indent text by prepending a given prefix to each line.'
        if not text:
            return ''
        lines = [prefix + line for line in text.split('\n')]
        if lines:
            lines[-1] = lines[-1].rstrip()
        return '\n'.join(lines)

    def section(self, title, contents):
        if False:
            while True:
                i = 10
        'Format a section with a given heading.'
        clean_contents = self.indent(contents).rstrip()
        return self.bold(title) + '\n' + clean_contents + '\n\n'

    def formattree(self, tree, modname, parent=None, prefix=''):
        if False:
            print('Hello World!')
        'Render in text a class tree as returned by inspect.getclasstree().'
        result = ''
        for entry in tree:
            if type(entry) is type(()):
                (c, bases) = entry
                result = result + prefix + classname(c, modname)
                if bases and bases != (parent,):
                    parents = (classname(c, modname) for c in bases)
                    result = result + '(%s)' % ', '.join(parents)
                result = result + '\n'
            elif type(entry) is type([]):
                result = result + self.formattree(entry, modname, c, prefix + '    ')
        return result

    def docmodule(self, object, name=None, mod=None):
        if False:
            while True:
                i = 10
        'Produce text documentation for a given module object.'
        name = object.__name__
        (synop, desc) = splitdoc(getdoc(object))
        result = self.section('NAME', name + (synop and ' - ' + synop))
        all = getattr(object, '__all__', None)
        docloc = self.getdocloc(object)
        if docloc is not None:
            result = result + self.section('MODULE REFERENCE', docloc + '\n\nThe following documentation is automatically generated from the Python\nsource files.  It may be incomplete, incorrect or include features that\nare considered implementation detail and may vary between Python\nimplementations.  When in doubt, consult the module reference at the\nlocation listed above.\n')
        if desc:
            result = result + self.section('DESCRIPTION', desc)
        classes = []
        for (key, value) in inspect.getmembers(object, _isclass):
            if all is not None or (inspect.getmodule(value) or object) is object:
                if visiblename(key, all, object):
                    classes.append((key, value))
        funcs = []
        for (key, value) in inspect.getmembers(object, inspect.isroutine):
            if all is not None or inspect.isbuiltin(value) or inspect.getmodule(value) is object:
                if visiblename(key, all, object):
                    funcs.append((key, value))
        data = []
        for (key, value) in inspect.getmembers(object, isdata):
            if visiblename(key, all, object):
                data.append((key, value))
        modpkgs = []
        modpkgs_names = set()
        if hasattr(object, '__path__'):
            for (importer, modname, ispkg) in pkgutil.iter_modules(object.__path__):
                modpkgs_names.add(modname)
                if ispkg:
                    modpkgs.append(modname + ' (package)')
                else:
                    modpkgs.append(modname)
            modpkgs.sort()
            result = result + self.section('PACKAGE CONTENTS', '\n'.join(modpkgs))
        submodules = []
        for (key, value) in inspect.getmembers(object, inspect.ismodule):
            if value.__name__.startswith(name + '.') and key not in modpkgs_names:
                submodules.append(key)
        if submodules:
            submodules.sort()
            result = result + self.section('SUBMODULES', '\n'.join(submodules))
        if classes:
            classlist = [value for (key, value) in classes]
            contents = [self.formattree(inspect.getclasstree(classlist, 1), name)]
            for (key, value) in classes:
                contents.append(self.document(value, key, name))
            result = result + self.section('CLASSES', '\n'.join(contents))
        if funcs:
            contents = []
            for (key, value) in funcs:
                contents.append(self.document(value, key, name))
            result = result + self.section('FUNCTIONS', '\n'.join(contents))
        if data:
            contents = []
            for (key, value) in data:
                contents.append(self.docother(value, key, name, maxlen=70))
            result = result + self.section('DATA', '\n'.join(contents))
        if hasattr(object, '__version__'):
            version = str(object.__version__)
            if version[:11] == '$' + 'Revision: ' and version[-1:] == '$':
                version = version[11:-1].strip()
            result = result + self.section('VERSION', version)
        if hasattr(object, '__date__'):
            result = result + self.section('DATE', str(object.__date__))
        if hasattr(object, '__author__'):
            result = result + self.section('AUTHOR', str(object.__author__))
        if hasattr(object, '__credits__'):
            result = result + self.section('CREDITS', str(object.__credits__))
        try:
            file = inspect.getabsfile(object)
        except TypeError:
            file = '(built-in)'
        result = result + self.section('FILE', file)
        return result

    def docclass(self, object, name=None, mod=None, *ignored):
        if False:
            i = 10
            return i + 15
        'Produce text documentation for a given class object.'
        realname = object.__name__
        name = name or realname
        bases = object.__bases__

        def makename(c, m=object.__module__):
            if False:
                print('Hello World!')
            return classname(c, m)
        if name == realname:
            title = 'class ' + self.bold(realname)
        else:
            title = self.bold(name) + ' = class ' + realname
        if bases:
            parents = map(makename, bases)
            title = title + '(%s)' % ', '.join(parents)
        contents = []
        push = contents.append
        try:
            signature = inspect.signature(object)
        except (ValueError, TypeError):
            signature = None
        if signature:
            argspec = str(signature)
            if argspec and argspec != '()':
                push(name + argspec + '\n')
        doc = getdoc(object)
        if doc:
            push(doc + '\n')
        mro = deque(inspect.getmro(object))
        if len(mro) > 2:
            push('Method resolution order:')
            for base in mro:
                push('    ' + makename(base))
            push('')
        subclasses = sorted((str(cls.__name__) for cls in type.__subclasses__(object) if not cls.__name__.startswith('_') and cls.__module__ == 'builtins'), key=str.lower)
        no_of_subclasses = len(subclasses)
        MAX_SUBCLASSES_TO_DISPLAY = 4
        if subclasses:
            push('Built-in subclasses:')
            for subclassname in subclasses[:MAX_SUBCLASSES_TO_DISPLAY]:
                push('    ' + subclassname)
            if no_of_subclasses > MAX_SUBCLASSES_TO_DISPLAY:
                push('    ... and ' + str(no_of_subclasses - MAX_SUBCLASSES_TO_DISPLAY) + ' other subclasses')
            push('')

        class HorizontalRule:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.needone = 0

            def maybe(self):
                if False:
                    return 10
                if self.needone:
                    push('-' * 70)
                self.needone = 1
        hr = HorizontalRule()

        def spill(msg, attrs, predicate):
            if False:
                i = 10
                return i + 15
            (ok, attrs) = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for (name, kind, homecls, value) in ok:
                    try:
                        value = getattr(object, name)
                    except Exception:
                        push(self.docdata(value, name, mod))
                    else:
                        push(self.document(value, name, mod, object))
            return attrs

        def spilldescriptors(msg, attrs, predicate):
            if False:
                for i in range(10):
                    print('nop')
            (ok, attrs) = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for (name, kind, homecls, value) in ok:
                    push(self.docdata(value, name, mod))
            return attrs

        def spilldata(msg, attrs, predicate):
            if False:
                i = 10
                return i + 15
            (ok, attrs) = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for (name, kind, homecls, value) in ok:
                    doc = getdoc(value)
                    try:
                        obj = getattr(object, name)
                    except AttributeError:
                        obj = homecls.__dict__[name]
                    push(self.docother(obj, name, mod, maxlen=70, doc=doc) + '\n')
            return attrs
        attrs = [(name, kind, cls, value) for (name, kind, cls, value) in classify_class_attrs(object) if visiblename(name, obj=object)]
        while attrs:
            if mro:
                thisclass = mro.popleft()
            else:
                thisclass = attrs[0][2]
            (attrs, inherited) = _split_list(attrs, lambda t: t[2] is thisclass)
            if object is not builtins.object and thisclass is builtins.object:
                attrs = inherited
                continue
            elif thisclass is object:
                tag = 'defined here'
            else:
                tag = 'inherited from %s' % classname(thisclass, object.__module__)
            sort_attributes(attrs, object)
            attrs = spill('Methods %s:\n' % tag, attrs, lambda t: t[1] == 'method')
            attrs = spill('Class methods %s:\n' % tag, attrs, lambda t: t[1] == 'class method')
            attrs = spill('Static methods %s:\n' % tag, attrs, lambda t: t[1] == 'static method')
            attrs = spilldescriptors('Readonly properties %s:\n' % tag, attrs, lambda t: t[1] == 'readonly property')
            attrs = spilldescriptors('Data descriptors %s:\n' % tag, attrs, lambda t: t[1] == 'data descriptor')
            attrs = spilldata('Data and other attributes %s:\n' % tag, attrs, lambda t: t[1] == 'data')
            assert attrs == []
            attrs = inherited
        contents = '\n'.join(contents)
        if not contents:
            return title + '\n'
        return title + '\n' + self.indent(contents.rstrip(), ' |  ') + '\n'

    def formatvalue(self, object):
        if False:
            return 10
        'Format an argument default value as text.'
        return '=' + self.repr(object)

    def docroutine(self, object, name=None, mod=None, cl=None):
        if False:
            while True:
                i = 10
        'Produce text documentation for a function or method object.'
        realname = object.__name__
        name = name or realname
        note = ''
        skipdocs = 0
        if _is_bound_method(object):
            imclass = object.__self__.__class__
            if cl:
                if imclass is not cl:
                    note = ' from ' + classname(imclass, mod)
            elif object.__self__ is not None:
                note = ' method of %s instance' % classname(object.__self__.__class__, mod)
            else:
                note = ' unbound %s method' % classname(imclass, mod)
        if inspect.iscoroutinefunction(object) or inspect.isasyncgenfunction(object):
            asyncqualifier = 'async '
        else:
            asyncqualifier = ''
        if name == realname:
            title = self.bold(realname)
        else:
            if cl and inspect.getattr_static(cl, realname, []) is object:
                skipdocs = 1
            title = self.bold(name) + ' = ' + realname
        argspec = None
        if inspect.isroutine(object):
            try:
                signature = inspect.signature(object)
            except (ValueError, TypeError):
                signature = None
            if signature:
                argspec = str(signature)
                if realname == '<lambda>':
                    title = self.bold(name) + ' lambda '
                    argspec = argspec[1:-1]
        if not argspec:
            argspec = '(...)'
        decl = asyncqualifier + title + argspec + note
        if skipdocs:
            return decl + '\n'
        else:
            doc = getdoc(object) or ''
            return decl + '\n' + (doc and self.indent(doc).rstrip() + '\n')

    def docdata(self, object, name=None, mod=None, cl=None):
        if False:
            print('Hello World!')
        'Produce text documentation for a data descriptor.'
        results = []
        push = results.append
        if name:
            push(self.bold(name))
            push('\n')
        doc = getdoc(object) or ''
        if doc:
            push(self.indent(doc))
            push('\n')
        return ''.join(results)
    docproperty = docdata

    def docother(self, object, name=None, mod=None, parent=None, maxlen=None, doc=None):
        if False:
            return 10
        'Produce text documentation for a data object.'
        repr = self.repr(object)
        if maxlen:
            line = (name and name + ' = ' or '') + repr
            chop = maxlen - len(line)
            if chop < 0:
                repr = repr[:chop] + '...'
        line = (name and self.bold(name) + ' = ' or '') + repr
        if not doc:
            doc = getdoc(object)
        if doc:
            line += '\n' + self.indent(str(doc)) + '\n'
        return line

class _PlainTextDoc(TextDoc):
    """Subclass of TextDoc which overrides string styling"""

    def bold(self, text):
        if False:
            print('Hello World!')
        return text

def pager(text):
    if False:
        print('Hello World!')
    'The first time this is called, determine what kind of pager to use.'
    global pager
    pager = getpager()
    pager(text)

def getpager():
    if False:
        print('Hello World!')
    'Decide what method to use for paging through text.'
    if not hasattr(sys.stdin, 'isatty'):
        return plainpager
    if not hasattr(sys.stdout, 'isatty'):
        return plainpager
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return plainpager
    use_pager = os.environ.get('MANPAGER') or os.environ.get('PAGER')
    if use_pager:
        if sys.platform == 'win32':
            return lambda text: tempfilepager(plain(text), use_pager)
        elif os.environ.get('TERM') in ('dumb', 'emacs'):
            return lambda text: pipepager(plain(text), use_pager)
        else:
            return lambda text: pipepager(text, use_pager)
    if os.environ.get('TERM') in ('dumb', 'emacs'):
        return plainpager
    if sys.platform == 'win32':
        return lambda text: tempfilepager(plain(text), 'more <')
    if hasattr(os, 'system') and os.system('(less) 2>/dev/null') == 0:
        return lambda text: pipepager(text, 'less')
    import tempfile
    (fd, filename) = tempfile.mkstemp()
    os.close(fd)
    try:
        if hasattr(os, 'system') and os.system('more "%s"' % filename) == 0:
            return lambda text: pipepager(text, 'more')
        else:
            return ttypager
    finally:
        os.unlink(filename)

def plain(text):
    if False:
        while True:
            i = 10
    'Remove boldface formatting from text.'
    return re.sub('.\x08', '', text)

def pipepager(text, cmd):
    if False:
        print('Hello World!')
    'Page through text by feeding it to another program.'
    import subprocess
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, errors='backslashreplace')
    try:
        with proc.stdin as pipe:
            try:
                pipe.write(text)
            except KeyboardInterrupt:
                pass
    except OSError:
        pass
    while True:
        try:
            proc.wait()
            break
        except KeyboardInterrupt:
            pass

def tempfilepager(text, cmd):
    if False:
        while True:
            i = 10
    'Page through text by invoking a program on a temporary file.'
    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'pydoc.out')
        with open(filename, 'w', errors='backslashreplace', encoding=os.device_encoding(0) if sys.platform == 'win32' else None) as file:
            file.write(text)
        os.system(cmd + ' "' + filename + '"')

def _escape_stdout(text):
    if False:
        while True:
            i = 10
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    return text.encode(encoding, 'backslashreplace').decode(encoding)

def ttypager(text):
    if False:
        while True:
            i = 10
    'Page through text on a text terminal.'
    lines = plain(_escape_stdout(text)).split('\n')
    try:
        import tty
        fd = sys.stdin.fileno()
        old = tty.tcgetattr(fd)
        tty.setcbreak(fd)
        getchar = lambda : sys.stdin.read(1)
    except (ImportError, AttributeError, io.UnsupportedOperation):
        tty = None
        getchar = lambda : sys.stdin.readline()[:-1][:1]
    try:
        try:
            h = int(os.environ.get('LINES', 0))
        except ValueError:
            h = 0
        if h <= 1:
            h = 25
        r = inc = h - 1
        sys.stdout.write('\n'.join(lines[:inc]) + '\n')
        while lines[r:]:
            sys.stdout.write('-- more --')
            sys.stdout.flush()
            c = getchar()
            if c in ('q', 'Q'):
                sys.stdout.write('\r          \r')
                break
            elif c in ('\r', '\n'):
                sys.stdout.write('\r          \r' + lines[r] + '\n')
                r = r + 1
                continue
            if c in ('b', 'B', '\x1b'):
                r = r - inc - inc
                if r < 0:
                    r = 0
            sys.stdout.write('\n' + '\n'.join(lines[r:r + inc]) + '\n')
            r = r + inc
    finally:
        if tty:
            tty.tcsetattr(fd, tty.TCSAFLUSH, old)

def plainpager(text):
    if False:
        while True:
            i = 10
    'Simply print unformatted text.  This is the ultimate fallback.'
    sys.stdout.write(plain(_escape_stdout(text)))

def describe(thing):
    if False:
        for i in range(10):
            print('nop')
    'Produce a short description of the given thing.'
    if inspect.ismodule(thing):
        if thing.__name__ in sys.builtin_module_names:
            return 'built-in module ' + thing.__name__
        if hasattr(thing, '__path__'):
            return 'package ' + thing.__name__
        else:
            return 'module ' + thing.__name__
    if inspect.isbuiltin(thing):
        return 'built-in function ' + thing.__name__
    if inspect.isgetsetdescriptor(thing):
        return 'getset descriptor %s.%s.%s' % (thing.__objclass__.__module__, thing.__objclass__.__name__, thing.__name__)
    if inspect.ismemberdescriptor(thing):
        return 'member descriptor %s.%s.%s' % (thing.__objclass__.__module__, thing.__objclass__.__name__, thing.__name__)
    if _isclass(thing):
        return 'class ' + thing.__name__
    if inspect.isfunction(thing):
        return 'function ' + thing.__name__
    if inspect.ismethod(thing):
        return 'method ' + thing.__name__
    return type(thing).__name__

def locate(path, forceload=0):
    if False:
        for i in range(10):
            print('nop')
    'Locate an object by name or dotted path, importing as necessary.'
    parts = [part for part in path.split('.') if part]
    (module, n) = (None, 0)
    while n < len(parts):
        nextmodule = safeimport('.'.join(parts[:n + 1]), forceload)
        if nextmodule:
            (module, n) = (nextmodule, n + 1)
        else:
            break
    if module:
        object = module
    else:
        object = builtins
    for part in parts[n:]:
        try:
            object = getattr(object, part)
        except AttributeError:
            return None
    return object
text = TextDoc()
plaintext = _PlainTextDoc()
html = HTMLDoc()

def resolve(thing, forceload=0):
    if False:
        return 10
    'Given an object or a path to an object, get the object and its name.'
    if isinstance(thing, str):
        object = locate(thing, forceload)
        if object is None:
            raise ImportError('No Python documentation found for %r.\nUse help() to get the interactive help utility.\nUse help(str) for help on the str class.' % thing)
        return (object, thing)
    else:
        name = getattr(thing, '__name__', None)
        return (thing, name if isinstance(name, str) else None)

def render_doc(thing, title='Python Library Documentation: %s', forceload=0, renderer=None):
    if False:
        for i in range(10):
            print('nop')
    'Render text documentation, given an object or a path to an object.'
    if renderer is None:
        renderer = text
    (object, name) = resolve(thing, forceload)
    desc = describe(object)
    module = inspect.getmodule(object)
    if name and '.' in name:
        desc += ' in ' + name[:name.rfind('.')]
    elif module and module is not object:
        desc += ' in module ' + module.__name__
    if not (inspect.ismodule(object) or _isclass(object) or inspect.isroutine(object) or inspect.isdatadescriptor(object) or _getdoc(object)):
        if hasattr(object, '__origin__'):
            object = object.__origin__
        else:
            object = type(object)
            desc += ' object'
    return title % desc + '\n\n' + renderer.document(object, name)

def doc(thing, title='Python Library Documentation: %s', forceload=0, output=None):
    if False:
        return 10
    'Display text documentation, given an object or a path to an object.'
    try:
        if output is None:
            pager(render_doc(thing, title, forceload))
        else:
            output.write(render_doc(thing, title, forceload, plaintext))
    except (ImportError, ErrorDuringImport) as value:
        print(value)

def writedoc(thing, forceload=0):
    if False:
        return 10
    'Write HTML documentation to a file in the current directory.'
    try:
        (object, name) = resolve(thing, forceload)
        page = html.page(describe(object), html.document(object, name))
        with open(name + '.html', 'w', encoding='utf-8') as file:
            file.write(page)
        print('wrote', name + '.html')
    except (ImportError, ErrorDuringImport) as value:
        print(value)

def writedocs(dir, pkgpath='', done=None):
    if False:
        print('Hello World!')
    'Write out HTML documentation for all modules in a directory tree.'
    if done is None:
        done = {}
    for (importer, modname, ispkg) in pkgutil.walk_packages([dir], pkgpath):
        writedoc(modname)
    return

class Helper:
    keywords = {'False': '', 'None': '', 'True': '', 'and': 'BOOLEAN', 'as': 'with', 'assert': ('assert', ''), 'async': ('async', ''), 'await': ('await', ''), 'break': ('break', 'while for'), 'class': ('class', 'CLASSES SPECIALMETHODS'), 'continue': ('continue', 'while for'), 'def': ('function', ''), 'del': ('del', 'BASICMETHODS'), 'elif': 'if', 'else': ('else', 'while for'), 'except': 'try', 'finally': 'try', 'for': ('for', 'break continue while'), 'from': 'import', 'global': ('global', 'nonlocal NAMESPACES'), 'if': ('if', 'TRUTHVALUE'), 'import': ('import', 'MODULES'), 'in': ('in', 'SEQUENCEMETHODS'), 'is': 'COMPARISON', 'lambda': ('lambda', 'FUNCTIONS'), 'nonlocal': ('nonlocal', 'global NAMESPACES'), 'not': 'BOOLEAN', 'or': 'BOOLEAN', 'pass': ('pass', ''), 'raise': ('raise', 'EXCEPTIONS'), 'return': ('return', 'FUNCTIONS'), 'try': ('try', 'EXCEPTIONS'), 'while': ('while', 'break continue if TRUTHVALUE'), 'with': ('with', 'CONTEXTMANAGERS EXCEPTIONS yield'), 'yield': ('yield', '')}
    _strprefixes = [p + q for p in ('b', 'f', 'r', 'u') for q in ("'", '"')]
    _symbols_inverse = {'STRINGS': ("'", "'''", '"', '"""', *_strprefixes), 'OPERATORS': ('+', '-', '*', '**', '/', '//', '%', '<<', '>>', '&', '|', '^', '~', '<', '>', '<=', '>=', '==', '!=', '<>'), 'COMPARISON': ('<', '>', '<=', '>=', '==', '!=', '<>'), 'UNARY': ('-', '~'), 'AUGMENTEDASSIGNMENT': ('+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**=', '//='), 'BITWISE': ('<<', '>>', '&', '|', '^', '~'), 'COMPLEX': ('j', 'J')}
    symbols = {'%': 'OPERATORS FORMATTING', '**': 'POWER', ',': 'TUPLES LISTS FUNCTIONS', '.': 'ATTRIBUTES FLOAT MODULES OBJECTS', '...': 'ELLIPSIS', ':': 'SLICINGS DICTIONARYLITERALS', '@': 'def class', '\\': 'STRINGS', '_': 'PRIVATENAMES', '__': 'PRIVATENAMES SPECIALMETHODS', '`': 'BACKQUOTES', '(': 'TUPLES FUNCTIONS CALLS', ')': 'TUPLES FUNCTIONS CALLS', '[': 'LISTS SUBSCRIPTS SLICINGS', ']': 'LISTS SUBSCRIPTS SLICINGS'}
    for (topic, symbols_) in _symbols_inverse.items():
        for symbol in symbols_:
            topics = symbols.get(symbol, topic)
            if topic not in topics:
                topics = topics + ' ' + topic
            symbols[symbol] = topics
    topics = {'TYPES': ('types', 'STRINGS UNICODE NUMBERS SEQUENCES MAPPINGS FUNCTIONS CLASSES MODULES FILES inspect'), 'STRINGS': ('strings', 'str UNICODE SEQUENCES STRINGMETHODS FORMATTING TYPES'), 'STRINGMETHODS': ('string-methods', 'STRINGS FORMATTING'), 'FORMATTING': ('formatstrings', 'OPERATORS'), 'UNICODE': ('strings', 'encodings unicode SEQUENCES STRINGMETHODS FORMATTING TYPES'), 'NUMBERS': ('numbers', 'INTEGER FLOAT COMPLEX TYPES'), 'INTEGER': ('integers', 'int range'), 'FLOAT': ('floating', 'float math'), 'COMPLEX': ('imaginary', 'complex cmath'), 'SEQUENCES': ('typesseq', 'STRINGMETHODS FORMATTING range LISTS'), 'MAPPINGS': 'DICTIONARIES', 'FUNCTIONS': ('typesfunctions', 'def TYPES'), 'METHODS': ('typesmethods', 'class def CLASSES TYPES'), 'CODEOBJECTS': ('bltin-code-objects', 'compile FUNCTIONS TYPES'), 'TYPEOBJECTS': ('bltin-type-objects', 'types TYPES'), 'FRAMEOBJECTS': 'TYPES', 'TRACEBACKS': 'TYPES', 'NONE': ('bltin-null-object', ''), 'ELLIPSIS': ('bltin-ellipsis-object', 'SLICINGS'), 'SPECIALATTRIBUTES': ('specialattrs', ''), 'CLASSES': ('types', 'class SPECIALMETHODS PRIVATENAMES'), 'MODULES': ('typesmodules', 'import'), 'PACKAGES': 'import', 'EXPRESSIONS': ('operator-summary', 'lambda or and not in is BOOLEAN COMPARISON BITWISE SHIFTING BINARY FORMATTING POWER UNARY ATTRIBUTES SUBSCRIPTS SLICINGS CALLS TUPLES LISTS DICTIONARIES'), 'OPERATORS': 'EXPRESSIONS', 'PRECEDENCE': 'EXPRESSIONS', 'OBJECTS': ('objects', 'TYPES'), 'SPECIALMETHODS': ('specialnames', 'BASICMETHODS ATTRIBUTEMETHODS CALLABLEMETHODS SEQUENCEMETHODS MAPPINGMETHODS NUMBERMETHODS CLASSES'), 'BASICMETHODS': ('customization', 'hash repr str SPECIALMETHODS'), 'ATTRIBUTEMETHODS': ('attribute-access', 'ATTRIBUTES SPECIALMETHODS'), 'CALLABLEMETHODS': ('callable-types', 'CALLS SPECIALMETHODS'), 'SEQUENCEMETHODS': ('sequence-types', 'SEQUENCES SEQUENCEMETHODS SPECIALMETHODS'), 'MAPPINGMETHODS': ('sequence-types', 'MAPPINGS SPECIALMETHODS'), 'NUMBERMETHODS': ('numeric-types', 'NUMBERS AUGMENTEDASSIGNMENT SPECIALMETHODS'), 'EXECUTION': ('execmodel', 'NAMESPACES DYNAMICFEATURES EXCEPTIONS'), 'NAMESPACES': ('naming', 'global nonlocal ASSIGNMENT DELETION DYNAMICFEATURES'), 'DYNAMICFEATURES': ('dynamic-features', ''), 'SCOPING': 'NAMESPACES', 'FRAMES': 'NAMESPACES', 'EXCEPTIONS': ('exceptions', 'try except finally raise'), 'CONVERSIONS': ('conversions', ''), 'IDENTIFIERS': ('identifiers', 'keywords SPECIALIDENTIFIERS'), 'SPECIALIDENTIFIERS': ('id-classes', ''), 'PRIVATENAMES': ('atom-identifiers', ''), 'LITERALS': ('atom-literals', 'STRINGS NUMBERS TUPLELITERALS LISTLITERALS DICTIONARYLITERALS'), 'TUPLES': 'SEQUENCES', 'TUPLELITERALS': ('exprlists', 'TUPLES LITERALS'), 'LISTS': ('typesseq-mutable', 'LISTLITERALS'), 'LISTLITERALS': ('lists', 'LISTS LITERALS'), 'DICTIONARIES': ('typesmapping', 'DICTIONARYLITERALS'), 'DICTIONARYLITERALS': ('dict', 'DICTIONARIES LITERALS'), 'ATTRIBUTES': ('attribute-references', 'getattr hasattr setattr ATTRIBUTEMETHODS'), 'SUBSCRIPTS': ('subscriptions', 'SEQUENCEMETHODS'), 'SLICINGS': ('slicings', 'SEQUENCEMETHODS'), 'CALLS': ('calls', 'EXPRESSIONS'), 'POWER': ('power', 'EXPRESSIONS'), 'UNARY': ('unary', 'EXPRESSIONS'), 'BINARY': ('binary', 'EXPRESSIONS'), 'SHIFTING': ('shifting', 'EXPRESSIONS'), 'BITWISE': ('bitwise', 'EXPRESSIONS'), 'COMPARISON': ('comparisons', 'EXPRESSIONS BASICMETHODS'), 'BOOLEAN': ('booleans', 'EXPRESSIONS TRUTHVALUE'), 'ASSERTION': 'assert', 'ASSIGNMENT': ('assignment', 'AUGMENTEDASSIGNMENT'), 'AUGMENTEDASSIGNMENT': ('augassign', 'NUMBERMETHODS'), 'DELETION': 'del', 'RETURNING': 'return', 'IMPORTING': 'import', 'CONDITIONAL': 'if', 'LOOPING': ('compound', 'for while break continue'), 'TRUTHVALUE': ('truth', 'if while and or not BASICMETHODS'), 'DEBUGGING': ('debugger', 'pdb'), 'CONTEXTMANAGERS': ('context-managers', 'with')}

    def __init__(self, input=None, output=None):
        if False:
            while True:
                i = 10
        self._input = input
        self._output = output

    @property
    def input(self):
        if False:
            while True:
                i = 10
        return self._input or sys.stdin

    @property
    def output(self):
        if False:
            for i in range(10):
                print('nop')
        return self._output or sys.stdout

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if inspect.stack()[1][3] == '?':
            self()
            return ''
        return '<%s.%s instance>' % (self.__class__.__module__, self.__class__.__qualname__)
    _GoInteractive = object()

    def __call__(self, request=_GoInteractive):
        if False:
            while True:
                i = 10
        if request is not self._GoInteractive:
            self.help(request)
        else:
            self.intro()
            self.interact()
            self.output.write('\nYou are now leaving help and returning to the Python interpreter.\nIf you want to ask for help on a particular object directly from the\ninterpreter, you can type "help(object)".  Executing "help(\'string\')"\nhas the same effect as typing a particular string at the help> prompt.\n')

    def interact(self):
        if False:
            while True:
                i = 10
        self.output.write('\n')
        while True:
            try:
                request = self.getline('help> ')
                if not request:
                    break
            except (KeyboardInterrupt, EOFError):
                break
            request = request.strip()
            if len(request) > 2 and request[0] == request[-1] in ("'", '"') and (request[0] not in request[1:-1]):
                request = request[1:-1]
            if request.lower() in ('q', 'quit'):
                break
            if request == 'help':
                self.intro()
            else:
                self.help(request)

    def getline(self, prompt):
        if False:
            return 10
        'Read one line, using input() when appropriate.'
        if self.input is sys.stdin:
            return input(prompt)
        else:
            self.output.write(prompt)
            self.output.flush()
            return self.input.readline()

    def help(self, request):
        if False:
            for i in range(10):
                print('nop')
        if type(request) is type(''):
            request = request.strip()
            if request == 'keywords':
                self.listkeywords()
            elif request == 'symbols':
                self.listsymbols()
            elif request == 'topics':
                self.listtopics()
            elif request == 'modules':
                self.listmodules()
            elif request[:8] == 'modules ':
                self.listmodules(request.split()[1])
            elif request in self.symbols:
                self.showsymbol(request)
            elif request in ['True', 'False', 'None']:
                doc(eval(request), 'Help on %s:')
            elif request in self.keywords:
                self.showtopic(request)
            elif request in self.topics:
                self.showtopic(request)
            elif request:
                doc(request, 'Help on %s:', output=self._output)
            else:
                doc(str, 'Help on %s:', output=self._output)
        elif isinstance(request, Helper):
            self()
        else:
            doc(request, 'Help on %s:', output=self._output)
        self.output.write('\n')

    def intro(self):
        if False:
            print('Hello World!')
        self.output.write('\nWelcome to Python {0}\'s help utility!\n\nIf this is your first time using Python, you should definitely check out\nthe tutorial on the internet at https://docs.python.org/{0}/tutorial/.\n\nEnter the name of any module, keyword, or topic to get help on writing\nPython programs and using Python modules.  To quit this help utility and\nreturn to the interpreter, just type "quit".\n\nTo get a list of available modules, keywords, symbols, or topics, type\n"modules", "keywords", "symbols", or "topics".  Each module also comes\nwith a one-line summary of what it does; to list the modules whose name\nor summary contain a given string such as "spam", type "modules spam".\n'.format('%d.%d' % sys.version_info[:2]))

    def list(self, items, columns=4, width=80):
        if False:
            print('Hello World!')
        items = list(sorted(items))
        colw = width // columns
        rows = (len(items) + columns - 1) // columns
        for row in range(rows):
            for col in range(columns):
                i = col * rows + row
                if i < len(items):
                    self.output.write(items[i])
                    if col < columns - 1:
                        self.output.write(' ' + ' ' * (colw - 1 - len(items[i])))
            self.output.write('\n')

    def listkeywords(self):
        if False:
            i = 10
            return i + 15
        self.output.write('\nHere is a list of the Python keywords.  Enter any keyword to get more help.\n\n')
        self.list(self.keywords.keys())

    def listsymbols(self):
        if False:
            for i in range(10):
                print('nop')
        self.output.write('\nHere is a list of the punctuation symbols which Python assigns special meaning\nto. Enter any symbol to get more help.\n\n')
        self.list(self.symbols.keys())

    def listtopics(self):
        if False:
            while True:
                i = 10
        self.output.write('\nHere is a list of available topics.  Enter any topic name to get more help.\n\n')
        self.list(self.topics.keys())

    def showtopic(self, topic, more_xrefs=''):
        if False:
            print('Hello World!')
        try:
            import pydoc_data.topics
        except ImportError:
            self.output.write('\nSorry, topic and keyword documentation is not available because the\nmodule "pydoc_data.topics" could not be found.\n')
            return
        target = self.topics.get(topic, self.keywords.get(topic))
        if not target:
            self.output.write('no documentation found for %s\n' % repr(topic))
            return
        if type(target) is type(''):
            return self.showtopic(target, more_xrefs)
        (label, xrefs) = target
        try:
            doc = pydoc_data.topics.topics[label]
        except KeyError:
            self.output.write('no documentation found for %s\n' % repr(topic))
            return
        doc = doc.strip() + '\n'
        if more_xrefs:
            xrefs = (xrefs or '') + ' ' + more_xrefs
        if xrefs:
            import textwrap
            text = 'Related help topics: ' + ', '.join(xrefs.split()) + '\n'
            wrapped_text = textwrap.wrap(text, 72)
            doc += '\n%s\n' % '\n'.join(wrapped_text)
        pager(doc)

    def _gettopic(self, topic, more_xrefs=''):
        if False:
            i = 10
            return i + 15
        'Return unbuffered tuple of (topic, xrefs).\n\n        If an error occurs here, the exception is caught and displayed by\n        the url handler.\n\n        This function duplicates the showtopic method but returns its\n        result directly so it can be formatted for display in an html page.\n        '
        try:
            import pydoc_data.topics
        except ImportError:
            return ('\nSorry, topic and keyword documentation is not available because the\nmodule "pydoc_data.topics" could not be found.\n', '')
        target = self.topics.get(topic, self.keywords.get(topic))
        if not target:
            raise ValueError('could not find topic')
        if isinstance(target, str):
            return self._gettopic(target, more_xrefs)
        (label, xrefs) = target
        doc = pydoc_data.topics.topics[label]
        if more_xrefs:
            xrefs = (xrefs or '') + ' ' + more_xrefs
        return (doc, xrefs)

    def showsymbol(self, symbol):
        if False:
            i = 10
            return i + 15
        target = self.symbols[symbol]
        (topic, _, xrefs) = target.partition(' ')
        self.showtopic(topic, xrefs)

    def listmodules(self, key=''):
        if False:
            while True:
                i = 10
        if key:
            self.output.write("\nHere is a list of modules whose name or summary contains '{}'.\nIf there are any, enter a module name to get more help.\n\n".format(key))
            apropos(key)
        else:
            self.output.write('\nPlease wait a moment while I gather a list of all available modules...\n\n')
            modules = {}

            def callback(path, modname, desc, modules=modules):
                if False:
                    return 10
                if modname and modname[-9:] == '.__init__':
                    modname = modname[:-9] + ' (package)'
                if modname.find('.') < 0:
                    modules[modname] = 1

            def onerror(modname):
                if False:
                    i = 10
                    return i + 15
                callback(None, modname, None)
            ModuleScanner().run(callback, onerror=onerror)
            self.list(modules.keys())
            self.output.write('\nEnter any module name to get more help.  Or, type "modules spam" to search\nfor modules whose name or summary contain the string "spam".\n')
help = Helper()

class ModuleScanner:
    """An interruptible scanner that searches module synopses."""

    def run(self, callback, key=None, completer=None, onerror=None):
        if False:
            i = 10
            return i + 15
        if key:
            key = key.lower()
        self.quit = False
        seen = {}
        for modname in sys.builtin_module_names:
            if modname != '__main__':
                seen[modname] = 1
                if key is None:
                    callback(None, modname, '')
                else:
                    name = __import__(modname).__doc__ or ''
                    desc = name.split('\n')[0]
                    name = modname + ' - ' + desc
                    if name.lower().find(key) >= 0:
                        callback(None, modname, desc)
        for (importer, modname, ispkg) in pkgutil.walk_packages(onerror=onerror):
            if self.quit:
                break
            if key is None:
                callback(None, modname, '')
            else:
                try:
                    spec = pkgutil._get_spec(importer, modname)
                except SyntaxError:
                    continue
                loader = spec.loader
                if hasattr(loader, 'get_source'):
                    try:
                        source = loader.get_source(modname)
                    except Exception:
                        if onerror:
                            onerror(modname)
                        continue
                    desc = source_synopsis(io.StringIO(source)) or ''
                    if hasattr(loader, 'get_filename'):
                        path = loader.get_filename(modname)
                    else:
                        path = None
                else:
                    try:
                        module = importlib._bootstrap._load(spec)
                    except ImportError:
                        if onerror:
                            onerror(modname)
                        continue
                    desc = module.__doc__.splitlines()[0] if module.__doc__ else ''
                    path = getattr(module, '__file__', None)
                name = modname + ' - ' + desc
                if name.lower().find(key) >= 0:
                    callback(path, modname, desc)
        if completer:
            completer()

def apropos(key):
    if False:
        print('Hello World!')
    'Print all the one-line module summaries that contain a substring.'

    def callback(path, modname, desc):
        if False:
            i = 10
            return i + 15
        if modname[-9:] == '.__init__':
            modname = modname[:-9] + ' (package)'
        print(modname, desc and '- ' + desc)

    def onerror(modname):
        if False:
            for i in range(10):
                print('nop')
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ModuleScanner().run(callback, key, onerror=onerror)

def _start_server(urlhandler, hostname, port):
    if False:
        print('Hello World!')
    "Start an HTTP server thread on a specific port.\n\n    Start an HTML/text server thread, so HTML or text documents can be\n    browsed dynamically and interactively with a web browser.  Example use:\n\n        >>> import time\n        >>> import pydoc\n\n        Define a URL handler.  To determine what the client is asking\n        for, check the URL and content_type.\n\n        Then get or generate some text or HTML code and return it.\n\n        >>> def my_url_handler(url, content_type):\n        ...     text = 'the URL sent was: (%s, %s)' % (url, content_type)\n        ...     return text\n\n        Start server thread on port 0.\n        If you use port 0, the server will pick a random port number.\n        You can then use serverthread.port to get the port number.\n\n        >>> port = 0\n        >>> serverthread = pydoc._start_server(my_url_handler, port)\n\n        Check that the server is really started.  If it is, open browser\n        and get first page.  Use serverthread.url as the starting page.\n\n        >>> if serverthread.serving:\n        ...    import webbrowser\n\n        The next two lines are commented out so a browser doesn't open if\n        doctest is run on this module.\n\n        #...    webbrowser.open(serverthread.url)\n        #True\n\n        Let the server do its thing. We just need to monitor its status.\n        Use time.sleep so the loop doesn't hog the CPU.\n\n        >>> starttime = time.monotonic()\n        >>> timeout = 1                    #seconds\n\n        This is a short timeout for testing purposes.\n\n        >>> while serverthread.serving:\n        ...     time.sleep(.01)\n        ...     if serverthread.serving and time.monotonic() - starttime > timeout:\n        ...          serverthread.stop()\n        ...          break\n\n        Print any errors that may have occurred.\n\n        >>> print(serverthread.error)\n        None\n   "
    import http.server
    import email.message
    import select
    import threading

    class DocHandler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            if False:
                for i in range(10):
                    print('nop')
            'Process a request from an HTML browser.\n\n            The URL received is in self.path.\n            Get an HTML page from self.urlhandler and send it.\n            '
            if self.path.endswith('.css'):
                content_type = 'text/css'
            else:
                content_type = 'text/html'
            self.send_response(200)
            self.send_header('Content-Type', '%s; charset=UTF-8' % content_type)
            self.end_headers()
            self.wfile.write(self.urlhandler(self.path, content_type).encode('utf-8'))

        def log_message(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            pass

    class DocServer(http.server.HTTPServer):

        def __init__(self, host, port, callback):
            if False:
                print('Hello World!')
            self.host = host
            self.address = (self.host, port)
            self.callback = callback
            self.base.__init__(self, self.address, self.handler)
            self.quit = False

        def serve_until_quit(self):
            if False:
                for i in range(10):
                    print('nop')
            while not self.quit:
                (rd, wr, ex) = select.select([self.socket.fileno()], [], [], 1)
                if rd:
                    self.handle_request()
            self.server_close()

        def server_activate(self):
            if False:
                print('Hello World!')
            self.base.server_activate(self)
            if self.callback:
                self.callback(self)

    class ServerThread(threading.Thread):

        def __init__(self, urlhandler, host, port):
            if False:
                i = 10
                return i + 15
            self.urlhandler = urlhandler
            self.host = host
            self.port = int(port)
            threading.Thread.__init__(self)
            self.serving = False
            self.error = None
            self.docserver = None

        def run(self):
            if False:
                print('Hello World!')
            'Start the server.'
            try:
                DocServer.base = http.server.HTTPServer
                DocServer.handler = DocHandler
                DocHandler.MessageClass = email.message.Message
                DocHandler.urlhandler = staticmethod(self.urlhandler)
                docsvr = DocServer(self.host, self.port, self.ready)
                self.docserver = docsvr
                docsvr.serve_until_quit()
            except Exception as e:
                self.error = e

        def ready(self, server):
            if False:
                for i in range(10):
                    print('nop')
            self.serving = True
            self.host = server.host
            self.port = server.server_port
            self.url = 'http://%s:%d/' % (self.host, self.port)

        def stop(self):
            if False:
                for i in range(10):
                    print('nop')
            'Stop the server and this thread nicely'
            self.docserver.quit = True
            self.join()
            self.docserver = None
            self.serving = False
            self.url = None
    thread = ServerThread(urlhandler, hostname, port)
    thread.start()
    while not thread.error and (not (thread.serving and thread.docserver)):
        time.sleep(0.01)
    return thread

def _url_handler(url, content_type='text/html'):
    if False:
        i = 10
        return i + 15
    "The pydoc url handler for use with the pydoc server.\n\n    If the content_type is 'text/css', the _pydoc.css style\n    sheet is read and returned if it exits.\n\n    If the content_type is 'text/html', then the result of\n    get_html_page(url) is returned.\n    "

    class _HTMLDoc(HTMLDoc):

        def page(self, title, contents):
            if False:
                while True:
                    i = 10
            'Format an HTML page.'
            css_path = 'pydoc_data/_pydoc.css'
            css_link = '<link rel="stylesheet" type="text/css" href="%s">' % css_path
            return '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">\n<html><head><title>Pydoc: %s</title>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n%s</head><body bgcolor="#f0f0f8">%s<div style="clear:both;padding-top:.5em;">%s</div>\n</body></html>' % (title, css_link, html_navbar(), contents)
    html = _HTMLDoc()

    def html_navbar():
        if False:
            while True:
                i = 10
        version = html.escape('%s [%s, %s]' % (platform.python_version(), platform.python_build()[0], platform.python_compiler()))
        return '\n            <div style=\'float:left\'>\n                Python %s<br>%s\n            </div>\n            <div style=\'float:right\'>\n                <div style=\'text-align:center\'>\n                  <a href="index.html">Module Index</a>\n                  : <a href="topics.html">Topics</a>\n                  : <a href="keywords.html">Keywords</a>\n                </div>\n                <div>\n                    <form action="get" style=\'display:inline;\'>\n                      <input type=text name=key size=15>\n                      <input type=submit value="Get">\n                    </form>&nbsp;\n                    <form action="search" style=\'display:inline;\'>\n                      <input type=text name=key size=15>\n                      <input type=submit value="Search">\n                    </form>\n                </div>\n            </div>\n            ' % (version, html.escape(platform.platform(terse=True)))

    def html_index():
        if False:
            print('Hello World!')
        'Module Index page.'

        def bltinlink(name):
            if False:
                i = 10
                return i + 15
            return '<a href="%s.html">%s</a>' % (name, name)
        heading = html.heading('<big><big><strong>Index of Modules</strong></big></big>', '#ffffff', '#7799ee')
        names = [name for name in sys.builtin_module_names if name != '__main__']
        contents = html.multicolumn(names, bltinlink)
        contents = [heading, '<p>' + html.bigsection('Built-in Modules', '#ffffff', '#ee77aa', contents)]
        seen = {}
        for dir in sys.path:
            contents.append(html.index(dir, seen))
        contents.append('<p align=right><font color="#909090" face="helvetica,arial"><strong>pydoc</strong> by Ka-Ping Yee&lt;ping@lfw.org&gt;</font>')
        return ('Index of Modules', ''.join(contents))

    def html_search(key):
        if False:
            print('Hello World!')
        'Search results page.'
        search_result = []

        def callback(path, modname, desc):
            if False:
                while True:
                    i = 10
            if modname[-9:] == '.__init__':
                modname = modname[:-9] + ' (package)'
            search_result.append((modname, desc and '- ' + desc))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            def onerror(modname):
                if False:
                    return 10
                pass
            ModuleScanner().run(callback, key, onerror=onerror)

        def bltinlink(name):
            if False:
                print('Hello World!')
            return '<a href="%s.html">%s</a>' % (name, name)
        results = []
        heading = html.heading('<big><big><strong>Search Results</strong></big></big>', '#ffffff', '#7799ee')
        for (name, desc) in search_result:
            results.append(bltinlink(name) + desc)
        contents = heading + html.bigsection('key = %s' % key, '#ffffff', '#ee77aa', '<br>'.join(results))
        return ('Search Results', contents)

    def html_topics():
        if False:
            for i in range(10):
                print('nop')
        'Index of topic texts available.'

        def bltinlink(name):
            if False:
                return 10
            return '<a href="topic?key=%s">%s</a>' % (name, name)
        heading = html.heading('<big><big><strong>INDEX</strong></big></big>', '#ffffff', '#7799ee')
        names = sorted(Helper.topics.keys())
        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection('Topics', '#ffffff', '#ee77aa', contents)
        return ('Topics', contents)

    def html_keywords():
        if False:
            return 10
        'Index of keywords.'
        heading = html.heading('<big><big><strong>INDEX</strong></big></big>', '#ffffff', '#7799ee')
        names = sorted(Helper.keywords.keys())

        def bltinlink(name):
            if False:
                for i in range(10):
                    print('nop')
            return '<a href="topic?key=%s">%s</a>' % (name, name)
        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection('Keywords', '#ffffff', '#ee77aa', contents)
        return ('Keywords', contents)

    def html_topicpage(topic):
        if False:
            return 10
        'Topic or keyword help page.'
        buf = io.StringIO()
        htmlhelp = Helper(buf, buf)
        (contents, xrefs) = htmlhelp._gettopic(topic)
        if topic in htmlhelp.keywords:
            title = 'KEYWORD'
        else:
            title = 'TOPIC'
        heading = html.heading('<big><big><strong>%s</strong></big></big>' % title, '#ffffff', '#7799ee')
        contents = '<pre>%s</pre>' % html.markup(contents)
        contents = html.bigsection(topic, '#ffffff', '#ee77aa', contents)
        if xrefs:
            xrefs = sorted(xrefs.split())

            def bltinlink(name):
                if False:
                    while True:
                        i = 10
                return '<a href="topic?key=%s">%s</a>' % (name, name)
            xrefs = html.multicolumn(xrefs, bltinlink)
            xrefs = html.section('Related help topics: ', '#ffffff', '#ee77aa', xrefs)
        return ('%s %s' % (title, topic), ''.join((heading, contents, xrefs)))

    def html_getobj(url):
        if False:
            for i in range(10):
                print('nop')
        obj = locate(url, forceload=1)
        if obj is None and url != 'None':
            raise ValueError('could not find object')
        title = describe(obj)
        content = html.document(obj, url)
        return (title, content)

    def html_error(url, exc):
        if False:
            while True:
                i = 10
        heading = html.heading('<big><big><strong>Error</strong></big></big>', '#ffffff', '#7799ee')
        contents = '<br>'.join((html.escape(line) for line in format_exception_only(type(exc), exc)))
        contents = heading + html.bigsection(url, '#ffffff', '#bb0000', contents)
        return ('Error - %s' % url, contents)

    def get_html_page(url):
        if False:
            print('Hello World!')
        'Generate an HTML page for url.'
        complete_url = url
        if url.endswith('.html'):
            url = url[:-5]
        try:
            if url in ('', 'index'):
                (title, content) = html_index()
            elif url == 'topics':
                (title, content) = html_topics()
            elif url == 'keywords':
                (title, content) = html_keywords()
            elif '=' in url:
                (op, _, url) = url.partition('=')
                if op == 'search?key':
                    (title, content) = html_search(url)
                elif op == 'topic?key':
                    try:
                        (title, content) = html_topicpage(url)
                    except ValueError:
                        (title, content) = html_getobj(url)
                elif op == 'get?key':
                    if url in ('', 'index'):
                        (title, content) = html_index()
                    else:
                        try:
                            (title, content) = html_getobj(url)
                        except ValueError:
                            (title, content) = html_topicpage(url)
                else:
                    raise ValueError('bad pydoc url')
            else:
                (title, content) = html_getobj(url)
        except Exception as exc:
            (title, content) = html_error(complete_url, exc)
        return html.page(title, content)
    if url.startswith('/'):
        url = url[1:]
    if content_type == 'text/css':
        path_here = os.path.dirname(os.path.realpath(__file__))
        css_path = os.path.join(path_here, url)
        with open(css_path) as fp:
            return ''.join(fp.readlines())
    elif content_type == 'text/html':
        return get_html_page(url)
    raise TypeError('unknown content type %r for url %s' % (content_type, url))

def browse(port=0, *, open_browser=True, hostname='localhost'):
    if False:
        i = 10
        return i + 15
    "Start the enhanced pydoc web server and open a web browser.\n\n    Use port '0' to start the server on an arbitrary port.\n    Set open_browser to False to suppress opening a browser.\n    "
    import webbrowser
    serverthread = _start_server(_url_handler, hostname, port)
    if serverthread.error:
        print(serverthread.error)
        return
    if serverthread.serving:
        server_help_msg = 'Server commands: [b]rowser, [q]uit'
        if open_browser:
            webbrowser.open(serverthread.url)
        try:
            print('Server ready at', serverthread.url)
            print(server_help_msg)
            while serverthread.serving:
                cmd = input('server> ')
                cmd = cmd.lower()
                if cmd == 'q':
                    break
                elif cmd == 'b':
                    webbrowser.open(serverthread.url)
                else:
                    print(server_help_msg)
        except (KeyboardInterrupt, EOFError):
            print()
        finally:
            if serverthread.serving:
                serverthread.stop()
                print('Server stopped')

def ispath(x):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(x, str) and x.find(os.sep) >= 0

def _get_revised_path(given_path, argv0):
    if False:
        print('Hello World!')
    "Ensures current directory is on returned path, and argv0 directory is not\n\n    Exception: argv0 dir is left alone if it's also pydoc's directory.\n\n    Returns a new path entry list, or None if no adjustment is needed.\n    "
    if '' in given_path or os.curdir in given_path or os.getcwd() in given_path:
        return None
    stdlib_dir = os.path.dirname(__file__)
    script_dir = os.path.dirname(argv0)
    revised_path = given_path.copy()
    if script_dir in given_path and (not os.path.samefile(script_dir, stdlib_dir)):
        revised_path.remove(script_dir)
    revised_path.insert(0, os.getcwd())
    return revised_path

def _adjust_cli_sys_path():
    if False:
        i = 10
        return i + 15
    "Ensures current directory is on sys.path, and __main__ directory is not.\n\n    Exception: __main__ dir is left alone if it's also pydoc's directory.\n    "
    revised_path = _get_revised_path(sys.path, sys.argv[0])
    if revised_path is not None:
        sys.path[:] = revised_path

def cli():
    if False:
        i = 10
        return i + 15
    'Command-line interface (looks at sys.argv to decide what to do).'
    import getopt

    class BadUsage(Exception):
        pass
    _adjust_cli_sys_path()
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'bk:n:p:w')
        writing = False
        start_server = False
        open_browser = False
        port = 0
        hostname = 'localhost'
        for (opt, val) in opts:
            if opt == '-b':
                start_server = True
                open_browser = True
            if opt == '-k':
                apropos(val)
                return
            if opt == '-p':
                start_server = True
                port = val
            if opt == '-w':
                writing = True
            if opt == '-n':
                start_server = True
                hostname = val
        if start_server:
            browse(port, hostname=hostname, open_browser=open_browser)
            return
        if not args:
            raise BadUsage
        for arg in args:
            if ispath(arg) and (not os.path.exists(arg)):
                print('file %r does not exist' % arg)
                break
            try:
                if ispath(arg) and os.path.isfile(arg):
                    arg = importfile(arg)
                if writing:
                    if ispath(arg) and os.path.isdir(arg):
                        writedocs(arg)
                    else:
                        writedoc(arg)
                else:
                    help.help(arg)
            except ErrorDuringImport as value:
                print(value)
    except (getopt.error, BadUsage):
        cmd = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        print("pydoc - the Python documentation tool\n\n{cmd} <name> ...\n    Show text documentation on something.  <name> may be the name of a\n    Python keyword, topic, function, module, or package, or a dotted\n    reference to a class or function within a module or module in a\n    package.  If <name> contains a '{sep}', it is used as the path to a\n    Python source file to document. If name is 'keywords', 'topics',\n    or 'modules', a listing of these things is displayed.\n\n{cmd} -k <keyword>\n    Search for a keyword in the synopsis lines of all available modules.\n\n{cmd} -n <hostname>\n    Start an HTTP server with the given hostname (default: localhost).\n\n{cmd} -p <port>\n    Start an HTTP server on the given port on the local machine.  Port\n    number 0 can be used to get an arbitrary unused port.\n\n{cmd} -b\n    Start an HTTP server on an arbitrary unused port and open a web browser\n    to interactively browse documentation.  This option can be used in\n    combination with -n and/or -p.\n\n{cmd} -w <name> ...\n    Write out the HTML documentation for a module to a file in the current\n    directory.  If <name> contains a '{sep}', it is treated as a filename; if\n    it names a directory, documentation is written for all the contents.\n".format(cmd=cmd, sep=os.sep))
if __name__ == '__main__':
    cli()