"""Tools for inspecting Python objects.

This file was forked from the IPython project:

* Copyright (c) 2008-2014, IPython Development Team
* Copyright (C) 2001-2007 Fernando Perez <fperez@colorado.edu>
* Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>
* Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>
"""
import inspect
import itertools
import linecache
import os
import sys
import types
from xonsh.lazyasd import LazyObject
from xonsh.lazyimps import pyghooks, pygments
from xonsh.openpy import read_py_file
from xonsh.platform import HAS_PYGMENTS
from xonsh.style_tools import partial_color_tokenize
from xonsh.tokenize import detect_encoding
from xonsh.tools import cast_unicode, format_color, indent, print_color, safe_hasattr
_func_call_docstring = LazyObject(lambda : types.FunctionType.__call__.__doc__, globals(), '_func_call_docstring')
_object_init_docstring = LazyObject(lambda : object.__init__.__doc__, globals(), '_object_init_docstring')
_builtin_type_docstrings = LazyObject(lambda : {inspect.getdoc(t) for t in (types.ModuleType, types.MethodType, types.FunctionType, property)}, globals(), '_builtin_type_docstrings')
_builtin_func_type = LazyObject(lambda : type(all), globals(), '_builtin_func_type')
_builtin_meth_type = LazyObject(lambda : type(str.upper), globals(), '_builtin_meth_type')
info_fields = LazyObject(lambda : ['type_name', 'base_class', 'string_form', 'namespace', 'length', 'file', 'definition', 'docstring', 'source', 'init_definition', 'class_docstring', 'init_docstring', 'call_def', 'call_docstring', 'ismagic', 'isalias', 'isclass', 'argspec', 'found', 'name'], globals(), 'info_fields')

def object_info(**kw):
    if False:
        return 10
    'Make an object info dict with all fields present.'
    infodict = dict(itertools.zip_longest(info_fields, [None]))
    infodict.update(kw)
    return infodict

def get_encoding(obj):
    if False:
        print('Hello World!')
    'Get encoding for python source file defining obj\n\n    Returns None if obj is not defined in a sourcefile.\n    '
    ofile = find_file(obj)
    if ofile is None:
        return None
    elif ofile.endswith(('.so', '.dll', '.pyd')):
        return None
    elif not os.path.isfile(ofile):
        return None
    else:
        with open(ofile, 'rb') as buf:
            (encoding, _) = detect_encoding(buf.readline)
        return encoding

def getdoc(obj):
    if False:
        while True:
            i = 10
    "Stable wrapper around inspect.getdoc.\n\n    This can't crash because of attribute problems.\n\n    It also attempts to call a getdoc() method on the given object.  This\n    allows objects which provide their docstrings via non-standard mechanisms\n    (like Pyro proxies) to still be inspected by ipython's ? system."
    try:
        ds = obj.getdoc()
    except Exception:
        pass
    else:
        if isinstance(ds, str):
            return inspect.cleandoc(ds)
    try:
        docstr = inspect.getdoc(obj)
        encoding = get_encoding(obj)
        return cast_unicode(docstr, encoding=encoding)
    except Exception:
        raise

def getsource(obj, is_binary=False):
    if False:
        print('Hello World!')
    'Wrapper around inspect.getsource.\n\n    This can be modified by other projects to provide customized source\n    extraction.\n\n    Inputs:\n\n    - obj: an object whose source code we will attempt to extract.\n\n    Optional inputs:\n\n    - is_binary: whether the object is known to come from a binary source.\n      This implementation will skip returning any output for binary objects,\n      but custom extractors may know how to meaningfully process them.'
    if is_binary:
        return None
    else:
        if hasattr(obj, '__wrapped__'):
            obj = obj.__wrapped__
        try:
            src = inspect.getsource(obj)
        except TypeError:
            if hasattr(obj, '__class__'):
                src = inspect.getsource(obj.__class__)
        encoding = get_encoding(obj)
        return cast_unicode(src, encoding=encoding)

def is_simple_callable(obj):
    if False:
        i = 10
        return i + 15
    'True if obj is a function ()'
    return inspect.isfunction(obj) or inspect.ismethod(obj) or isinstance(obj, _builtin_func_type) or isinstance(obj, _builtin_meth_type)

def getargspec(obj):
    if False:
        while True:
            i = 10
    'Wrapper around :func:`inspect.getfullargspec` on Python 3, and\n    :func:inspect.getargspec` on Python 2.\n\n    In addition to functions and methods, this can also handle objects with a\n    ``__call__`` attribute.\n    '
    if safe_hasattr(obj, '__call__') and (not is_simple_callable(obj)):
        obj = obj.__call__
    return inspect.getfullargspec(obj)

def format_argspec(argspec):
    if False:
        for i in range(10):
            print('nop')
    "Format argspect, convenience wrapper around inspect's.\n\n    This takes a dict instead of ordered arguments and calls\n    inspect.format_argspec with the arguments in the necessary order.\n    "
    return inspect.formatargspec(argspec['args'], argspec['varargs'], argspec['varkw'], argspec['defaults'])

def call_tip(oinfo, format_call=True):
    if False:
        for i in range(10):
            print('nop')
    "Extract call tip data from an oinfo dict.\n\n    Parameters\n    ----------\n    oinfo : dict\n    format_call : bool, optional\n        If True, the call line is formatted and returned as a string.  If not, a\n        tuple of (name, argspec) is returned.\n\n    Returns\n    -------\n    call_info : None, str or (str, dict) tuple.\n        When format_call is True, the whole call information is formatted as a\n        single string.  Otherwise, the object's name and its argspec dict are\n        returned.  If no call information is available, None is returned.\n    docstring : str or None\n        The most relevant docstring for calling purposes is returned, if\n        available.  The priority is: call docstring for callable instances, then\n        constructor docstring for classes, then main object's docstring otherwise\n        (regular functions).\n    "
    argspec = oinfo.get('argspec')
    if argspec is None:
        call_line = None
    else:
        try:
            has_self = argspec['args'][0] == 'self'
        except (KeyError, IndexError):
            pass
        else:
            if has_self:
                argspec['args'] = argspec['args'][1:]
        call_line = oinfo['name'] + format_argspec(argspec)
    doc = oinfo.get('call_docstring')
    if doc is None:
        doc = oinfo.get('init_docstring')
    if doc is None:
        doc = oinfo.get('docstring', '')
    return (call_line, doc)

def find_file(obj):
    if False:
        return 10
    'Find the absolute path to the file where an object was defined.\n\n    This is essentially a robust wrapper around `inspect.getabsfile`.\n\n    Returns None if no file can be found.\n\n    Parameters\n    ----------\n    obj : any Python object\n\n    Returns\n    -------\n    fname : str\n        The absolute path to the file where the object was defined.\n    '
    if safe_hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__
    fname = None
    try:
        fname = inspect.getabsfile(obj)
    except TypeError:
        if hasattr(obj, '__class__'):
            try:
                fname = inspect.getabsfile(obj.__class__)
            except TypeError:
                pass
    except Exception:
        pass
    return cast_unicode(fname)

def find_source_lines(obj):
    if False:
        while True:
            i = 10
    'Find the line number in a file where an object was defined.\n\n    This is essentially a robust wrapper around `inspect.getsourcelines`.\n\n    Returns None if no file can be found.\n\n    Parameters\n    ----------\n    obj : any Python object\n\n    Returns\n    -------\n    lineno : int\n        The line number where the object definition starts.\n    '
    if safe_hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__
    try:
        try:
            lineno = inspect.getsourcelines(obj)[1]
        except TypeError:
            if hasattr(obj, '__class__'):
                lineno = inspect.getsourcelines(obj.__class__)[1]
            else:
                lineno = None
    except Exception:
        return None
    return lineno

class Inspector:
    """Inspects objects."""

    def __init__(self, str_detail_level=0):
        if False:
            while True:
                i = 10
        self.str_detail_level = str_detail_level

    def _getdef(self, obj, oname=''):
        if False:
            print('Hello World!')
        'Return the call signature for any callable object.\n\n        If any exception is generated, None is returned instead and the\n        exception is suppressed.\n        '
        try:
            hdef = oname + str(inspect.signature(obj))
            return cast_unicode(hdef)
        except Exception:
            return None

    def noinfo(self, msg, oname):
        if False:
            print('Hello World!')
        'Generic message when no information is found.'
        print('No %s found' % msg, end=' ')
        if oname:
            print('for %s' % oname)
        else:
            print()

    def pdef(self, obj, oname=''):
        if False:
            for i in range(10):
                print('nop')
        'Print the call signature for any callable object.\n\n        If the object is a class, print the constructor information.\n        '
        if not callable(obj):
            print('Object is not callable.')
            return
        header = ''
        if inspect.isclass(obj):
            header = self.__head('Class constructor information:\n')
            obj = obj.__init__
        output = self._getdef(obj, oname)
        if output is None:
            self.noinfo('definition header', oname)
        else:
            print(header, output, end=' ', file=sys.stdout)

    def pdoc(self, obj, oname=''):
        if False:
            while True:
                i = 10
        'Print the docstring for any object.\n\n        Optional\n\n        -formatter: a function to run the docstring through for specially\n        formatted docstrings.\n        '
        head = self.__head
        lines = []
        ds = getdoc(obj)
        if ds:
            lines.append(head('Class docstring:'))
            lines.append(indent(ds))
        if inspect.isclass(obj) and hasattr(obj, '__init__'):
            init_ds = getdoc(obj.__init__)
            if init_ds is not None:
                lines.append(head('Init docstring:'))
                lines.append(indent(init_ds))
        elif callable(obj):
            call_ds = getdoc(obj.__call__)
            if call_ds:
                lines.append(head('Call docstring:'))
                lines.append(indent(call_ds))
        if not lines:
            self.noinfo('documentation', oname)
        else:
            print('\n'.join(lines))

    def psource(self, obj, oname=''):
        if False:
            i = 10
            return i + 15
        'Print the source code for an object.'
        linecache.checkcache()
        try:
            src = getsource(obj)
        except Exception:
            self.noinfo('source', oname)
        else:
            print(src)

    def pfile(self, obj, oname=''):
        if False:
            for i in range(10):
                print('nop')
        'Show the whole file where an object was defined.'
        lineno = find_source_lines(obj)
        if lineno is None:
            self.noinfo('file', oname)
            return
        ofile = find_file(obj)
        if ofile.endswith(('.so', '.dll', '.pyd')):
            print('File %r is binary, not printing.' % ofile)
        elif not os.path.isfile(ofile):
            print('File %r does not exist, not printing.' % ofile)
        else:
            o = read_py_file(ofile, skip_encoding_cookie=False)
            print(o, lineno - 1)

    def _format_fields_str(self, fields, title_width=0):
        if False:
            i = 10
            return i + 15
        'Formats a list of fields for display using color strings.\n\n        Parameters\n        ----------\n        fields : list\n            A list of 2-tuples: (field_title, field_content)\n        title_width : int\n            How many characters to pad titles to. Default to longest title.\n        '
        out = []
        if title_width == 0:
            title_width = max((len(title) + 2 for (title, _) in fields))
        for (title, content) in fields:
            title_len = len(title)
            title = '{BOLD_RED}' + title + ':{RESET}'
            if len(content.splitlines()) > 1:
                title += '\n'
            else:
                title += ' '.ljust(title_width - title_len)
            out.append(cast_unicode(title) + cast_unicode(content))
        return format_color('\n'.join(out) + '\n')

    def _format_fields_tokens(self, fields, title_width=0):
        if False:
            return 10
        'Formats a list of fields for display using color tokens from\n        pygments.\n\n        Parameters\n        ----------\n        fields : list\n            A list of 2-tuples: (field_title, field_content)\n        title_width : int\n            How many characters to pad titles to. Default to longest title.\n        '
        out = []
        if title_width == 0:
            title_width = max((len(title) + 2 for (title, _) in fields))
        for (title, content) in fields:
            title_len = len(title)
            title = '{BOLD_RED}' + title + ':{RESET}'
            if not isinstance(content, str) or len(content.splitlines()) > 1:
                title += '\n'
            else:
                title += ' '.ljust(title_width - title_len)
            out += partial_color_tokenize(title)
            if isinstance(content, str):
                out[-1] = (out[-1][0], out[-1][1] + content + '\n')
            else:
                out += content
                out[-1] = (out[-1][0], out[-1][1] + '\n')
        out[-1] = (out[-1][0], out[-1][1] + '\n')
        return out

    def _format_fields(self, fields, title_width=0):
        if False:
            return 10
        'Formats a list of fields for display using color tokens from\n        pygments.\n\n        Parameters\n        ----------\n        fields : list\n            A list of 2-tuples: (field_title, field_content)\n        title_width : int\n            How many characters to pad titles to. Default to longest title.\n        '
        if HAS_PYGMENTS:
            rtn = self._format_fields_tokens(fields, title_width=title_width)
        else:
            rtn = self._format_fields_str(fields, title_width=title_width)
        return rtn
    pinfo_fields1 = [('Type', 'type_name')]
    pinfo_fields2 = [('String form', 'string_form')]
    pinfo_fields3 = [('Length', 'length'), ('File', 'file'), ('Definition', 'definition')]
    pinfo_fields_obj = [('Class docstring', 'class_docstring'), ('Init docstring', 'init_docstring'), ('Call def', 'call_def'), ('Call docstring', 'call_docstring')]

    def pinfo(self, obj, oname='', info=None, detail_level=0):
        if False:
            print('Hello World!')
        'Show detailed information about an object.\n\n        Parameters\n        ----------\n        obj : object\n        oname : str, optional\n            name of the variable pointing to the object.\n        info : dict, optional\n            a structure with some information fields which may have been\n            precomputed already.\n        detail_level : int, optional\n            if set to 1, more information is given.\n        '
        info = self.info(obj, oname=oname, info=info, detail_level=detail_level)
        displayfields = []

        def add_fields(fields):
            if False:
                print('Hello World!')
            for (title, key) in fields:
                field = info[key]
                if field is not None:
                    displayfields.append((title, field.rstrip()))
        add_fields(self.pinfo_fields1)
        add_fields(self.pinfo_fields2)
        if info['namespace'] is not None and info['namespace'] != 'Interactive':
            displayfields.append(('Namespace', info['namespace'].rstrip()))
        add_fields(self.pinfo_fields3)
        if info['isclass'] and info['init_definition']:
            displayfields.append(('Init definition', info['init_definition'].rstrip()))
        if detail_level > 0 and info['source'] is not None:
            displayfields.append(('Source', cast_unicode(info['source'])))
        elif info['docstring'] is not None:
            displayfields.append(('Docstring', info['docstring']))
        if info['isclass']:
            if info['init_docstring'] is not None:
                displayfields.append(('Init docstring', info['init_docstring']))
        else:
            add_fields(self.pinfo_fields_obj)
        if displayfields:
            print_color(self._format_fields(displayfields))

    def info(self, obj, oname='', info=None, detail_level=0):
        if False:
            i = 10
            return i + 15
        'Compute a dict with detailed information about an object.\n\n        Optional arguments:\n\n        - oname: name of the variable pointing to the object.\n\n        - info: a structure with some information fields which may have been\n          precomputed already.\n\n        - detail_level: if set to 1, more information is given.\n        '
        obj_type = type(obj)
        if info is None:
            ismagic = 0
            isalias = 0
            ospace = ''
        else:
            ismagic = info.ismagic
            isalias = info.isalias
            ospace = info.namespace
        if isalias:
            if not callable(obj):
                if len(obj) >= 2 and isinstance(obj[1], str):
                    ds = f'Alias to the system command:\n  {obj[1]}'
                else:
                    ds = 'Alias: ' + str(obj)
            else:
                ds = 'Alias to ' + str(obj)
                if obj.__doc__:
                    ds += '\nDocstring:\n' + obj.__doc__
        else:
            ds = getdoc(obj)
            if ds is None:
                ds = '<no docstring>'
        out = dict(name=oname, found=True, isalias=isalias, ismagic=ismagic)
        string_max = 200
        shalf = int((string_max - 5) / 2)
        if ismagic:
            obj_type_name = 'Magic function'
        elif isalias:
            obj_type_name = 'System alias'
        else:
            obj_type_name = obj_type.__name__
        out['type_name'] = obj_type_name
        try:
            bclass = obj.__class__
            out['base_class'] = str(bclass)
        except Exception:
            pass
        if detail_level >= self.str_detail_level:
            try:
                ostr = str(obj)
                str_head = 'string_form'
                if not detail_level and len(ostr) > string_max:
                    ostr = ostr[:shalf] + ' <...> ' + ostr[-shalf:]
                    ostr = ('\n' + ' ' * len(str_head.expandtabs())).join((q.strip() for q in ostr.split('\n')))
                out[str_head] = ostr
            except Exception:
                pass
        if ospace:
            out['namespace'] = ospace
        try:
            out['length'] = str(len(obj))
        except Exception:
            pass
        binary_file = False
        fname = find_file(obj)
        if fname is None:
            binary_file = True
        else:
            if fname.endswith(('.so', '.dll', '.pyd')):
                binary_file = True
            elif fname.endswith('<string>'):
                fname = 'Dynamically generated function. No source code available.'
            out['file'] = fname
        if ds and detail_level == 0:
            out['docstring'] = ds
        if detail_level:
            linecache.checkcache()
            source = None
            try:
                try:
                    source = getsource(obj, binary_file)
                except TypeError:
                    if hasattr(obj, '__class__'):
                        source = getsource(obj.__class__, binary_file)
                if source is not None:
                    source = source.rstrip()
                    if HAS_PYGMENTS:
                        lexer = pyghooks.XonshLexer()
                        source = list(pygments.lex(source, lexer=lexer))
                    out['source'] = source
            except Exception:
                pass
            if ds and source is None:
                out['docstring'] = ds
        if inspect.isclass(obj):
            out['isclass'] = True
            try:
                obj_init = obj.__init__
            except AttributeError:
                init_def = init_ds = None
            else:
                init_def = self._getdef(obj_init, oname)
                init_ds = getdoc(obj_init)
                if init_ds == _object_init_docstring:
                    init_ds = None
            if init_def or init_ds:
                if init_def:
                    out['init_definition'] = init_def
                if init_ds:
                    out['init_docstring'] = init_ds
        else:
            defln = self._getdef(obj, oname)
            if defln:
                out['definition'] = defln
            if ds:
                try:
                    cls = obj.__class__
                except Exception:
                    class_ds = None
                else:
                    class_ds = getdoc(cls)
                if class_ds in _builtin_type_docstrings:
                    class_ds = None
                if class_ds and ds != class_ds:
                    out['class_docstring'] = class_ds
            try:
                init_ds = getdoc(obj.__init__)
                if init_ds == _object_init_docstring:
                    init_ds = None
            except AttributeError:
                init_ds = None
            if init_ds:
                out['init_docstring'] = init_ds
            if safe_hasattr(obj, '__call__') and (not is_simple_callable(obj)):
                call_def = self._getdef(obj.__call__, oname)
                if call_def:
                    call_def = call_def
                    if call_def != out.get('definition'):
                        out['call_def'] = call_def
                call_ds = getdoc(obj.__call__)
                if call_ds == _func_call_docstring:
                    call_ds = None
                if call_ds:
                    out['call_docstring'] = call_ds
        if inspect.isclass(obj):
            callable_obj = getattr(obj, '__init__', None)
        elif callable(obj):
            callable_obj = obj
        else:
            callable_obj = None
        if callable_obj:
            try:
                argspec = getargspec(callable_obj)
            except (TypeError, AttributeError):
                pass
            else:
                out['argspec'] = argspec_dict = dict(argspec._asdict())
                if 'varkw' not in argspec_dict:
                    argspec_dict['varkw'] = argspec_dict.pop('keywords')
        return object_info(**out)