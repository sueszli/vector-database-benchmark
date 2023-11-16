"""Utilities and wrappers around inspect module"""
import builtins
import inspect
import re
SYMBOLS = '[^\\\'\\"a-zA-Z0-9_.]'

def getobj(txt, last=False):
    if False:
        print('Hello World!')
    'Return the last valid object name in string'
    txt_end = ''
    for (startchar, endchar) in ['[]', '()']:
        if txt.endswith(endchar):
            pos = txt.rfind(startchar)
            if pos:
                txt_end = txt[pos:]
                txt = txt[:pos]
    tokens = re.split(SYMBOLS, txt)
    token = None
    try:
        while token is None or re.match(SYMBOLS, token):
            token = tokens.pop()
        if token.endswith('.'):
            token = token[:-1]
        if token.startswith('.'):
            return None
        if last:
            token += txt[txt.rfind(token) + len(token)]
        token += txt_end
        if token:
            return token
    except IndexError:
        return None

def getobjdir(obj):
    if False:
        for i in range(10):
            print('nop')
    '\n    For standard objects, will simply return dir(obj)\n    In special cases (e.g. WrapITK package), will return only string elements\n    of result returned by dir(obj)\n    '
    return [item for item in dir(obj) if isinstance(item, str)]

def getdoc(obj):
    if False:
        return 10
    "\n    Return text documentation from an object. This comes in a form of\n    dictionary with four keys:\n\n    name:\n      The name of the inspected object\n    argspec:\n      It's argspec\n    note:\n      A phrase describing the type of object (function or method) we are\n      inspecting, and the module it belongs to.\n    docstring:\n      It's docstring\n    "
    docstring = inspect.getdoc(obj) or inspect.getcomments(obj) or ''
    try:
        docstring = str(docstring)
    except:
        pass
    doc = {'name': '', 'argspec': '', 'note': '', 'docstring': docstring}
    if callable(obj):
        try:
            name = obj.__name__
        except AttributeError:
            doc['docstring'] = docstring
            return doc
        if inspect.ismethod(obj):
            imclass = obj.__self__.__class__
            if obj.__self__ is not None:
                doc['note'] = 'Method of %s instance' % obj.__self__.__class__.__name__
            else:
                doc['note'] = 'Unbound %s method' % imclass.__name__
            obj = obj.__func__
        elif hasattr(obj, '__module__'):
            doc['note'] = 'Function of %s module' % obj.__module__
        else:
            doc['note'] = 'Function'
        doc['name'] = obj.__name__
        if inspect.isfunction(obj):
            try:
                sig = inspect.signature(obj)
            except ValueError:
                sig = getargspecfromtext(doc['docstring'])
                if not sig:
                    sig = '(...)'
            doc['argspec'] = str(sig)
            if name == '<lambda>':
                doc['name'] = name + ' lambda '
                doc['argspec'] = doc['argspec'][1:-1]
        else:
            argspec = getargspecfromtext(doc['docstring'])
            if argspec:
                doc['argspec'] = argspec
                signature = doc['name'] + doc['argspec']
                docstring_blocks = doc['docstring'].split('\n\n')
                first_block = docstring_blocks[0].strip()
                if first_block == signature:
                    doc['docstring'] = doc['docstring'].replace(signature, '', 1).lstrip()
            else:
                doc['argspec'] = '(...)'
        argspec = doc['argspec']
        doc['argspec'] = argspec.replace('(self)', '()').replace('(self, ', '(')
    return doc

def getsource(obj):
    if False:
        print('Hello World!')
    'Wrapper around inspect.getsource'
    try:
        try:
            src = str(inspect.getsource(obj))
        except TypeError:
            if hasattr(obj, '__class__'):
                src = str(inspect.getsource(obj.__class__))
            else:
                src = getdoc(obj)
        return src
    except (TypeError, IOError):
        return

def getsignaturefromtext(text, objname):
    if False:
        while True:
            i = 10
    'Get object signature from text (i.e. object documentation).'
    if isinstance(text, dict):
        text = text.get('docstring', '')
    args_re = '(\\(.+?\\))'
    if objname:
        signature_re = objname + args_re
    else:
        identifier_re = '(\\w+)'
        signature_re = identifier_re + args_re
    if not text:
        text = ''
    sigs = re.findall(signature_re, text)
    sig = ''
    if sigs:
        default_ipy_sigs = ['(*args, **kwargs)', '(self, /, *args, **kwargs)']
        if objname:
            real_sigs = [s for s in sigs if s not in default_ipy_sigs]
            if real_sigs:
                sig = real_sigs[0]
            else:
                sig = sigs[0]
        else:
            valid_sigs = [s for s in sigs if s[0].isidentifier()]
            if valid_sigs:
                real_sigs = [s for s in valid_sigs if s[1] not in default_ipy_sigs]
                if real_sigs:
                    sig = real_sigs[0][1]
                else:
                    sig = valid_sigs[0][1]
    return sig

def getargspecfromtext(text):
    if False:
        print('Hello World!')
    '\n    Try to get the formatted argspec of a callable from the first block of its\n    docstring.\n    \n    This will return something like `(x, y, k=1)`.\n    '
    blocks = text.split('\n\n')
    first_block = blocks[0].strip().replace('\n', '')
    return getsignaturefromtext(first_block, '')

def getargsfromtext(text, objname):
    if False:
        i = 10
        return i + 15
    'Get arguments from text (object documentation).'
    signature = getsignaturefromtext(text, objname)
    if signature:
        argtxt = signature[signature.find('(') + 1:-1]
        return argtxt.split(',')

def getargsfromdoc(obj):
    if False:
        print('Hello World!')
    'Get arguments from object doc'
    if obj.__doc__ is not None:
        return getargsfromtext(obj.__doc__, obj.__name__)

def getargs(obj):
    if False:
        i = 10
        return i + 15
    "Get the names and default values of a function's arguments"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        func_obj = obj
    elif inspect.ismethod(obj):
        func_obj = obj.__func__
    elif inspect.isclass(obj) and hasattr(obj, '__init__'):
        func_obj = getattr(obj, '__init__')
    else:
        return []
    if not hasattr(func_obj, '__code__'):
        args = getargsfromdoc(func_obj)
        if args is not None:
            return args
        else:
            return getargsfromdoc(obj)
    (args, _, _) = inspect.getargs(func_obj.__code__)
    if not args:
        return getargsfromdoc(obj)
    for (i_arg, arg) in enumerate(args):
        if isinstance(arg, list):
            args[i_arg] = '(%s)' % ', '.join(arg)
    defaults = func_obj.__defaults__
    if defaults is not None:
        for (index, default) in enumerate(defaults):
            args[index + len(args) - len(defaults)] += '=' + repr(default)
    if inspect.isclass(obj) or inspect.ismethod(obj):
        if len(args) == 1:
            return None
    if 'self' in args:
        args.remove('self')
    return args

def getargtxt(obj, one_arg_per_line=True):
    if False:
        while True:
            i = 10
    "\n    Get the names and default values of a function's arguments\n    Return list with separators (', ') formatted for calltips\n    "
    args = getargs(obj)
    if args:
        sep = ', '
        textlist = None
        for (i_arg, arg) in enumerate(args):
            if textlist is None:
                textlist = ['']
            textlist[-1] += arg
            if i_arg < len(args) - 1:
                textlist[-1] += sep
                if len(textlist[-1]) >= 32 or one_arg_per_line:
                    textlist.append('')
        if inspect.isclass(obj) or inspect.ismethod(obj):
            if len(textlist) == 1:
                return None
            if 'self' + sep in textlist:
                textlist.remove('self' + sep)
        return textlist

def isdefined(obj, force_import=False, namespace=None):
    if False:
        for i in range(10):
            print('nop')
    'Return True if object is defined in namespace\n    If namespace is None --> namespace = locals()'
    if namespace is None:
        namespace = locals()
    attr_list = obj.split('.')
    base = attr_list.pop(0)
    if len(base) == 0:
        return False
    if base not in builtins.__dict__ and base not in namespace:
        if force_import:
            try:
                module = __import__(base, globals(), namespace)
                if base not in globals():
                    globals()[base] = module
                namespace[base] = module
            except Exception:
                return False
        else:
            return False
    for attr in attr_list:
        try:
            attr_not_found = not hasattr(eval(base, namespace), attr)
        except (AttributeError, SyntaxError, TypeError):
            return False
        if attr_not_found:
            if force_import:
                try:
                    __import__(base + '.' + attr, globals(), namespace)
                except (ImportError, SyntaxError):
                    return False
            else:
                return False
        base += '.' + attr
    return True