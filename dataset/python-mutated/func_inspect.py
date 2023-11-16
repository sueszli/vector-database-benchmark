"""
My own variation on function-specific inspect-like features.
"""
import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
full_argspec_fields = 'args varargs varkw defaults kwonlyargs kwonlydefaults annotations'
full_argspec_type = collections.namedtuple('FullArgSpec', full_argspec_fields)

def get_func_code(func):
    if False:
        while True:
            i = 10
    " Attempts to retrieve a reliable function code hash.\n\n        The reason we don't use inspect.getsource is that it caches the\n        source, whereas we want this to be modified on the fly when the\n        function is modified.\n\n        Returns\n        -------\n        func_code: string\n            The function code\n        source_file: string\n            The path to the file in which the function is defined.\n        first_line: int\n            The first line of the code in the source file.\n\n        Notes\n        ------\n        This function does a bit more magic than inspect, and is thus\n        more robust.\n    "
    source_file = None
    try:
        code = func.__code__
        source_file = code.co_filename
        if not os.path.exists(source_file):
            source_code = ''.join(inspect.getsourcelines(func)[0])
            line_no = 1
            if source_file.startswith('<doctest '):
                (source_file, line_no) = re.match('\\<doctest (.*\\.rst)\\[(.*)\\]\\>', source_file).groups()
                line_no = int(line_no)
                source_file = '<doctest %s>' % source_file
            return (source_code, source_file, line_no)
        with open_py_source(source_file) as source_file_obj:
            first_line = code.co_firstlineno
            source_lines = list(islice(source_file_obj, first_line - 1, None))
        return (''.join(inspect.getblock(source_lines)), source_file, first_line)
    except:
        if hasattr(func, '__code__'):
            return (str(func.__code__.__hash__()), source_file, -1)
        else:
            return (repr(func), source_file, -1)

def _clean_win_chars(string):
    if False:
        i = 10
        return i + 15
    'Windows cannot encode some characters in filename.'
    import urllib
    if hasattr(urllib, 'quote'):
        quote = urllib.quote
    else:
        import urllib.parse
        quote = urllib.parse.quote
    for char in ('<', '>', '!', ':', '\\'):
        string = string.replace(char, quote(char))
    return string

def get_func_name(func, resolv_alias=True, win_characters=True):
    if False:
        for i in range(10):
            print('nop')
    ' Return the function import path (as a list of module names), and\n        a name for the function.\n\n        Parameters\n        ----------\n        func: callable\n            The func to inspect\n        resolv_alias: boolean, optional\n            If true, possible local aliases are indicated.\n        win_characters: boolean, optional\n            If true, substitute special characters using urllib.quote\n            This is useful in Windows, as it cannot encode some filenames\n    '
    if hasattr(func, '__module__'):
        module = func.__module__
    else:
        try:
            module = inspect.getmodule(func)
        except TypeError:
            if hasattr(func, '__class__'):
                module = func.__class__.__module__
            else:
                module = 'unknown'
    if module is None:
        module = ''
    if module == '__main__':
        try:
            filename = os.path.abspath(inspect.getsourcefile(func))
        except:
            filename = None
        if filename is not None:
            parts = filename.split(os.sep)
            if parts[-1].startswith('<ipython-input'):
                splitted = parts[-1].split('-')
                parts[-1] = '-'.join(splitted[:2] + splitted[3:])
            elif len(parts) > 2 and parts[-2].startswith('ipykernel_'):
                parts[-2] = 'ipykernel'
            filename = '-'.join(parts)
            if filename.endswith('.py'):
                filename = filename[:-3]
            module = module + '-' + filename
    module = module.split('.')
    if hasattr(func, 'func_name'):
        name = func.func_name
    elif hasattr(func, '__name__'):
        name = func.__name__
    else:
        name = 'unknown'
    if resolv_alias:
        if hasattr(func, 'func_globals') and name in func.func_globals:
            if not func.func_globals[name] is func:
                name = '%s-alias' % name
    if hasattr(func, '__qualname__') and func.__qualname__ != name:
        module.extend(func.__qualname__.split('.')[:-1])
    if inspect.ismethod(func):
        if hasattr(func, 'im_class'):
            klass = func.im_class
            module.append(klass.__name__)
    if os.name == 'nt' and win_characters:
        name = _clean_win_chars(name)
        module = [_clean_win_chars(s) for s in module]
    return (module, name)

def _signature_str(function_name, arg_sig):
    if False:
        print('Hello World!')
    'Helper function to output a function signature'
    return '{}{}'.format(function_name, arg_sig)

def _function_called_str(function_name, args, kwargs):
    if False:
        i = 10
        return i + 15
    'Helper function to output a function call'
    template_str = '{0}({1}, {2})'
    args_str = repr(args)[1:-1]
    kwargs_str = ', '.join(('%s=%s' % (k, v) for (k, v) in kwargs.items()))
    return template_str.format(function_name, args_str, kwargs_str)

def filter_args(func, ignore_lst, args=(), kwargs=dict()):
    if False:
        for i in range(10):
            print('nop')
    " Filters the given args and kwargs using a list of arguments to\n        ignore, and a function specification.\n\n        Parameters\n        ----------\n        func: callable\n            Function giving the argument specification\n        ignore_lst: list of strings\n            List of arguments to ignore (either a name of an argument\n            in the function spec, or '*', or '**')\n        *args: list\n            Positional arguments passed to the function.\n        **kwargs: dict\n            Keyword arguments passed to the function\n\n        Returns\n        -------\n        filtered_args: list\n            List of filtered positional and keyword arguments.\n    "
    args = list(args)
    if isinstance(ignore_lst, str):
        raise ValueError('ignore_lst must be a list of parameters to ignore %s (type %s) was given' % (ignore_lst, type(ignore_lst)))
    if not inspect.ismethod(func) and (not inspect.isfunction(func)):
        if ignore_lst:
            warnings.warn('Cannot inspect object %s, ignore list will not work.' % func, stacklevel=2)
        return {'*': args, '**': kwargs}
    arg_sig = inspect.signature(func)
    arg_names = []
    arg_defaults = []
    arg_kwonlyargs = []
    arg_varargs = None
    arg_varkw = None
    for param in arg_sig.parameters.values():
        if param.kind is param.POSITIONAL_OR_KEYWORD:
            arg_names.append(param.name)
        elif param.kind is param.KEYWORD_ONLY:
            arg_names.append(param.name)
            arg_kwonlyargs.append(param.name)
        elif param.kind is param.VAR_POSITIONAL:
            arg_varargs = param.name
        elif param.kind is param.VAR_KEYWORD:
            arg_varkw = param.name
        if param.default is not param.empty:
            arg_defaults.append(param.default)
    if inspect.ismethod(func):
        args = [func.__self__] + args
        class_method_sig = inspect.signature(func.__func__)
        self_name = next(iter(class_method_sig.parameters))
        arg_names = [self_name] + arg_names
    (_, name) = get_func_name(func, resolv_alias=False)
    arg_dict = dict()
    arg_position = -1
    for (arg_position, arg_name) in enumerate(arg_names):
        if arg_position < len(args):
            if arg_name not in arg_kwonlyargs:
                arg_dict[arg_name] = args[arg_position]
            else:
                raise ValueError("Keyword-only parameter '%s' was passed as positional parameter for %s:\n     %s was called." % (arg_name, _signature_str(name, arg_sig), _function_called_str(name, args, kwargs)))
        else:
            position = arg_position - len(arg_names)
            if arg_name in kwargs:
                arg_dict[arg_name] = kwargs[arg_name]
            else:
                try:
                    arg_dict[arg_name] = arg_defaults[position]
                except (IndexError, KeyError) as e:
                    raise ValueError('Wrong number of arguments for %s:\n     %s was called.' % (_signature_str(name, arg_sig), _function_called_str(name, args, kwargs))) from e
    varkwargs = dict()
    for (arg_name, arg_value) in sorted(kwargs.items()):
        if arg_name in arg_dict:
            arg_dict[arg_name] = arg_value
        elif arg_varkw is not None:
            varkwargs[arg_name] = arg_value
        else:
            raise TypeError("Ignore list for %s() contains an unexpected keyword argument '%s'" % (name, arg_name))
    if arg_varkw is not None:
        arg_dict['**'] = varkwargs
    if arg_varargs is not None:
        varargs = args[arg_position + 1:]
        arg_dict['*'] = varargs
    for item in ignore_lst:
        if item in arg_dict:
            arg_dict.pop(item)
        else:
            raise ValueError("Ignore list: argument '%s' is not defined for function %s" % (item, _signature_str(name, arg_sig)))
    return arg_dict

def _format_arg(arg):
    if False:
        i = 10
        return i + 15
    formatted_arg = pformat(arg, indent=2)
    if len(formatted_arg) > 1500:
        formatted_arg = '%s...' % formatted_arg[:700]
    return formatted_arg

def format_signature(func, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (module, name) = get_func_name(func)
    module = [m for m in module if m]
    if module:
        module.append(name)
        module_path = '.'.join(module)
    else:
        module_path = name
    arg_str = list()
    previous_length = 0
    for arg in args:
        formatted_arg = _format_arg(arg)
        if previous_length > 80:
            formatted_arg = '\n%s' % formatted_arg
        previous_length = len(formatted_arg)
        arg_str.append(formatted_arg)
    arg_str.extend(['%s=%s' % (v, _format_arg(i)) for (v, i) in kwargs.items()])
    arg_str = ', '.join(arg_str)
    signature = '%s(%s)' % (name, arg_str)
    return (module_path, signature)

def format_call(func, args, kwargs, object_name='Memory'):
    if False:
        while True:
            i = 10
    ' Returns a nicely formatted statement displaying the function\n        call with the given arguments.\n    '
    (path, signature) = format_signature(func, *args, **kwargs)
    msg = '%s\n[%s] Calling %s...\n%s' % (80 * '_', object_name, path, signature)
    return msg