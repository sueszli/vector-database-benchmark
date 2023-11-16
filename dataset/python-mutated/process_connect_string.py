""" a clumsy attempt at a macro language to let the programmer execute code on the server (ex: determine 64bit)"""
from . import is64bit as is64bit

def macro_call(macro_name, args, kwargs):
    if False:
        print('Hello World!')
    "allow the programmer to perform limited processing on the server by passing macro names and args\n\n    :new_key - the key name the macro will create\n    :args[0] - macro name\n    :args[1:] - any arguments\n    :code - the value of the keyword item\n    :kwargs - the connection keyword dictionary. ??key has been removed\n    --> the value to put in for kwargs['name'] = value\n    "
    if isinstance(args, (str, str)):
        args = [args]
    new_key = args[0]
    try:
        if macro_name == 'is64bit':
            if is64bit.Python():
                return (new_key, args[1])
            else:
                try:
                    return (new_key, args[2])
                except IndexError:
                    return (new_key, '')
        elif macro_name == 'getuser':
            if not new_key in kwargs:
                import getpass
                return (new_key, getpass.getuser())
        elif macro_name == 'getnode':
            import platform
            try:
                return (new_key, args[1] % platform.node())
            except IndexError:
                return (new_key, platform.node())
        elif macro_name == 'getenv':
            try:
                dflt = args[2]
            except IndexError:
                dflt = ''
            return (new_key, os.environ.get(args[1], dflt))
        elif macro_name == 'auto_security':
            if not 'user' in kwargs or not kwargs['user']:
                return (new_key, 'Integrated Security=SSPI')
            return (new_key, 'User ID=%(user)s; Password=%(password)s' % kwargs)
        elif macro_name == 'find_temp_test_path':
            import os
            import tempfile
            return (new_key, os.path.join(tempfile.gettempdir(), 'adodbapi_test', args[1]))
        raise ValueError('Unknown connect string macro=%s' % macro_name)
    except:
        raise ValueError('Error in macro processing %s %s' % (macro_name, repr(args)))

def process(args, kwargs, expand_macros=False):
    if False:
        while True:
            i = 10
    'attempts to inject arguments into a connection string using Python "%" operator for strings\n\n    co: adodbapi connection object\n    args: positional parameters from the .connect() call\n    kvargs: keyword arguments from the .connect() call\n    '
    try:
        dsn = args[0]
    except IndexError:
        dsn = None
    if isinstance(dsn, dict):
        kwargs.update(dsn)
    elif dsn:
        kwargs['connection_string'] = dsn
    try:
        a1 = args[1]
    except IndexError:
        a1 = None
    if isinstance(a1, int):
        kwargs['timeout'] = a1
    elif isinstance(a1, str):
        kwargs['user'] = a1
    elif isinstance(a1, dict):
        kwargs.update(a1)
    try:
        kwargs['password'] = args[2]
        kwargs['host'] = args[3]
        kwargs['database'] = args[4]
    except IndexError:
        pass
    if not 'connection_string' in kwargs:
        try:
            kwargs['connection_string'] = kwargs['dsn']
        except KeyError:
            try:
                kwargs['connection_string'] = kwargs['host']
            except KeyError:
                raise TypeError("Must define 'connection_string' for ado connections")
    if expand_macros:
        for kwarg in list(kwargs.keys()):
            if kwarg.startswith('macro_'):
                macro_name = kwarg[6:]
                macro_code = kwargs.pop(kwarg)
                (new_key, rslt) = macro_call(macro_name, macro_code, kwargs)
                kwargs[new_key] = rslt
    try:
        s = kwargs['proxy_host']
        if ':' in s:
            if s[0] != '[':
                kwargs['proxy_host'] = s.join(('[', ']'))
    except KeyError:
        pass
    return kwargs