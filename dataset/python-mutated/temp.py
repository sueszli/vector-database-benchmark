"""
Methods for serialized objects (or source code) stored in temporary files
and file-like objects.
"""
__all__ = ['dump_source', 'dump', 'dumpIO_source', 'dumpIO', 'load_source', 'load', 'loadIO_source', 'loadIO', 'capture']
import contextlib
from ._dill import PY3

@contextlib.contextmanager
def capture(stream='stdout'):
    if False:
        return 10
    'builds a context that temporarily replaces the given stream name\n\n    >>> with capture(\'stdout\') as out:\n    ...   print "foo!"\n    ... \n    >>> print out.getvalue()\n    foo!\n\n    '
    import sys
    if PY3:
        from io import StringIO
    else:
        from StringIO import StringIO
    orig = getattr(sys, stream)
    setattr(sys, stream, StringIO())
    try:
        yield getattr(sys, stream)
    finally:
        setattr(sys, stream, orig)

def b(x):
    if False:
        while True:
            i = 10
    import codecs
    return codecs.latin_1_encode(x)[0]

def load_source(file, **kwds):
    if False:
        print('Hello World!')
    "load an object that was stored with dill.temp.dump_source\n\n    file: filehandle\n    alias: string name of stored object\n    mode: mode to open the file, one of: {'r', 'rb'}\n\n    >>> f = lambda x: x**2\n    >>> pyfile = dill.temp.dump_source(f, alias='_f')\n    >>> _f = dill.temp.load_source(pyfile)\n    >>> _f(4)\n    16\n    "
    alias = kwds.pop('alias', None)
    mode = kwds.pop('mode', 'r')
    fname = getattr(file, 'name', file)
    source = open(fname, mode=mode, **kwds).read()
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError('unknown name for code: %s' % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval('%s' % alias, local)
    return _

def dump_source(object, **kwds):
    if False:
        i = 10
        return i + 15
    'write object source to a NamedTemporaryFile (instead of dill.dump)\nLoads with "import" or "dill.temp.load_source".  Returns the filehandle.\n\n    >>> f = lambda x: x**2\n    >>> pyfile = dill.temp.dump_source(f, alias=\'_f\')\n    >>> _f = dill.temp.load_source(pyfile)\n    >>> _f(4)\n    16\n\n    >>> f = lambda x: x**2\n    >>> pyfile = dill.temp.dump_source(f, dir=\'.\')\n    >>> modulename = os.path.basename(pyfile.name).split(\'.py\')[0]\n    >>> exec(\'from %s import f as _f\' % modulename)\n    >>> _f(4)\n    16\n\nOptional kwds:\n    If \'alias\' is specified, the object will be renamed to the given string.\n\n    If \'prefix\' is specified, the file name will begin with that prefix,\n    otherwise a default prefix is used.\n    \n    If \'dir\' is specified, the file will be created in that directory,\n    otherwise a default directory is used.\n    \n    If \'text\' is specified and true, the file is opened in text\n    mode.  Else (the default) the file is opened in binary mode.  On\n    some operating systems, this makes no difference.\n\nNOTE: Keep the return value for as long as you want your file to exist !\n    '
    from .source import importable, getname
    import tempfile
    kwds.pop('suffix', '')
    alias = kwds.pop('alias', '')
    name = str(alias) or getname(object)
    name = '\n#NAME: %s\n' % name
    file = tempfile.NamedTemporaryFile(suffix='.py', **kwds)
    file.write(b(''.join([importable(object, alias=alias), name])))
    file.flush()
    return file

def load(file, **kwds):
    if False:
        return 10
    "load an object that was stored with dill.temp.dump\n\n    file: filehandle\n    mode: mode to open the file, one of: {'r', 'rb'}\n\n    >>> dumpfile = dill.temp.dump([1, 2, 3, 4, 5])\n    >>> dill.temp.load(dumpfile)\n    [1, 2, 3, 4, 5]\n    "
    import dill as pickle
    mode = kwds.pop('mode', 'rb')
    name = getattr(file, 'name', file)
    return pickle.load(open(name, mode=mode, **kwds))

def dump(object, **kwds):
    if False:
        for i in range(10):
            print('nop')
    'dill.dump of object to a NamedTemporaryFile.\nLoads with "dill.temp.load".  Returns the filehandle.\n\n    >>> dumpfile = dill.temp.dump([1, 2, 3, 4, 5])\n    >>> dill.temp.load(dumpfile)\n    [1, 2, 3, 4, 5]\n\nOptional kwds:\n    If \'suffix\' is specified, the file name will end with that suffix,\n    otherwise there will be no suffix.\n    \n    If \'prefix\' is specified, the file name will begin with that prefix,\n    otherwise a default prefix is used.\n    \n    If \'dir\' is specified, the file will be created in that directory,\n    otherwise a default directory is used.\n    \n    If \'text\' is specified and true, the file is opened in text\n    mode.  Else (the default) the file is opened in binary mode.  On\n    some operating systems, this makes no difference.\n\nNOTE: Keep the return value for as long as you want your file to exist !\n    '
    import dill as pickle
    import tempfile
    file = tempfile.NamedTemporaryFile(**kwds)
    pickle.dump(object, file)
    file.flush()
    return file

def loadIO(buffer, **kwds):
    if False:
        return 10
    'load an object that was stored with dill.temp.dumpIO\n\n    buffer: buffer object\n\n    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])\n    >>> dill.temp.loadIO(dumpfile)\n    [1, 2, 3, 4, 5]\n    '
    import dill as pickle
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    value = getattr(buffer, 'getvalue', buffer)
    if value != buffer:
        value = value()
    return pickle.load(StringIO(value))

def dumpIO(object, **kwds):
    if False:
        for i in range(10):
            print('nop')
    'dill.dump of object to a buffer.\nLoads with "dill.temp.loadIO".  Returns the buffer object.\n\n    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])\n    >>> dill.temp.loadIO(dumpfile)\n    [1, 2, 3, 4, 5]\n    '
    import dill as pickle
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    file = StringIO()
    pickle.dump(object, file)
    file.flush()
    return file

def loadIO_source(buffer, **kwds):
    if False:
        return 10
    "load an object that was stored with dill.temp.dumpIO_source\n\n    buffer: buffer object\n    alias: string name of stored object\n\n    >>> f = lambda x:x**2\n    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')\n    >>> _f = dill.temp.loadIO_source(pyfile)\n    >>> _f(4)\n    16\n    "
    alias = kwds.pop('alias', None)
    source = getattr(buffer, 'getvalue', buffer)
    if source != buffer:
        source = source()
    if PY3:
        source = source.decode()
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError('unknown name for code: %s' % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval('%s' % alias, local)
    return _

def dumpIO_source(object, **kwds):
    if False:
        for i in range(10):
            print('nop')
    "write object source to a buffer (instead of dill.dump)\nLoads by with dill.temp.loadIO_source.  Returns the buffer object.\n\n    >>> f = lambda x:x**2\n    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')\n    >>> _f = dill.temp.loadIO_source(pyfile)\n    >>> _f(4)\n    16\n\nOptional kwds:\n    If 'alias' is specified, the object will be renamed to the given string.\n    "
    from .source import importable, getname
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    alias = kwds.pop('alias', '')
    name = str(alias) or getname(object)
    name = '\n#NAME: %s\n' % name
    file = StringIO()
    file.write(b(''.join([importable(object, alias=alias), name])))
    file.flush()
    return file
del contextlib