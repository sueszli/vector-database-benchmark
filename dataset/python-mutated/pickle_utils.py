"""Pickle file loading and saving."""
import gzip
import pickle
import sys
from pytype.pytd import serialize_ast
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
_PICKLE_RECURSION_LIMIT_AST = 40000

def LoadAst(data):
    if False:
        while True:
            i = 10
    'Load data that has been read from a pickled file.'
    return pickle.loads(data)

class LoadPickleError(Exception):
    """Errors when loading a pickled pytd file."""

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename
        msg = f'Error loading pickle file: {filename}'
        super().__init__(msg)

def _LoadPickle(f, filename):
    if False:
        while True:
            i = 10
    'Load a pickle file, raising a custom exception on failure.'
    try:
        return pickle.load(f)
    except Exception as e:
        raise LoadPickleError(filename) from e

def LoadPickle(filename, compress=False, open_function=open):
    if False:
        i = 10
        return i + 15
    with open_function(filename, 'rb') as fi:
        if compress:
            with gzip.GzipFile(fileobj=fi) as zfi:
                return _LoadPickle(zfi, filename)
        else:
            return _LoadPickle(fi, filename)

def SavePickle(data, filename=None, compress=False, open_function=open):
    if False:
        print('Hello World!')
    'Pickle the data.'
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(_PICKLE_RECURSION_LIMIT_AST)
    assert not compress or filename, 'gzip only supported with a filename'
    try:
        if compress:
            with open_function(filename, mode='wb') as fi:
                with gzip.GzipFile(filename='', mode='wb', fileobj=fi, mtime=1.0) as zfi:
                    pickle.dump(data, zfi, _PICKLE_PROTOCOL)
        elif filename is not None:
            with open_function(filename, 'wb') as fi:
                pickle.dump(data, fi, _PICKLE_PROTOCOL)
        else:
            return pickle.dumps(data, _PICKLE_PROTOCOL)
    finally:
        sys.setrecursionlimit(recursion_limit)

def StoreAst(ast, filename=None, open_function=open, src_path=None, metadata=None):
    if False:
        i = 10
        return i + 15
    'Loads and stores an ast to disk.\n\n  Args:\n    ast: The pytd.TypeDeclUnit to save to disk.\n    filename: The filename for the pickled output. If this is None, this\n      function instead returns the pickled string.\n    open_function: A custom file opening function.\n    src_path: Optionally, the filepath of the original source file.\n    metadata: A list of arbitrary string-encoded metadata.\n\n  Returns:\n    The pickled string, if no filename was given. (None otherwise.)\n  '
    out = serialize_ast.SerializeAst(ast, src_path, metadata)
    return SavePickle(out, filename, open_function=open_function)