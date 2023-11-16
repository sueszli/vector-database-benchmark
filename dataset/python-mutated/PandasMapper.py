"""
Lean Pandas Remapper
Wraps key indexing functions of Pandas to remap keys to SIDs when accessing dataframes.
Allowing support for indexing of Lean created Indexes with tickers like "SPY", Symbol objs, and SIDs

"""
import pandas as pd
from pandas.core.indexes.frozen import FrozenList as pdFrozenList
from clr import AddReference
AddReference('QuantConnect.Common')
from QuantConnect import *

def mapper(key):
    if False:
        i = 10
        return i + 15
    'Maps a Symbol object or a Symbol Ticker (string) to the string representation of\n    Symbol SecurityIdentifier.If cannot map, returns the object\n    '
    keyType = type(key)
    if keyType is Symbol:
        return str(key.ID)
    if keyType is str:
        reserved = ['high', 'low', 'open', 'close']
        if key in reserved:
            return key
        kvp = SymbolCache.TryGetSymbol(key, None)
        if kvp[0]:
            return str(kvp[1].ID)
    if keyType is list:
        return [mapper(x) for x in key]
    if keyType is tuple:
        return tuple([mapper(x) for x in key])
    if keyType is dict:
        return {k: mapper(v) for (k, v) in key.items()}
    return key

def wrap_keyerror_function(f):
    if False:
        while True:
            i = 10
    'Wraps function f with wrapped_function, used for functions that throw KeyError when not found.\n    wrapped_function converts the args / kwargs to use alternative index keys and then calls the function. \n    If this fails we fall back to the original key and try it as well, if they both fail we throw our error.\n    '

    def wrapped_function(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            newargs = args
            newkwargs = kwargs
            if len(args) > 1:
                newargs = mapper(args)
            if len(kwargs) > 0:
                newkwargs = mapper(kwargs)
            return f(*newargs, **newkwargs)
        except KeyError as e:
            mKey = [arg for arg in newargs if isinstance(arg, str)]
        try:
            return f(*args, **kwargs)
        except KeyError as e:
            oKey = [arg for arg in args if isinstance(arg, str)]
            raise KeyError(f'No key found for either mapped or original key. Mapped Key: {mKey}; Original Key: {oKey}')
    wrapped_function.__name__ = f.__name__
    return wrapped_function

def wrap_bool_function(f):
    if False:
        print('Hello World!')
    'Wraps function f with wrapped_function, used for functions that reply true/false if key is found.\n    wrapped_function attempts with the original args, if its false, it converts the args / kwargs to use\n    alternative index keys and then attempts with the mapped args.\n    '

    def wrapped_function(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        originalResult = f(*args, **kwargs)
        if originalResult:
            return originalResult
        newargs = args
        newkwargs = kwargs
        if len(args) > 1:
            newargs = mapper(args)
        if len(kwargs) > 0:
            newkwargs = mapper(kwargs)
        return f(*newargs, **newkwargs)
    wrapped_function.__name__ = f.__name__
    return wrapped_function
pd.core.indexing._LocationIndexer.__getitem__ = wrap_keyerror_function(pd.core.indexing._LocationIndexer.__getitem__)
pd.core.indexing._ScalarAccessIndexer.__getitem__ = wrap_keyerror_function(pd.core.indexing._ScalarAccessIndexer.__getitem__)
pd.core.indexes.base.Index.get_loc = wrap_keyerror_function(pd.core.indexes.base.Index.get_loc)
pd.core.frame.DataFrame.__getitem__ = wrap_keyerror_function(pd.core.frame.DataFrame.__getitem__)
if int(pd.__version__.split('.')[0]) < 1:
    pd.core.indexes.base.Index.get_value = wrap_keyerror_function(pd.core.indexes.base.Index.get_value)
pd.core.indexes.base.Index.__contains__ = wrap_bool_function(pd.core.indexes.base.Index.__contains__)
FrozenList = pdFrozenList
Index = pd.Index
MultiIndex = pd.MultiIndex
Series = pd.Series
DataFrame = pd.DataFrame