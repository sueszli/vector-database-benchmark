__revision__ = 'src/engine/SCons/PathList.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__doc__ = 'SCons.PathList\n\nA module for handling lists of directory paths (the sort of things\nthat get set as CPPPATH, LIBPATH, etc.) with as much caching of data and\nefficiency as we can, while still keeping the evaluation delayed so that we\nDo the Right Thing (almost) regardless of how the variable is specified.\n\n'
import os
import SCons.Memoize
import SCons.Node
import SCons.Util
TYPE_STRING_NO_SUBST = 0
TYPE_STRING_SUBST = 1
TYPE_OBJECT = 2

def node_conv(obj):
    if False:
        while True:
            i = 10
    '\n    This is the "string conversion" routine that we have our substitutions\n    use to return Nodes, not strings.  This relies on the fact that an\n    EntryProxy object has a get() method that returns the underlying\n    Node that it wraps, which is a bit of architectural dependence\n    that we might need to break or modify in the future in response to\n    additional requirements.\n    '
    try:
        get = obj.get
    except AttributeError:
        if isinstance(obj, SCons.Node.Node) or SCons.Util.is_Sequence(obj):
            result = obj
        else:
            result = str(obj)
    else:
        result = get()
    return result

class _PathList(object):
    """
    An actual PathList object.
    """

    def __init__(self, pathlist):
        if False:
            i = 10
            return i + 15
        '\n        Initializes a PathList object, canonicalizing the input and\n        pre-processing it for quicker substitution later.\n\n        The stored representation of the PathList is a list of tuples\n        containing (type, value), where the "type" is one of the TYPE_*\n        variables defined above.  We distinguish between:\n\n            strings that contain no \'$\' and therefore need no\n            delayed-evaluation string substitution (we expect that there\n            will be many of these and that we therefore get a pretty\n            big win from avoiding string substitution)\n\n            strings that contain \'$\' and therefore need substitution\n            (the hard case is things like \'${TARGET.dir}/include\',\n            which require re-evaluation for every target + source)\n\n            other objects (which may be something like an EntryProxy\n            that needs a method called to return a Node)\n\n        Pre-identifying the type of each element in the PathList up-front\n        and storing the type in the list of tuples is intended to reduce\n        the amount of calculation when we actually do the substitution\n        over and over for each target.\n        '
        if SCons.Util.is_String(pathlist):
            pathlist = pathlist.split(os.pathsep)
        elif not SCons.Util.is_Sequence(pathlist):
            pathlist = [pathlist]
        pl = []
        for p in pathlist:
            try:
                found = '$' in p
            except (AttributeError, TypeError):
                type = TYPE_OBJECT
            else:
                if not found:
                    type = TYPE_STRING_NO_SUBST
                else:
                    type = TYPE_STRING_SUBST
            pl.append((type, p))
        self.pathlist = tuple(pl)

    def __len__(self):
        if False:
            return 10
        return len(self.pathlist)

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        return self.pathlist[i]

    def subst_path(self, env, target, source):
        if False:
            i = 10
            return i + 15
        '\n        Performs construction variable substitution on a pre-digested\n        PathList for a specific target and source.\n        '
        result = []
        for (type, value) in self.pathlist:
            if type == TYPE_STRING_SUBST:
                value = env.subst(value, target=target, source=source, conv=node_conv)
                if SCons.Util.is_Sequence(value):
                    result.extend(SCons.Util.flatten(value))
                elif value:
                    result.append(value)
            elif type == TYPE_OBJECT:
                value = node_conv(value)
                if value:
                    result.append(value)
            elif value:
                result.append(value)
        return tuple(result)

class PathListCache(object):
    """
    A class to handle caching of PathList lookups.

    This class gets instantiated once and then deleted from the namespace,
    so it's used as a Singleton (although we don't enforce that in the
    usual Pythonic ways).  We could have just made the cache a dictionary
    in the module namespace, but putting it in this class allows us to
    use the same Memoizer pattern that we use elsewhere to count cache
    hits and misses, which is very valuable.

    Lookup keys in the cache are computed by the _PathList_key() method.
    Cache lookup should be quick, so we don't spend cycles canonicalizing
    all forms of the same lookup key.  For example, 'x:y' and ['x',
    'y'] logically represent the same list, but we don't bother to
    split string representations and treat those two equivalently.
    (Note, however, that we do, treat lists and tuples the same.)

    The main type of duplication we're trying to catch will come from
    looking up the same path list from two different clones of the
    same construction environment.  That is, given
    
        env2 = env1.Clone()

    both env1 and env2 will have the same CPPPATH value, and we can
    cheaply avoid re-parsing both values of CPPPATH by using the
    common value from this cache.
    """

    def __init__(self):
        if False:
            return 10
        self._memo = {}

    def _PathList_key(self, pathlist):
        if False:
            print('Hello World!')
        "\n        Returns the key for memoization of PathLists.\n\n        Note that we want this to be pretty quick, so we don't completely\n        canonicalize all forms of the same list.  For example,\n        'dir1:$ROOT/dir2' and ['$ROOT/dir1', 'dir'] may logically\n        represent the same list if you're executing from $ROOT, but\n        we're not going to bother splitting strings into path elements,\n        or massaging strings into Nodes, to identify that equivalence.\n        We just want to eliminate obvious redundancy from the normal\n        case of re-using exactly the same cloned value for a path.\n        "
        if SCons.Util.is_Sequence(pathlist):
            pathlist = tuple(SCons.Util.flatten(pathlist))
        return pathlist

    @SCons.Memoize.CountDictCall(_PathList_key)
    def PathList(self, pathlist):
        if False:
            return 10
        '\n        Returns the cached _PathList object for the specified pathlist,\n        creating and caching a new object as necessary.\n        '
        pathlist = self._PathList_key(pathlist)
        try:
            memo_dict = self._memo['PathList']
        except KeyError:
            memo_dict = {}
            self._memo['PathList'] = memo_dict
        else:
            try:
                return memo_dict[pathlist]
            except KeyError:
                pass
        result = _PathList(pathlist)
        memo_dict[pathlist] = result
        return result
PathList = PathListCache().PathList
del PathListCache