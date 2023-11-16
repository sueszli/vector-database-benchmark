"""
Basic data classes for representing feature structures, and for
performing basic operations on those feature structures.  A feature
structure is a mapping from feature identifiers to feature values,
where each feature value is either a basic value (such as a string or
an integer), or a nested feature structure.  There are two types of
feature structure, implemented by two subclasses of ``FeatStruct``:

    - feature dictionaries, implemented by ``FeatDict``, act like
      Python dictionaries.  Feature identifiers may be strings or
      instances of the ``Feature`` class.
    - feature lists, implemented by ``FeatList``, act like Python
      lists.  Feature identifiers are integers.

Feature structures are typically used to represent partial information
about objects.  A feature identifier that is not mapped to a value
stands for a feature whose value is unknown (*not* a feature without
a value).  Two feature structures that represent (potentially
overlapping) information about the same object can be combined by
unification.  When two inconsistent feature structures are unified,
the unification fails and returns None.

Features can be specified using "feature paths", or tuples of feature
identifiers that specify path through the nested feature structures to
a value.  Feature structures may contain reentrant feature values.  A
"reentrant feature value" is a single feature value that can be
accessed via multiple feature paths.  Unification preserves the
reentrance relations imposed by both of the unified feature
structures.  In the feature structure resulting from unification, any
modifications to a reentrant feature value will be visible using any
of its feature paths.

Feature structure variables are encoded using the ``nltk.sem.Variable``
class.  The variables' values are tracked using a bindings
dictionary, which maps variables to their values.  When two feature
structures are unified, a fresh bindings dictionary is created to
track their values; and before unification completes, all bound
variables are replaced by their values.  Thus, the bindings
dictionaries are usually strictly internal to the unification process.
However, it is possible to track the bindings of variables if you
choose to, by supplying your own initial bindings dictionary to the
``unify()`` function.

When unbound variables are unified with one another, they become
aliased.  This is encoded by binding one variable to the other.

Lightweight Feature Structures
==============================
Many of the functions defined by ``nltk.featstruct`` can be applied
directly to simple Python dictionaries and lists, rather than to
full-fledged ``FeatDict`` and ``FeatList`` objects.  In other words,
Python ``dicts`` and ``lists`` can be used as "light-weight" feature
structures.

    >>> from nltk.featstruct import unify
    >>> unify(dict(x=1, y=dict()), dict(a='a', y=dict(b='b')))  # doctest: +SKIP
    {'y': {'b': 'b'}, 'x': 1, 'a': 'a'}

However, you should keep in mind the following caveats:

  - Python dictionaries & lists ignore reentrance when checking for
    equality between values.  But two FeatStructs with different
    reentrances are considered nonequal, even if all their base
    values are equal.

  - FeatStructs can be easily frozen, allowing them to be used as
    keys in hash tables.  Python dictionaries and lists can not.

  - FeatStructs display reentrance in their string representations;
    Python dictionaries and lists do not.

  - FeatStructs may *not* be mixed with Python dictionaries and lists
    (e.g., when performing unification).

  - FeatStructs provide a number of useful methods, such as ``walk()``
    and ``cyclic()``, which are not available for Python dicts and lists.

In general, if your feature structures will contain any reentrances,
or if you plan to use them as dictionary keys, it is strongly
recommended that you use full-fledged ``FeatStruct`` objects.
"""
import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import Expression, LogicalExpressionException, LogicParser, SubstituteBindingsI, Variable

@total_ordering
class FeatStruct(SubstituteBindingsI):
    """
    A mapping from feature identifiers to feature values, where each
    feature value is either a basic value (such as a string or an
    integer), or a nested feature structure.  There are two types of
    feature structure:

      - feature dictionaries, implemented by ``FeatDict``, act like
        Python dictionaries.  Feature identifiers may be strings or
        instances of the ``Feature`` class.
      - feature lists, implemented by ``FeatList``, act like Python
        lists.  Feature identifiers are integers.

    Feature structures may be indexed using either simple feature
    identifiers or 'feature paths.'  A feature path is a sequence
    of feature identifiers that stand for a corresponding sequence of
    indexing operations.  In particular, ``fstruct[(f1,f2,...,fn)]`` is
    equivalent to ``fstruct[f1][f2]...[fn]``.

    Feature structures may contain reentrant feature structures.  A
    "reentrant feature structure" is a single feature structure
    object that can be accessed via multiple feature paths.  Feature
    structures may also be cyclic.  A feature structure is "cyclic"
    if there is any feature path from the feature structure to itself.

    Two feature structures are considered equal if they assign the
    same values to all features, and have the same reentrancies.

    By default, feature structures are mutable.  They may be made
    immutable with the ``freeze()`` method.  Once they have been
    frozen, they may be hashed, and thus used as dictionary keys.
    """
    _frozen = False
    ':ivar: A flag indicating whether this feature structure is\n       frozen or not.  Once this flag is set, it should never be\n       un-set; and no further modification should be made to this\n       feature structure.'

    def __new__(cls, features=None, **morefeatures):
        if False:
            print('Hello World!')
        '\n        Construct and return a new feature structure.  If this\n        constructor is called directly, then the returned feature\n        structure will be an instance of either the ``FeatDict`` class\n        or the ``FeatList`` class.\n\n        :param features: The initial feature values for this feature\n            structure:\n\n            - FeatStruct(string) -> FeatStructReader().read(string)\n            - FeatStruct(mapping) -> FeatDict(mapping)\n            - FeatStruct(sequence) -> FeatList(sequence)\n            - FeatStruct() -> FeatDict()\n        :param morefeatures: If ``features`` is a mapping or None,\n            then ``morefeatures`` provides additional features for the\n            ``FeatDict`` constructor.\n        '
        if cls is FeatStruct:
            if features is None:
                return FeatDict.__new__(FeatDict, **morefeatures)
            elif _is_mapping(features):
                return FeatDict.__new__(FeatDict, features, **morefeatures)
            elif morefeatures:
                raise TypeError('Keyword arguments may only be specified if features is None or is a mapping.')
            if isinstance(features, str):
                if FeatStructReader._START_FDICT_RE.match(features):
                    return FeatDict.__new__(FeatDict, features, **morefeatures)
                else:
                    return FeatList.__new__(FeatList, features, **morefeatures)
            elif _is_sequence(features):
                return FeatList.__new__(FeatList, features)
            else:
                raise TypeError('Expected string or mapping or sequence')
        else:
            return super().__new__(cls, features, **morefeatures)

    def _keys(self):
        if False:
            i = 10
            return i + 15
        'Return an iterable of the feature identifiers used by this\n        FeatStruct.'
        raise NotImplementedError()

    def _values(self):
        if False:
            while True:
                i = 10
        'Return an iterable of the feature values directly defined\n        by this FeatStruct.'
        raise NotImplementedError()

    def _items(self):
        if False:
            while True:
                i = 10
        'Return an iterable of (fid,fval) pairs, where fid is a\n        feature identifier and fval is the corresponding feature\n        value, for all features defined by this FeatStruct.'
        raise NotImplementedError()

    def equal_values(self, other, check_reentrance=False):
        if False:
            return 10
        '\n        Return True if ``self`` and ``other`` assign the same value to\n        to every feature.  In particular, return true if\n        ``self[p]==other[p]`` for every feature path *p* such\n        that ``self[p]`` or ``other[p]`` is a base value (i.e.,\n        not a nested feature structure).\n\n        :param check_reentrance: If True, then also return False if\n            there is any difference between the reentrances of ``self``\n            and ``other``.\n        :note: the ``==`` is equivalent to ``equal_values()`` with\n            ``check_reentrance=True``.\n        '
        return self._equal(other, check_reentrance, set(), set(), set())

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        '\n        Return true if ``self`` and ``other`` are both feature structures,\n        assign the same values to all features, and contain the same\n        reentrances.  I.e., return\n        ``self.equal_values(other, check_reentrance=True)``.\n\n        :see: ``equal_values()``\n        '
        return self._equal(other, True, set(), set(), set())

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, FeatStruct):
            return self.__class__.__name__ < other.__class__.__name__
        else:
            return len(self) < len(other)

    def __hash__(self):
        if False:
            while True:
                i = 10
        '\n        If this feature structure is frozen, return its hash value;\n        otherwise, raise ``TypeError``.\n        '
        if not self._frozen:
            raise TypeError('FeatStructs must be frozen before they can be hashed.')
        try:
            return self._hash
        except AttributeError:
            self._hash = self._calculate_hashvalue(set())
            return self._hash

    def _equal(self, other, check_reentrance, visited_self, visited_other, visited_pairs):
        if False:
            return 10
        "\n        Return True iff self and other have equal values.\n\n        :param visited_self: A set containing the ids of all ``self``\n            feature structures we've already visited.\n        :param visited_other: A set containing the ids of all ``other``\n            feature structures we've already visited.\n        :param visited_pairs: A set containing ``(selfid, otherid)`` pairs\n            for all pairs of feature structures we've already visited.\n        "
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if len(self) != len(other):
            return False
        if set(self._keys()) != set(other._keys()):
            return False
        if check_reentrance:
            if id(self) in visited_self or id(other) in visited_other:
                return (id(self), id(other)) in visited_pairs
        elif (id(self), id(other)) in visited_pairs:
            return True
        visited_self.add(id(self))
        visited_other.add(id(other))
        visited_pairs.add((id(self), id(other)))
        for (fname, self_fval) in self._items():
            other_fval = other[fname]
            if isinstance(self_fval, FeatStruct):
                if not self_fval._equal(other_fval, check_reentrance, visited_self, visited_other, visited_pairs):
                    return False
            elif self_fval != other_fval:
                return False
        return True

    def _calculate_hashvalue(self, visited):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a hash value for this feature structure.\n\n        :require: ``self`` must be frozen.\n        :param visited: A set containing the ids of all feature\n            structures we've already visited while hashing.\n        "
        if id(self) in visited:
            return 1
        visited.add(id(self))
        hashval = 5831
        for (fname, fval) in sorted(self._items()):
            hashval *= 37
            hashval += hash(fname)
            hashval *= 37
            if isinstance(fval, FeatStruct):
                hashval += fval._calculate_hashvalue(visited)
            else:
                hashval += hash(fval)
            hashval = int(hashval & 2147483647)
        return hashval
    _FROZEN_ERROR = 'Frozen FeatStructs may not be modified.'

    def freeze(self):
        if False:
            while True:
                i = 10
        "\n        Make this feature structure, and any feature structures it\n        contains, immutable.  Note: this method does not attempt to\n        'freeze' any feature value that is not a ``FeatStruct``; it\n        is recommended that you use only immutable feature values.\n        "
        if self._frozen:
            return
        self._freeze(set())

    def frozen(self):
        if False:
            return 10
        '\n        Return True if this feature structure is immutable.  Feature\n        structures can be made immutable with the ``freeze()`` method.\n        Immutable feature structures may not be made mutable again,\n        but new mutable copies can be produced with the ``copy()`` method.\n        '
        return self._frozen

    def _freeze(self, visited):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make this feature structure, and any feature structure it\n        contains, immutable.\n\n        :param visited: A set containing the ids of all feature\n            structures we've already visited while freezing.\n        "
        if id(self) in visited:
            return
        visited.add(id(self))
        self._frozen = True
        for (fname, fval) in sorted(self._items()):
            if isinstance(fval, FeatStruct):
                fval._freeze(visited)

    def copy(self, deep=True):
        if False:
            while True:
                i = 10
        '\n        Return a new copy of ``self``.  The new copy will not be frozen.\n\n        :param deep: If true, create a deep copy; if false, create\n            a shallow copy.\n        '
        if deep:
            return copy.deepcopy(self)
        else:
            return self.__class__(self)

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def cyclic(self):
        if False:
            i = 10
            return i + 15
        '\n        Return True if this feature structure contains itself.\n        '
        return self._find_reentrances({})[id(self)]

    def walk(self):
        if False:
            i = 10
            return i + 15
        '\n        Return an iterator that generates this feature structure, and\n        each feature structure it contains.  Each feature structure will\n        be generated exactly once.\n        '
        return self._walk(set())

    def _walk(self, visited):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return an iterator that generates this feature structure, and\n        each feature structure it contains.\n\n        :param visited: A set containing the ids of all feature\n            structures we've already visited while freezing.\n        "
        raise NotImplementedError()

    def _walk(self, visited):
        if False:
            return 10
        if id(self) in visited:
            return
        visited.add(id(self))
        yield self
        for fval in self._values():
            if isinstance(fval, FeatStruct):
                yield from fval._walk(visited)

    def _find_reentrances(self, reentrances):
        if False:
            i = 10
            return i + 15
        '\n        Return a dictionary that maps from the ``id`` of each feature\n        structure contained in ``self`` (including ``self``) to a\n        boolean value, indicating whether it is reentrant or not.\n        '
        if id(self) in reentrances:
            reentrances[id(self)] = True
        else:
            reentrances[id(self)] = False
            for fval in self._values():
                if isinstance(fval, FeatStruct):
                    fval._find_reentrances(reentrances)
        return reentrances

    def substitute_bindings(self, bindings):
        if False:
            return 10
        ':see: ``nltk.featstruct.substitute_bindings()``'
        return substitute_bindings(self, bindings)

    def retract_bindings(self, bindings):
        if False:
            print('Hello World!')
        ':see: ``nltk.featstruct.retract_bindings()``'
        return retract_bindings(self, bindings)

    def variables(self):
        if False:
            print('Hello World!')
        ':see: ``nltk.featstruct.find_variables()``'
        return find_variables(self)

    def rename_variables(self, vars=None, used_vars=(), new_vars=None):
        if False:
            return 10
        ':see: ``nltk.featstruct.rename_variables()``'
        return rename_variables(self, vars, used_vars, new_vars)

    def remove_variables(self):
        if False:
            print('Hello World!')
        '\n        Return the feature structure that is obtained by deleting\n        any feature whose value is a ``Variable``.\n\n        :rtype: FeatStruct\n        '
        return remove_variables(self)

    def unify(self, other, bindings=None, trace=False, fail=None, rename_vars=True):
        if False:
            while True:
                i = 10
        return unify(self, other, bindings, trace, fail, rename_vars)

    def subsumes(self, other):
        if False:
            return 10
        '\n        Return True if ``self`` subsumes ``other``.  I.e., return true\n        If unifying ``self`` with ``other`` would result in a feature\n        structure equal to ``other``.\n        '
        return subsumes(self, other)

    def __repr__(self):
        if False:
            print('Hello World!')
        '\n        Display a single-line representation of this feature structure,\n        suitable for embedding in other representations.\n        '
        return self._repr(self._find_reentrances({}), {})

    def _repr(self, reentrances, reentrance_ids):
        if False:
            print('Hello World!')
        '\n        Return a string representation of this feature structure.\n\n        :param reentrances: A dictionary that maps from the ``id`` of\n            each feature value in self, indicating whether that value\n            is reentrant or not.\n        :param reentrance_ids: A dictionary mapping from each ``id``\n            of a feature value to a unique identifier.  This is modified\n            by ``repr``: the first time a reentrant feature value is\n            displayed, an identifier is added to ``reentrance_ids`` for it.\n        '
        raise NotImplementedError()
_FROZEN_ERROR = 'Frozen FeatStructs may not be modified.'
_FROZEN_NOTICE = '\n%sIf self is frozen, raise ValueError.'

def _check_frozen(method, indent=''):
    if False:
        i = 10
        return i + 15
    '\n    Given a method function, return a new method function that first\n    checks if ``self._frozen`` is true; and if so, raises ``ValueError``\n    with an appropriate message.  Otherwise, call the method and return\n    its result.\n    '

    def wrapped(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        else:
            return method(self, *args, **kwargs)
    wrapped.__name__ = method.__name__
    wrapped.__doc__ = (method.__doc__ or '') + _FROZEN_NOTICE % indent
    return wrapped

class FeatDict(FeatStruct, dict):
    """
    A feature structure that acts like a Python dictionary.  I.e., a
    mapping from feature identifiers to feature values, where a feature
    identifier can be a string or a ``Feature``; and where a feature value
    can be either a basic value (such as a string or an integer), or a nested
    feature structure.  A feature identifiers for a ``FeatDict`` is
    sometimes called a "feature name".

    Two feature dicts are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """

    def __init__(self, features=None, **morefeatures):
        if False:
            print('Hello World!')
        '\n        Create a new feature dictionary, with the specified features.\n\n        :param features: The initial value for this feature\n            dictionary.  If ``features`` is a ``FeatStruct``, then its\n            features are copied (shallow copy).  If ``features`` is a\n            dict, then a feature is created for each item, mapping its\n            key to its value.  If ``features`` is a string, then it is\n            processed using ``FeatStructReader``.  If ``features`` is a list of\n            tuples ``(name, val)``, then a feature is created for each tuple.\n        :param morefeatures: Additional features for the new feature\n            dictionary.  If a feature is listed under both ``features`` and\n            ``morefeatures``, then the value from ``morefeatures`` will be\n            used.\n        '
        if isinstance(features, str):
            FeatStructReader().fromstring(features, self)
            self.update(**morefeatures)
        else:
            self.update(features, **morefeatures)
    _INDEX_ERROR = 'Expected feature name or path.  Got %r.'

    def __getitem__(self, name_or_path):
        if False:
            while True:
                i = 10
        'If the feature with the given name or path exists, return\n        its value; otherwise, raise ``KeyError``.'
        if isinstance(name_or_path, (str, Feature)):
            return dict.__getitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            try:
                val = self
                for fid in name_or_path:
                    if not isinstance(val, FeatStruct):
                        raise KeyError
                    val = val[fid]
                return val
            except (KeyError, IndexError) as e:
                raise KeyError(name_or_path) from e
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def get(self, name_or_path, default=None):
        if False:
            return 10
        'If the feature with the given name or path exists, return its\n        value; otherwise, return ``default``.'
        try:
            return self[name_or_path]
        except KeyError:
            return default

    def __contains__(self, name_or_path):
        if False:
            for i in range(10):
                print('nop')
        'Return true if a feature with the given name or path exists.'
        try:
            self[name_or_path]
            return True
        except KeyError:
            return False

    def has_key(self, name_or_path):
        if False:
            for i in range(10):
                print('nop')
        'Return true if a feature with the given name or path exists.'
        return name_or_path in self

    def __delitem__(self, name_or_path):
        if False:
            for i in range(10):
                print('nop')
        'If the feature with the given name or path exists, delete\n        its value; otherwise, raise ``KeyError``.'
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (str, Feature)):
            return dict.__delitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError('The path () can not be set')
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)
                del parent[name_or_path[-1]]
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __setitem__(self, name_or_path, value):
        if False:
            while True:
                i = 10
        'Set the value for the feature with the given name or path\n        to ``value``.  If ``name_or_path`` is an invalid path, raise\n        ``KeyError``.'
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (str, Feature)):
            return dict.__setitem__(self, name_or_path, value)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError('The path () can not be set')
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)
                parent[name_or_path[-1]] = value
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)
    clear = _check_frozen(dict.clear)
    pop = _check_frozen(dict.pop)
    popitem = _check_frozen(dict.popitem)
    setdefault = _check_frozen(dict.setdefault)

    def update(self, features=None, **morefeatures):
        if False:
            for i in range(10):
                print('nop')
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if features is None:
            items = ()
        elif hasattr(features, 'items') and callable(features.items):
            items = features.items()
        elif hasattr(features, '__iter__'):
            items = features
        else:
            raise ValueError('Expected mapping or list of tuples')
        for (key, val) in items:
            if not isinstance(key, (str, Feature)):
                raise TypeError('Feature names must be strings')
            self[key] = val
        for (key, val) in morefeatures.items():
            if not isinstance(key, (str, Feature)):
                raise TypeError('Feature names must be strings')
            self[key] = val

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        memo[id(self)] = selfcopy = self.__class__()
        for (key, val) in self._items():
            selfcopy[copy.deepcopy(key, memo)] = copy.deepcopy(val, memo)
        return selfcopy

    def _keys(self):
        if False:
            while True:
                i = 10
        return self.keys()

    def _values(self):
        if False:
            for i in range(10):
                print('nop')
        return self.values()

    def _items(self):
        if False:
            for i in range(10):
                print('nop')
        return self.items()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        '\n        Display a multi-line representation of this feature dictionary\n        as an FVM (feature value matrix).\n        '
        return '\n'.join(self._str(self._find_reentrances({}), {}))

    def _repr(self, reentrances, reentrance_ids):
        if False:
            i = 10
            return i + 15
        segments = []
        prefix = ''
        suffix = ''
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
        for (fname, fval) in sorted(self.items()):
            display = getattr(fname, 'display', None)
            if id(fval) in reentrance_ids:
                segments.append(f'{fname}->({reentrance_ids[id(fval)]})')
            elif display == 'prefix' and (not prefix) and isinstance(fval, (Variable, str)):
                prefix = '%s' % fval
            elif display == 'slash' and (not suffix):
                if isinstance(fval, Variable):
                    suffix = '/%s' % fval.name
                else:
                    suffix = '/%s' % repr(fval)
            elif isinstance(fval, Variable):
                segments.append(f'{fname}={fval.name}')
            elif fval is True:
                segments.append('+%s' % fname)
            elif fval is False:
                segments.append('-%s' % fname)
            elif isinstance(fval, Expression):
                segments.append(f'{fname}=<{fval}>')
            elif not isinstance(fval, FeatStruct):
                segments.append(f'{fname}={repr(fval)}')
            else:
                fval_repr = fval._repr(reentrances, reentrance_ids)
                segments.append(f'{fname}={fval_repr}')
        if reentrances[id(self)]:
            prefix = f'({reentrance_ids[id(self)]}){prefix}'
        return '{}[{}]{}'.format(prefix, ', '.join(segments), suffix)

    def _str(self, reentrances, reentrance_ids):
        if False:
            return 10
        '\n        :return: A list of lines composing a string representation of\n            this feature dictionary.\n        :param reentrances: A dictionary that maps from the ``id`` of\n            each feature value in self, indicating whether that value\n            is reentrant or not.\n        :param reentrance_ids: A dictionary mapping from each ``id``\n            of a feature value to a unique identifier.  This is modified\n            by ``repr``: the first time a reentrant feature value is\n            displayed, an identifier is added to ``reentrance_ids`` for\n            it.\n        '
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
        if len(self) == 0:
            if reentrances[id(self)]:
                return ['(%s) []' % reentrance_ids[id(self)]]
            else:
                return ['[]']
        maxfnamelen = max((len('%s' % k) for k in self.keys()))
        lines = []
        for (fname, fval) in sorted(self.items()):
            fname = ('%s' % fname).ljust(maxfnamelen)
            if isinstance(fval, Variable):
                lines.append(f'{fname} = {fval.name}')
            elif isinstance(fval, Expression):
                lines.append(f'{fname} = <{fval}>')
            elif isinstance(fval, FeatList):
                fval_repr = fval._repr(reentrances, reentrance_ids)
                lines.append(f'{fname} = {repr(fval_repr)}')
            elif not isinstance(fval, FeatDict):
                lines.append(f'{fname} = {repr(fval)}')
            elif id(fval) in reentrance_ids:
                lines.append(f'{fname} -> ({reentrance_ids[id(fval)]})')
            else:
                if lines and lines[-1] != '':
                    lines.append('')
                fval_lines = fval._str(reentrances, reentrance_ids)
                fval_lines = [' ' * (maxfnamelen + 3) + l for l in fval_lines]
                nameline = (len(fval_lines) - 1) // 2
                fval_lines[nameline] = fname + ' =' + fval_lines[nameline][maxfnamelen + 2:]
                lines += fval_lines
                lines.append('')
        if lines[-1] == '':
            lines.pop()
        maxlen = max((len(line) for line in lines))
        lines = ['[ {}{} ]'.format(line, ' ' * (maxlen - len(line))) for line in lines]
        if reentrances[id(self)]:
            idstr = '(%s) ' % reentrance_ids[id(self)]
            lines = [' ' * len(idstr) + l for l in lines]
            idline = (len(lines) - 1) // 2
            lines[idline] = idstr + lines[idline][len(idstr):]
        return lines

class FeatList(FeatStruct, list):
    """
    A list of feature values, where each feature value is either a
    basic value (such as a string or an integer), or a nested feature
    structure.

    Feature lists may contain reentrant feature values.  A "reentrant
    feature value" is a single feature value that can be accessed via
    multiple feature paths.  Feature lists may also be cyclic.

    Two feature lists are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """

    def __init__(self, features=()):
        if False:
            i = 10
            return i + 15
        '\n        Create a new feature list, with the specified features.\n\n        :param features: The initial list of features for this feature\n            list.  If ``features`` is a string, then it is paresd using\n            ``FeatStructReader``.  Otherwise, it should be a sequence\n            of basic values and nested feature structures.\n        '
        if isinstance(features, str):
            FeatStructReader().fromstring(features, self)
        else:
            list.__init__(self, features)
    _INDEX_ERROR = 'Expected int or feature path.  Got %r.'

    def __getitem__(self, name_or_path):
        if False:
            i = 10
            return i + 15
        if isinstance(name_or_path, int):
            return list.__getitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            try:
                val = self
                for fid in name_or_path:
                    if not isinstance(val, FeatStruct):
                        raise KeyError
                    val = val[fid]
                return val
            except (KeyError, IndexError) as e:
                raise KeyError(name_or_path) from e
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __delitem__(self, name_or_path):
        if False:
            while True:
                i = 10
        'If the feature with the given name or path exists, delete\n        its value; otherwise, raise ``KeyError``.'
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (int, slice)):
            return list.__delitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError('The path () can not be set')
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)
                del parent[name_or_path[-1]]
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __setitem__(self, name_or_path, value):
        if False:
            for i in range(10):
                print('nop')
        'Set the value for the feature with the given name or path\n        to ``value``.  If ``name_or_path`` is an invalid path, raise\n        ``KeyError``.'
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (int, slice)):
            return list.__setitem__(self, name_or_path, value)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError('The path () can not be set')
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)
                parent[name_or_path[-1]] = value
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)
    __iadd__ = _check_frozen(list.__iadd__)
    __imul__ = _check_frozen(list.__imul__)
    append = _check_frozen(list.append)
    extend = _check_frozen(list.extend)
    insert = _check_frozen(list.insert)
    pop = _check_frozen(list.pop)
    remove = _check_frozen(list.remove)
    reverse = _check_frozen(list.reverse)
    sort = _check_frozen(list.sort)

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        memo[id(self)] = selfcopy = self.__class__()
        selfcopy.extend((copy.deepcopy(fval, memo) for fval in self))
        return selfcopy

    def _keys(self):
        if False:
            for i in range(10):
                print('nop')
        return list(range(len(self)))

    def _values(self):
        if False:
            print('Hello World!')
        return self

    def _items(self):
        if False:
            for i in range(10):
                print('nop')
        return enumerate(self)

    def _repr(self, reentrances, reentrance_ids):
        if False:
            while True:
                i = 10
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
            prefix = '(%s)' % reentrance_ids[id(self)]
        else:
            prefix = ''
        segments = []
        for fval in self:
            if id(fval) in reentrance_ids:
                segments.append('->(%s)' % reentrance_ids[id(fval)])
            elif isinstance(fval, Variable):
                segments.append(fval.name)
            elif isinstance(fval, Expression):
                segments.append('%s' % fval)
            elif isinstance(fval, FeatStruct):
                segments.append(fval._repr(reentrances, reentrance_ids))
            else:
                segments.append('%s' % repr(fval))
        return '{}[{}]'.format(prefix, ', '.join(segments))

def substitute_bindings(fstruct, bindings, fs_class='default'):
    if False:
        while True:
            i = 10
    "\n    Return the feature structure that is obtained by replacing each\n    variable bound by ``bindings`` with its binding.  If a variable is\n    aliased to a bound variable, then it will be replaced by that\n    variable's value.  If a variable is aliased to an unbound\n    variable, then it will be replaced by that variable.\n\n    :type bindings: dict(Variable -> any)\n    :param bindings: A dictionary mapping from variables to values.\n    "
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct)
    fstruct = copy.deepcopy(fstruct)
    _substitute_bindings(fstruct, bindings, fs_class, set())
    return fstruct

def _substitute_bindings(fstruct, bindings, fs_class, visited):
    if False:
        print('Hello World!')
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for (fname, fval) in items:
        while isinstance(fval, Variable) and fval in bindings:
            fval = fstruct[fname] = bindings[fval]
        if isinstance(fval, fs_class):
            _substitute_bindings(fval, bindings, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            fstruct[fname] = fval.substitute_bindings(bindings)

def retract_bindings(fstruct, bindings, fs_class='default'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the feature structure that is obtained by replacing each\n    feature structure value that is bound by ``bindings`` with the\n    variable that binds it.  A feature structure value must be\n    identical to a bound value (i.e., have equal id) to be replaced.\n\n    ``bindings`` is modified to point to this new feature structure,\n    rather than the original feature structure.  Feature structure\n    values in ``bindings`` may be modified if they are contained in\n    ``fstruct``.\n    '
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct)
    (fstruct, new_bindings) = copy.deepcopy((fstruct, bindings))
    bindings.update(new_bindings)
    inv_bindings = {id(val): var for (var, val) in bindings.items()}
    _retract_bindings(fstruct, inv_bindings, fs_class, set())
    return fstruct

def _retract_bindings(fstruct, inv_bindings, fs_class, visited):
    if False:
        while True:
            i = 10
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for (fname, fval) in items:
        if isinstance(fval, fs_class):
            if id(fval) in inv_bindings:
                fstruct[fname] = inv_bindings[id(fval)]
            _retract_bindings(fval, inv_bindings, fs_class, visited)

def find_variables(fstruct, fs_class='default'):
    if False:
        print('Hello World!')
    '\n    :return: The set of variables used by this feature structure.\n    :rtype: set(Variable)\n    '
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct)
    return _variables(fstruct, set(), fs_class, set())

def _variables(fstruct, vars, fs_class, visited):
    if False:
        print('Hello World!')
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for (fname, fval) in items:
        if isinstance(fval, Variable):
            vars.add(fval)
        elif isinstance(fval, fs_class):
            _variables(fval, vars, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            vars.update(fval.variables())
    return vars

def rename_variables(fstruct, vars=None, used_vars=(), new_vars=None, fs_class='default'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the feature structure that is obtained by replacing\n    any of this feature structure's variables that are in ``vars``\n    with new variables.  The names for these new variables will be\n    names that are not used by any variable in ``vars``, or in\n    ``used_vars``, or in this feature structure.\n\n    :type vars: set\n    :param vars: The set of variables that should be renamed.\n        If not specified, ``find_variables(fstruct)`` is used; i.e., all\n        variables will be given new names.\n    :type used_vars: set\n    :param used_vars: A set of variables whose names should not be\n        used by the new variables.\n    :type new_vars: dict(Variable -> Variable)\n    :param new_vars: A dictionary that is used to hold the mapping\n        from old variables to new variables.  For each variable *v*\n        in this feature structure:\n\n        - If ``new_vars`` maps *v* to *v'*, then *v* will be\n          replaced by *v'*.\n        - If ``new_vars`` does not contain *v*, but ``vars``\n          does contain *v*, then a new entry will be added to\n          ``new_vars``, mapping *v* to the new variable that is used\n          to replace it.\n\n    To consistently rename the variables in a set of feature\n    structures, simply apply rename_variables to each one, using\n    the same dictionary:\n\n        >>> from nltk.featstruct import FeatStruct\n        >>> fstruct1 = FeatStruct('[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]')\n        >>> fstruct2 = FeatStruct('[subj=[agr=[number=?z,gender=?y]], obj=[agr=[number=?z,gender=?y]]]')\n        >>> new_vars = {}  # Maps old vars to alpha-renamed vars\n        >>> fstruct1.rename_variables(new_vars=new_vars)\n        [obj=[agr=[gender=?y2]], subj=[agr=[gender=?y2]]]\n        >>> fstruct2.rename_variables(new_vars=new_vars)\n        [obj=[agr=[gender=?y2, number=?z2]], subj=[agr=[gender=?y2, number=?z2]]]\n\n    If new_vars is not specified, then an empty dictionary is used.\n    "
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct)
    if new_vars is None:
        new_vars = {}
    if vars is None:
        vars = find_variables(fstruct, fs_class)
    else:
        vars = set(vars)
    used_vars = find_variables(fstruct, fs_class).union(used_vars)
    return _rename_variables(copy.deepcopy(fstruct), vars, used_vars, new_vars, fs_class, set())

def _rename_variables(fstruct, vars, used_vars, new_vars, fs_class, visited):
    if False:
        return 10
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for (fname, fval) in items:
        if isinstance(fval, Variable):
            if fval in new_vars:
                fstruct[fname] = new_vars[fval]
            elif fval in vars:
                new_vars[fval] = _rename_variable(fval, used_vars)
                fstruct[fname] = new_vars[fval]
                used_vars.add(new_vars[fval])
        elif isinstance(fval, fs_class):
            _rename_variables(fval, vars, used_vars, new_vars, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            for var in fval.variables():
                if var in vars and var not in new_vars:
                    new_vars[var] = _rename_variable(var, used_vars)
                    used_vars.add(new_vars[var])
            fstruct[fname] = fval.substitute_bindings(new_vars)
    return fstruct

def _rename_variable(var, used_vars):
    if False:
        while True:
            i = 10
    (name, n) = (re.sub('\\d+$', '', var.name), 2)
    if not name:
        name = '?'
    while Variable(f'{name}{n}') in used_vars:
        n += 1
    return Variable(f'{name}{n}')

def remove_variables(fstruct, fs_class='default'):
    if False:
        for i in range(10):
            print('nop')
    '\n    :rtype: FeatStruct\n    :return: The feature structure that is obtained by deleting\n        all features whose values are ``Variables``.\n    '
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct)
    return _remove_variables(copy.deepcopy(fstruct), fs_class, set())

def _remove_variables(fstruct, fs_class, visited):
    if False:
        while True:
            i = 10
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = list(fstruct.items())
    elif _is_sequence(fstruct):
        items = list(enumerate(fstruct))
    else:
        raise ValueError('Expected mapping or sequence')
    for (fname, fval) in items:
        if isinstance(fval, Variable):
            del fstruct[fname]
        elif isinstance(fval, fs_class):
            _remove_variables(fval, fs_class, visited)
    return fstruct

class _UnificationFailure:

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'nltk.featstruct.UnificationFailure'
UnificationFailure = _UnificationFailure()
'A unique value used to indicate unification failure.  It can be\n   returned by ``Feature.unify_base_values()`` or by custom ``fail()``\n   functions to indicate that unificaiton should fail.'

def unify(fstruct1, fstruct2, bindings=None, trace=False, fail=None, rename_vars=True, fs_class='default'):
    if False:
        i = 10
        return i + 15
    "\n    Unify ``fstruct1`` with ``fstruct2``, and return the resulting feature\n    structure.  This unified feature structure is the minimal\n    feature structure that contains all feature value assignments from both\n    ``fstruct1`` and ``fstruct2``, and that preserves all reentrancies.\n\n    If no such feature structure exists (because ``fstruct1`` and\n    ``fstruct2`` specify incompatible values for some feature), then\n    unification fails, and ``unify`` returns None.\n\n    Bound variables are replaced by their values.  Aliased\n    variables are replaced by their representative variable\n    (if unbound) or the value of their representative variable\n    (if bound).  I.e., if variable *v* is in ``bindings``,\n    then *v* is replaced by ``bindings[v]``.  This will\n    be repeated until the variable is replaced by an unbound\n    variable or a non-variable value.\n\n    Unbound variables are bound when they are unified with\n    values; and aliased when they are unified with variables.\n    I.e., if variable *v* is not in ``bindings``, and is\n    unified with a variable or value *x*, then\n    ``bindings[v]`` is set to *x*.\n\n    If ``bindings`` is unspecified, then all variables are\n    assumed to be unbound.  I.e., ``bindings`` defaults to an\n    empty dict.\n\n        >>> from nltk.featstruct import FeatStruct\n        >>> FeatStruct('[a=?x]').unify(FeatStruct('[b=?x]'))\n        [a=?x, b=?x2]\n\n    :type bindings: dict(Variable -> any)\n    :param bindings: A set of variable bindings to be used and\n        updated during unification.\n    :type trace: bool\n    :param trace: If true, generate trace output.\n    :type rename_vars: bool\n    :param rename_vars: If True, then rename any variables in\n        ``fstruct2`` that are also used in ``fstruct1``, in order to\n        avoid collisions on variable names.\n    "
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct1)
        if _default_fs_class(fstruct2) != fs_class:
            raise ValueError('Mixing FeatStruct objects with Python dicts and lists is not supported.')
    assert isinstance(fstruct1, fs_class)
    assert isinstance(fstruct2, fs_class)
    user_bindings = bindings is not None
    if bindings is None:
        bindings = {}
    (fstruct1copy, fstruct2copy, bindings_copy) = copy.deepcopy((fstruct1, fstruct2, bindings))
    bindings.update(bindings_copy)
    if rename_vars:
        vars1 = find_variables(fstruct1copy, fs_class)
        vars2 = find_variables(fstruct2copy, fs_class)
        _rename_variables(fstruct2copy, vars1, vars2, {}, fs_class, set())
    forward = {}
    if trace:
        _trace_unify_start((), fstruct1copy, fstruct2copy)
    try:
        result = _destructively_unify(fstruct1copy, fstruct2copy, bindings, forward, trace, fail, fs_class, ())
    except _UnificationFailureError:
        return None
    if result is UnificationFailure:
        if fail is None:
            return None
        else:
            return fail(fstruct1copy, fstruct2copy, ())
    result = _apply_forwards(result, forward, fs_class, set())
    if user_bindings:
        _apply_forwards_to_bindings(forward, bindings)
    _resolve_aliases(bindings)
    _substitute_bindings(result, bindings, fs_class, set())
    if trace:
        _trace_unify_succeed((), result)
    if trace:
        _trace_bindings((), bindings)
    return result

class _UnificationFailureError(Exception):
    """An exception that is used by ``_destructively_unify`` to abort
    unification when a failure is encountered."""

def _destructively_unify(fstruct1, fstruct2, bindings, forward, trace, fail, fs_class, path):
    if False:
        for i in range(10):
            print('nop')
    "\n    Attempt to unify ``fstruct1`` and ``fstruct2`` by modifying them\n    in-place.  If the unification succeeds, then ``fstruct1`` will\n    contain the unified value, the value of ``fstruct2`` is undefined,\n    and forward[id(fstruct2)] is set to fstruct1.  If the unification\n    fails, then a _UnificationFailureError is raised, and the\n    values of ``fstruct1`` and ``fstruct2`` are undefined.\n\n    :param bindings: A dictionary mapping variables to values.\n    :param forward: A dictionary mapping feature structures ids\n        to replacement structures.  When two feature structures\n        are merged, a mapping from one to the other will be added\n        to the forward dictionary; and changes will be made only\n        to the target of the forward dictionary.\n        ``_destructively_unify`` will always 'follow' any links\n        in the forward dictionary for fstruct1 and fstruct2 before\n        actually unifying them.\n    :param trace: If true, generate trace output\n    :param path: The feature path that led us to this unification\n        step.  Used for trace output.\n    "
    if fstruct1 is fstruct2:
        if trace:
            _trace_unify_identity(path, fstruct1)
        return fstruct1
    forward[id(fstruct2)] = fstruct1
    if _is_mapping(fstruct1) and _is_mapping(fstruct2):
        for fname in fstruct1:
            if getattr(fname, 'default', None) is not None:
                fstruct2.setdefault(fname, fname.default)
        for fname in fstruct2:
            if getattr(fname, 'default', None) is not None:
                fstruct1.setdefault(fname, fname.default)
        for (fname, fval2) in sorted(fstruct2.items()):
            if fname in fstruct1:
                fstruct1[fname] = _unify_feature_values(fname, fstruct1[fname], fval2, bindings, forward, trace, fail, fs_class, path + (fname,))
            else:
                fstruct1[fname] = fval2
        return fstruct1
    elif _is_sequence(fstruct1) and _is_sequence(fstruct2):
        if len(fstruct1) != len(fstruct2):
            return UnificationFailure
        for findex in range(len(fstruct1)):
            fstruct1[findex] = _unify_feature_values(findex, fstruct1[findex], fstruct2[findex], bindings, forward, trace, fail, fs_class, path + (findex,))
        return fstruct1
    elif (_is_sequence(fstruct1) or _is_mapping(fstruct1)) and (_is_sequence(fstruct2) or _is_mapping(fstruct2)):
        return UnificationFailure
    raise TypeError('Expected mappings or sequences')

def _unify_feature_values(fname, fval1, fval2, bindings, forward, trace, fail, fs_class, fpath):
    if False:
        for i in range(10):
            print('nop')
    "\n    Attempt to unify ``fval1`` and and ``fval2``, and return the\n    resulting unified value.  The method of unification will depend on\n    the types of ``fval1`` and ``fval2``:\n\n      1. If they're both feature structures, then destructively\n         unify them (see ``_destructively_unify()``.\n      2. If they're both unbound variables, then alias one variable\n         to the other (by setting bindings[v2]=v1).\n      3. If one is an unbound variable, and the other is a value,\n         then bind the unbound variable to the value.\n      4. If one is a feature structure, and the other is a base value,\n         then fail.\n      5. If they're both base values, then unify them.  By default,\n         this will succeed if they are equal, and fail otherwise.\n    "
    if trace:
        _trace_unify_start(fpath, fval1, fval2)
    while id(fval1) in forward:
        fval1 = forward[id(fval1)]
    while id(fval2) in forward:
        fval2 = forward[id(fval2)]
    fvar1 = fvar2 = None
    while isinstance(fval1, Variable) and fval1 in bindings:
        fvar1 = fval1
        fval1 = bindings[fval1]
    while isinstance(fval2, Variable) and fval2 in bindings:
        fvar2 = fval2
        fval2 = bindings[fval2]
    if isinstance(fval1, fs_class) and isinstance(fval2, fs_class):
        result = _destructively_unify(fval1, fval2, bindings, forward, trace, fail, fs_class, fpath)
    elif isinstance(fval1, Variable) and isinstance(fval2, Variable):
        if fval1 != fval2:
            bindings[fval2] = fval1
        result = fval1
    elif isinstance(fval1, Variable):
        bindings[fval1] = fval2
        result = fval1
    elif isinstance(fval2, Variable):
        bindings[fval2] = fval1
        result = fval2
    elif isinstance(fval1, fs_class) or isinstance(fval2, fs_class):
        result = UnificationFailure
    else:
        if isinstance(fname, Feature):
            result = fname.unify_base_values(fval1, fval2, bindings)
        elif isinstance(fval1, CustomFeatureValue):
            result = fval1.unify(fval2)
            if isinstance(fval2, CustomFeatureValue) and result != fval2.unify(fval1):
                raise AssertionError('CustomFeatureValue objects %r and %r disagree about unification value: %r vs. %r' % (fval1, fval2, result, fval2.unify(fval1)))
        elif isinstance(fval2, CustomFeatureValue):
            result = fval2.unify(fval1)
        elif fval1 == fval2:
            result = fval1
        else:
            result = UnificationFailure
        if result is not UnificationFailure:
            if fvar1 is not None:
                bindings[fvar1] = result
                result = fvar1
            if fvar2 is not None and fvar2 != fvar1:
                bindings[fvar2] = result
                result = fvar2
    if result is UnificationFailure:
        if fail is not None:
            result = fail(fval1, fval2, fpath)
        if trace:
            _trace_unify_fail(fpath[:-1], result)
        if result is UnificationFailure:
            raise _UnificationFailureError
    if isinstance(result, fs_class):
        result = _apply_forwards(result, forward, fs_class, set())
    if trace:
        _trace_unify_succeed(fpath, result)
    if trace and isinstance(result, fs_class):
        _trace_bindings(fpath, bindings)
    return result

def _apply_forwards_to_bindings(forward, bindings):
    if False:
        print('Hello World!')
    '\n    Replace any feature structure that has a forward pointer with\n    the target of its forward pointer (to preserve reentrancy).\n    '
    for (var, value) in bindings.items():
        while id(value) in forward:
            value = forward[id(value)]
        bindings[var] = value

def _apply_forwards(fstruct, forward, fs_class, visited):
    if False:
        while True:
            i = 10
    '\n    Replace any feature structure that has a forward pointer with\n    the target of its forward pointer (to preserve reentrancy).\n    '
    while id(fstruct) in forward:
        fstruct = forward[id(fstruct)]
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError('Expected mapping or sequence')
    for (fname, fval) in items:
        if isinstance(fval, fs_class):
            while id(fval) in forward:
                fval = forward[id(fval)]
            fstruct[fname] = fval
            _apply_forwards(fval, forward, fs_class, visited)
    return fstruct

def _resolve_aliases(bindings):
    if False:
        while True:
            i = 10
    '\n    Replace any bound aliased vars with their binding; and replace\n    any unbound aliased vars with their representative var.\n    '
    for (var, value) in bindings.items():
        while isinstance(value, Variable) and value in bindings:
            value = bindings[var] = bindings[value]

def _trace_unify_start(path, fval1, fval2):
    if False:
        for i in range(10):
            print('nop')
    if path == ():
        print('\nUnification trace:')
    else:
        fullname = '.'.join(('%s' % n for n in path))
        print('  ' + '|   ' * (len(path) - 1) + '|')
        print('  ' + '|   ' * (len(path) - 1) + '| Unify feature: %s' % fullname)
    print('  ' + '|   ' * len(path) + ' / ' + _trace_valrepr(fval1))
    print('  ' + '|   ' * len(path) + '|\\ ' + _trace_valrepr(fval2))

def _trace_unify_identity(path, fval1):
    if False:
        return 10
    print('  ' + '|   ' * len(path) + '|')
    print('  ' + '|   ' * len(path) + '| (identical objects)')
    print('  ' + '|   ' * len(path) + '|')
    print('  ' + '|   ' * len(path) + '+-->' + repr(fval1))

def _trace_unify_fail(path, result):
    if False:
        return 10
    if result is UnificationFailure:
        resume = ''
    else:
        resume = ' (nonfatal)'
    print('  ' + '|   ' * len(path) + '|   |')
    print('  ' + 'X   ' * len(path) + 'X   X <-- FAIL' + resume)

def _trace_unify_succeed(path, fval1):
    if False:
        for i in range(10):
            print('nop')
    print('  ' + '|   ' * len(path) + '|')
    print('  ' + '|   ' * len(path) + '+-->' + repr(fval1))

def _trace_bindings(path, bindings):
    if False:
        i = 10
        return i + 15
    if len(bindings) > 0:
        binditems = sorted(bindings.items(), key=lambda v: v[0].name)
        bindstr = '{%s}' % ', '.join((f'{var}: {_trace_valrepr(val)}' for (var, val) in binditems))
        print('  ' + '|   ' * len(path) + '    Bindings: ' + bindstr)

def _trace_valrepr(val):
    if False:
        while True:
            i = 10
    if isinstance(val, Variable):
        return '%s' % val
    else:
        return '%s' % repr(val)

def subsumes(fstruct1, fstruct2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return True if ``fstruct1`` subsumes ``fstruct2``.  I.e., return\n    true if unifying ``fstruct1`` with ``fstruct2`` would result in a\n    feature structure equal to ``fstruct2.``\n\n    :rtype: bool\n    '
    return fstruct2 == unify(fstruct1, fstruct2)

def conflicts(fstruct1, fstruct2, trace=0):
    if False:
        while True:
            i = 10
    '\n    Return a list of the feature paths of all features which are\n    assigned incompatible values by ``fstruct1`` and ``fstruct2``.\n\n    :rtype: list(tuple)\n    '
    conflict_list = []

    def add_conflict(fval1, fval2, path):
        if False:
            print('Hello World!')
        conflict_list.append(path)
        return fval1
    unify(fstruct1, fstruct2, fail=add_conflict, trace=trace)
    return conflict_list

def _is_mapping(v):
    if False:
        print('Hello World!')
    return hasattr(v, '__contains__') and hasattr(v, 'keys')

def _is_sequence(v):
    if False:
        print('Hello World!')
    return hasattr(v, '__iter__') and hasattr(v, '__len__') and (not isinstance(v, str))

def _default_fs_class(obj):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, FeatStruct):
        return FeatStruct
    if isinstance(obj, (dict, list)):
        return (dict, list)
    else:
        raise ValueError('To unify objects of type %s, you must specify fs_class explicitly.' % obj.__class__.__name__)

class SubstituteBindingsSequence(SubstituteBindingsI):
    """
    A mixin class for sequence classes that distributes variables() and
    substitute_bindings() over the object's elements.
    """

    def variables(self):
        if False:
            i = 10
            return i + 15
        return [elt for elt in self if isinstance(elt, Variable)] + sum((list(elt.variables()) for elt in self if isinstance(elt, SubstituteBindingsI)), [])

    def substitute_bindings(self, bindings):
        if False:
            return 10
        return self.__class__([self.subst(v, bindings) for v in self])

    def subst(self, v, bindings):
        if False:
            print('Hello World!')
        if isinstance(v, SubstituteBindingsI):
            return v.substitute_bindings(bindings)
        else:
            return bindings.get(v, v)

class FeatureValueTuple(SubstituteBindingsSequence, tuple):
    """
    A base feature value that is a tuple of other base feature values.
    FeatureValueTuple implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueTuple`` is immutable.
    """

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self) == 0:
            return '()'
        return '(%s)' % ', '.join((f'{b}' for b in self))

class FeatureValueSet(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that is a set of other base feature values.
    FeatureValueSet implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueSet`` is immutable.
    """

    def __repr__(self):
        if False:
            print('Hello World!')
        if len(self) == 0:
            return '{/}'
        return '{%s}' % ', '.join(sorted((f'{b}' for b in self)))
    __str__ = __repr__

class FeatureValueUnion(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that represents the union of two or more
    ``FeatureValueSet`` or ``Variable``.
    """

    def __new__(cls, values):
        if False:
            print('Hello World!')
        values = _flatten(values, FeatureValueUnion)
        if sum((isinstance(v, Variable) for v in values)) == 0:
            values = _flatten(values, FeatureValueSet)
            return FeatureValueSet(values)
        if len(values) == 1:
            return list(values)[0]
        return frozenset.__new__(cls, values)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{%s}' % '+'.join(sorted((f'{b}' for b in self)))

class FeatureValueConcat(SubstituteBindingsSequence, tuple):
    """
    A base feature value that represents the concatenation of two or
    more ``FeatureValueTuple`` or ``Variable``.
    """

    def __new__(cls, values):
        if False:
            while True:
                i = 10
        values = _flatten(values, FeatureValueConcat)
        if sum((isinstance(v, Variable) for v in values)) == 0:
            values = _flatten(values, FeatureValueTuple)
            return FeatureValueTuple(values)
        if len(values) == 1:
            return list(values)[0]
        return tuple.__new__(cls, values)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '(%s)' % '+'.join((f'{b}' for b in self))

def _flatten(lst, cls):
    if False:
        return 10
    '\n    Helper function -- return a copy of list, with all elements of\n    type ``cls`` spliced in rather than appended in.\n    '
    result = []
    for elt in lst:
        if isinstance(elt, cls):
            result.extend(elt)
        else:
            result.append(elt)
    return result

@total_ordering
class Feature:
    """
    A feature identifier that's specialized to put additional
    constraints, default values, etc.
    """

    def __init__(self, name, default=None, display=None):
        if False:
            while True:
                i = 10
        assert display in (None, 'prefix', 'slash')
        self._name = name
        self._default = default
        self._display = display
        if self._display == 'prefix':
            self._sortkey = (-1, self._name)
        elif self._display == 'slash':
            self._sortkey = (1, self._name)
        else:
            self._sortkey = (0, self._name)

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'The name of this feature.'
        return self._name

    @property
    def default(self):
        if False:
            for i in range(10):
                print('nop')
        'Default value for this feature.'
        return self._default

    @property
    def display(self):
        if False:
            for i in range(10):
                print('nop')
        'Custom display location: can be prefix, or slash.'
        return self._display

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '*%s*' % self.name

    def __lt__(self, other):
        if False:
            return 10
        if isinstance(other, str):
            return True
        if not isinstance(other, Feature):
            raise_unorderable_types('<', self, other)
        return self._sortkey < other._sortkey

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return type(self) == type(other) and self._name == other._name

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self._name)

    def read_value(self, s, position, reentrances, parser):
        if False:
            i = 10
            return i + 15
        return parser.read_value(s, position, reentrances)

    def unify_base_values(self, fval1, fval2, bindings):
        if False:
            while True:
                i = 10
        '\n        If possible, return a single value..  If not, return\n        the value ``UnificationFailure``.\n        '
        if fval1 == fval2:
            return fval1
        else:
            return UnificationFailure

class SlashFeature(Feature):

    def read_value(self, s, position, reentrances, parser):
        if False:
            while True:
                i = 10
        return parser.read_partial(s, position, reentrances)

class RangeFeature(Feature):
    RANGE_RE = re.compile('(-?\\d+):(-?\\d+)')

    def read_value(self, s, position, reentrances, parser):
        if False:
            for i in range(10):
                print('nop')
        m = self.RANGE_RE.match(s, position)
        if not m:
            raise ValueError('range', position)
        return ((int(m.group(1)), int(m.group(2))), m.end())

    def unify_base_values(self, fval1, fval2, bindings):
        if False:
            for i in range(10):
                print('nop')
        if fval1 is None:
            return fval2
        if fval2 is None:
            return fval1
        rng = (max(fval1[0], fval2[0]), min(fval1[1], fval2[1]))
        if rng[1] < rng[0]:
            return UnificationFailure
        return rng
SLASH = SlashFeature('slash', default=False, display='slash')
TYPE = Feature('type', display='prefix')

@total_ordering
class CustomFeatureValue:
    """
    An abstract base class for base values that define a custom
    unification method.  The custom unification method of
    ``CustomFeatureValue`` will be used during unification if:

      - The ``CustomFeatureValue`` is unified with another base value.
      - The ``CustomFeatureValue`` is not the value of a customized
        ``Feature`` (which defines its own unification method).

    If two ``CustomFeatureValue`` objects are unified with one another
    during feature structure unification, then the unified base values
    they return *must* be equal; otherwise, an ``AssertionError`` will
    be raised.

    Subclasses must define ``unify()``, ``__eq__()`` and ``__lt__()``.
    Subclasses may also wish to define ``__hash__()``.
    """

    def unify(self, other):
        if False:
            while True:
                i = 10
        '\n        If this base value unifies with ``other``, then return the\n        unified value.  Otherwise, return ``UnificationFailure``.\n        '
        raise NotImplementedError('abstract base class')

    def __eq__(self, other):
        if False:
            return 10
        return NotImplemented

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __lt__(self, other):
        if False:
            return 10
        return NotImplemented

    def __hash__(self):
        if False:
            print('Hello World!')
        raise TypeError('%s objects or unhashable' % self.__class__.__name__)

class FeatStructReader:

    def __init__(self, features=(SLASH, TYPE), fdict_class=FeatStruct, flist_class=FeatList, logic_parser=None):
        if False:
            print('Hello World!')
        self._features = {f.name: f for f in features}
        self._fdict_class = fdict_class
        self._flist_class = flist_class
        self._prefix_feature = None
        self._slash_feature = None
        for feature in features:
            if feature.display == 'slash':
                if self._slash_feature:
                    raise ValueError('Multiple features w/ display=slash')
                self._slash_feature = feature
            if feature.display == 'prefix':
                if self._prefix_feature:
                    raise ValueError('Multiple features w/ display=prefix')
                self._prefix_feature = feature
        self._features_with_defaults = [feature for feature in features if feature.default is not None]
        if logic_parser is None:
            logic_parser = LogicParser()
        self._logic_parser = logic_parser

    def fromstring(self, s, fstruct=None):
        if False:
            return 10
        "\n        Convert a string representation of a feature structure (as\n        displayed by repr) into a ``FeatStruct``.  This process\n        imposes the following restrictions on the string\n        representation:\n\n        - Feature names cannot contain any of the following:\n          whitespace, parentheses, quote marks, equals signs,\n          dashes, commas, and square brackets.  Feature names may\n          not begin with plus signs or minus signs.\n        - Only the following basic feature value are supported:\n          strings, integers, variables, None, and unquoted\n          alphanumeric strings.\n        - For reentrant values, the first mention must specify\n          a reentrance identifier and a value; and any subsequent\n          mentions must use arrows (``'->'``) to reference the\n          reentrance identifier.\n        "
        s = s.strip()
        (value, position) = self.read_partial(s, 0, {}, fstruct)
        if position != len(s):
            self._error(s, 'end of string', position)
        return value
    _START_FSTRUCT_RE = re.compile('\\s*(?:\\((\\d+)\\)\\s*)?(\\??[\\w-]+)?(\\[)')
    _END_FSTRUCT_RE = re.compile('\\s*]\\s*')
    _SLASH_RE = re.compile('/')
    _FEATURE_NAME_RE = re.compile('\\s*([+-]?)([^\\s\\(\\)<>"\\\'\\-=\\[\\],]+)\\s*')
    _REENTRANCE_RE = re.compile('\\s*->\\s*')
    _TARGET_RE = re.compile('\\s*\\((\\d+)\\)\\s*')
    _ASSIGN_RE = re.compile('\\s*=\\s*')
    _COMMA_RE = re.compile('\\s*,\\s*')
    _BARE_PREFIX_RE = re.compile('\\s*(?:\\((\\d+)\\)\\s*)?(\\??[\\w-]+\\s*)()')
    _START_FDICT_RE = re.compile('(%s)|(%s\\s*(%s\\s*(=|->)|[+-]%s|\\]))' % (_BARE_PREFIX_RE.pattern, _START_FSTRUCT_RE.pattern, _FEATURE_NAME_RE.pattern, _FEATURE_NAME_RE.pattern))

    def read_partial(self, s, position=0, reentrances=None, fstruct=None):
        if False:
            print('Hello World!')
        '\n        Helper function that reads in a feature structure.\n\n        :param s: The string to read.\n        :param position: The position in the string to start parsing.\n        :param reentrances: A dictionary from reentrance ids to values.\n            Defaults to an empty dictionary.\n        :return: A tuple (val, pos) of the feature structure created by\n            parsing and the position where the parsed feature structure ends.\n        :rtype: bool\n        '
        if reentrances is None:
            reentrances = {}
        try:
            return self._read_partial(s, position, reentrances, fstruct)
        except ValueError as e:
            if len(e.args) != 2:
                raise
            self._error(s, *e.args)

    def _read_partial(self, s, position, reentrances, fstruct=None):
        if False:
            return 10
        if fstruct is None:
            if self._START_FDICT_RE.match(s, position):
                fstruct = self._fdict_class()
            else:
                fstruct = self._flist_class()
        match = self._START_FSTRUCT_RE.match(s, position)
        if not match:
            match = self._BARE_PREFIX_RE.match(s, position)
            if not match:
                raise ValueError('open bracket or identifier', position)
        position = match.end()
        if match.group(1):
            identifier = match.group(1)
            if identifier in reentrances:
                raise ValueError('new identifier', match.start(1))
            reentrances[identifier] = fstruct
        if isinstance(fstruct, FeatDict):
            fstruct.clear()
            return self._read_partial_featdict(s, position, match, reentrances, fstruct)
        else:
            del fstruct[:]
            return self._read_partial_featlist(s, position, match, reentrances, fstruct)

    def _read_partial_featlist(self, s, position, match, reentrances, fstruct):
        if False:
            i = 10
            return i + 15
        if match.group(2):
            raise ValueError('open bracket')
        if not match.group(3):
            raise ValueError('open bracket')
        while position < len(s):
            match = self._END_FSTRUCT_RE.match(s, position)
            if match is not None:
                return (fstruct, match.end())
            match = self._REENTRANCE_RE.match(s, position)
            if match:
                position = match.end()
                match = self._TARGET_RE.match(s, position)
                if not match:
                    raise ValueError('identifier', position)
                target = match.group(1)
                if target not in reentrances:
                    raise ValueError('bound identifier', position)
                position = match.end()
                fstruct.append(reentrances[target])
            else:
                (value, position) = self._read_value(0, s, position, reentrances)
                fstruct.append(value)
            if self._END_FSTRUCT_RE.match(s, position):
                continue
            match = self._COMMA_RE.match(s, position)
            if match is None:
                raise ValueError('comma', position)
            position = match.end()
        raise ValueError('close bracket', position)

    def _read_partial_featdict(self, s, position, match, reentrances, fstruct):
        if False:
            return 10
        if match.group(2):
            if self._prefix_feature is None:
                raise ValueError('open bracket or identifier', match.start(2))
            prefixval = match.group(2).strip()
            if prefixval.startswith('?'):
                prefixval = Variable(prefixval)
            fstruct[self._prefix_feature] = prefixval
        if not match.group(3):
            return self._finalize(s, match.end(), reentrances, fstruct)
        while position < len(s):
            name = value = None
            match = self._END_FSTRUCT_RE.match(s, position)
            if match is not None:
                return self._finalize(s, match.end(), reentrances, fstruct)
            match = self._FEATURE_NAME_RE.match(s, position)
            if match is None:
                raise ValueError('feature name', position)
            name = match.group(2)
            position = match.end()
            if name[0] == '*' and name[-1] == '*':
                name = self._features.get(name[1:-1])
                if name is None:
                    raise ValueError('known special feature', match.start(2))
            if name in fstruct:
                raise ValueError('new name', match.start(2))
            if match.group(1) == '+':
                value = True
            if match.group(1) == '-':
                value = False
            if value is None:
                match = self._REENTRANCE_RE.match(s, position)
                if match is not None:
                    position = match.end()
                    match = self._TARGET_RE.match(s, position)
                    if not match:
                        raise ValueError('identifier', position)
                    target = match.group(1)
                    if target not in reentrances:
                        raise ValueError('bound identifier', position)
                    position = match.end()
                    value = reentrances[target]
            if value is None:
                match = self._ASSIGN_RE.match(s, position)
                if match:
                    position = match.end()
                    (value, position) = self._read_value(name, s, position, reentrances)
                else:
                    raise ValueError('equals sign', position)
            fstruct[name] = value
            if self._END_FSTRUCT_RE.match(s, position):
                continue
            match = self._COMMA_RE.match(s, position)
            if match is None:
                raise ValueError('comma', position)
            position = match.end()
        raise ValueError('close bracket', position)

    def _finalize(self, s, pos, reentrances, fstruct):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when we see the close brace -- checks for a slash feature,\n        and adds in default values.\n        '
        match = self._SLASH_RE.match(s, pos)
        if match:
            name = self._slash_feature
            (v, pos) = self._read_value(name, s, match.end(), reentrances)
            fstruct[name] = v
        return (fstruct, pos)

    def _read_value(self, name, s, position, reentrances):
        if False:
            return 10
        if isinstance(name, Feature):
            return name.read_value(s, position, reentrances, self)
        else:
            return self.read_value(s, position, reentrances)

    def read_value(self, s, position, reentrances):
        if False:
            return 10
        for (handler, regexp) in self.VALUE_HANDLERS:
            match = regexp.match(s, position)
            if match:
                handler_func = getattr(self, handler)
                return handler_func(s, position, reentrances, match)
        raise ValueError('value', position)

    def _error(self, s, expected, position):
        if False:
            i = 10
            return i + 15
        lines = s.split('\n')
        while position > len(lines[0]):
            position -= len(lines.pop(0)) + 1
        estr = 'Error parsing feature structure\n    ' + lines[0] + '\n    ' + ' ' * position + '^ ' + 'Expected %s' % expected
        raise ValueError(estr)
    VALUE_HANDLERS = [('read_fstruct_value', _START_FSTRUCT_RE), ('read_var_value', re.compile('\\?[a-zA-Z_][a-zA-Z0-9_]*')), ('read_str_value', re.compile('[uU]?[rR]?([\'"])')), ('read_int_value', re.compile('-?\\d+')), ('read_sym_value', re.compile('[a-zA-Z_][a-zA-Z0-9_]*')), ('read_app_value', re.compile('<(app)\\((\\?[a-z][a-z]*)\\s*,\\s*(\\?[a-z][a-z]*)\\)>')), ('read_logic_value', re.compile('<(.*?)(?<!-)>')), ('read_set_value', re.compile('{')), ('read_tuple_value', re.compile('\\('))]

    def read_fstruct_value(self, s, position, reentrances, match):
        if False:
            print('Hello World!')
        return self.read_partial(s, position, reentrances)

    def read_str_value(self, s, position, reentrances, match):
        if False:
            return 10
        return read_str(s, position)

    def read_int_value(self, s, position, reentrances, match):
        if False:
            return 10
        return (int(match.group()), match.end())

    def read_var_value(self, s, position, reentrances, match):
        if False:
            print('Hello World!')
        return (Variable(match.group()), match.end())
    _SYM_CONSTS = {'None': None, 'True': True, 'False': False}

    def read_sym_value(self, s, position, reentrances, match):
        if False:
            while True:
                i = 10
        (val, end) = (match.group(), match.end())
        return (self._SYM_CONSTS.get(val, val), end)

    def read_app_value(self, s, position, reentrances, match):
        if False:
            while True:
                i = 10
        'Mainly included for backwards compat.'
        return (self._logic_parser.parse('%s(%s)' % match.group(2, 3)), match.end())

    def read_logic_value(self, s, position, reentrances, match):
        if False:
            for i in range(10):
                print('nop')
        try:
            try:
                expr = self._logic_parser.parse(match.group(1))
            except LogicalExpressionException as e:
                raise ValueError from e
            return (expr, match.end())
        except ValueError as e:
            raise ValueError('logic expression', match.start(1)) from e

    def read_tuple_value(self, s, position, reentrances, match):
        if False:
            print('Hello World!')
        return self._read_seq_value(s, position, reentrances, match, ')', FeatureValueTuple, FeatureValueConcat)

    def read_set_value(self, s, position, reentrances, match):
        if False:
            print('Hello World!')
        return self._read_seq_value(s, position, reentrances, match, '}', FeatureValueSet, FeatureValueUnion)

    def _read_seq_value(self, s, position, reentrances, match, close_paren, seq_class, plus_class):
        if False:
            while True:
                i = 10
        '\n        Helper function used by read_tuple_value and read_set_value.\n        '
        cp = re.escape(close_paren)
        position = match.end()
        m = re.compile('\\s*/?\\s*%s' % cp).match(s, position)
        if m:
            return (seq_class(), m.end())
        values = []
        seen_plus = False
        while True:
            m = re.compile('\\s*%s' % cp).match(s, position)
            if m:
                if seen_plus:
                    return (plus_class(values), m.end())
                else:
                    return (seq_class(values), m.end())
            (val, position) = self.read_value(s, position, reentrances)
            values.append(val)
            m = re.compile('\\s*(,|\\+|(?=%s))\\s*' % cp).match(s, position)
            if not m:
                raise ValueError("',' or '+' or '%s'" % cp, position)
            if m.group(1) == '+':
                seen_plus = True
            position = m.end()

def display_unification(fs1, fs2, indent='  '):
    if False:
        return 10
    fs1_lines = ('%s' % fs1).split('\n')
    fs2_lines = ('%s' % fs2).split('\n')
    if len(fs1_lines) > len(fs2_lines):
        blankline = '[' + ' ' * (len(fs2_lines[0]) - 2) + ']'
        fs2_lines += [blankline] * len(fs1_lines)
    else:
        blankline = '[' + ' ' * (len(fs1_lines[0]) - 2) + ']'
        fs1_lines += [blankline] * len(fs2_lines)
    for (fs1_line, fs2_line) in zip(fs1_lines, fs2_lines):
        print(indent + fs1_line + '   ' + fs2_line)
    print(indent + '-' * len(fs1_lines[0]) + '   ' + '-' * len(fs2_lines[0]))
    linelen = len(fs1_lines[0]) * 2 + 3
    print(indent + '|               |'.center(linelen))
    print(indent + '+-----UNIFY-----+'.center(linelen))
    print(indent + '|'.center(linelen))
    print(indent + 'V'.center(linelen))
    bindings = {}
    result = fs1.unify(fs2, bindings)
    if result is None:
        print(indent + '(FAILED)'.center(linelen))
    else:
        print('\n'.join((indent + l.center(linelen) for l in ('%s' % result).split('\n'))))
        if bindings and len(bindings.bound_variables()) > 0:
            print(repr(bindings).center(linelen))
    return result

def interactive_demo(trace=False):
    if False:
        print('Hello World!')
    import random
    import sys
    HELP = '\n    1-%d: Select the corresponding feature structure\n    q: Quit\n    t: Turn tracing on or off\n    l: List all feature structures\n    ?: Help\n    '
    print('\n    This demo will repeatedly present you with a list of feature\n    structures, and ask you to choose two for unification.  Whenever a\n    new feature structure is generated, it is added to the list of\n    choices that you can pick from.  However, since this can be a\n    large number of feature structures, the demo will only print out a\n    random subset for you to choose between at a given time.  If you\n    want to see the complete lists, type "l".  For a list of valid\n    commands, type "?".\n    ')
    print('Press "Enter" to continue...')
    sys.stdin.readline()
    fstruct_strings = ['[agr=[number=sing, gender=masc]]', '[agr=[gender=masc, person=3]]', '[agr=[gender=fem, person=3]]', '[subj=[agr=(1)[]], agr->(1)]', '[obj=?x]', '[subj=?x]', '[/=None]', '[/=NP]', '[cat=NP]', '[cat=VP]', '[cat=PP]', '[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]', '[gender=masc, agr=?C]', '[gender=?S, agr=[gender=?S,person=3]]']
    all_fstructs = [(i, FeatStruct(fstruct_strings[i])) for i in range(len(fstruct_strings))]

    def list_fstructs(fstructs):
        if False:
            while True:
                i = 10
        for (i, fstruct) in fstructs:
            print()
            lines = ('%s' % fstruct).split('\n')
            print('%3d: %s' % (i + 1, lines[0]))
            for line in lines[1:]:
                print('     ' + line)
        print()
    while True:
        MAX_CHOICES = 5
        if len(all_fstructs) > MAX_CHOICES:
            fstructs = sorted(random.sample(all_fstructs, MAX_CHOICES))
        else:
            fstructs = all_fstructs
        print('_' * 75)
        print('Choose two feature structures to unify:')
        list_fstructs(fstructs)
        selected = [None, None]
        for (nth, i) in (('First', 0), ('Second', 1)):
            while selected[i] is None:
                print('%s feature structure (1-%d,q,t,l,?): ' % (nth, len(all_fstructs)), end=' ')
                try:
                    input = sys.stdin.readline().strip()
                    if input in ('q', 'Q', 'x', 'X'):
                        return
                    if input in ('t', 'T'):
                        trace = not trace
                        print('   Trace = %s' % trace)
                        continue
                    if input in ('h', 'H', '?'):
                        print(HELP % len(fstructs))
                        continue
                    if input in ('l', 'L'):
                        list_fstructs(all_fstructs)
                        continue
                    num = int(input) - 1
                    selected[i] = all_fstructs[num][1]
                    print()
                except:
                    print('Bad sentence number')
                    continue
        if trace:
            result = selected[0].unify(selected[1], trace=1)
        else:
            result = display_unification(selected[0], selected[1])
        if result is not None:
            for (i, fstruct) in all_fstructs:
                if repr(result) == repr(fstruct):
                    break
            else:
                all_fstructs.append((len(all_fstructs), result))
        print('\nType "Enter" to continue unifying; or "q" to quit.')
        input = sys.stdin.readline().strip()
        if input in ('q', 'Q', 'x', 'X'):
            return

def demo(trace=False):
    if False:
        print('Hello World!')
    '\n    Just for testing\n    '
    fstruct_strings = ['[agr=[number=sing, gender=masc]]', '[agr=[gender=masc, person=3]]', '[agr=[gender=fem, person=3]]', '[subj=[agr=(1)[]], agr->(1)]', '[obj=?x]', '[subj=?x]', '[/=None]', '[/=NP]', '[cat=NP]', '[cat=VP]', '[cat=PP]', '[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]', '[gender=masc, agr=?C]', '[gender=?S, agr=[gender=?S,person=3]]']
    all_fstructs = [FeatStruct(fss) for fss in fstruct_strings]
    for fs1 in all_fstructs:
        for fs2 in all_fstructs:
            print('\n*******************\nfs1 is:\n%s\n\nfs2 is:\n%s\n\nresult is:\n%s' % (fs1, fs2, unify(fs1, fs2)))
if __name__ == '__main__':
    demo()
__all__ = ['FeatStruct', 'FeatDict', 'FeatList', 'unify', 'subsumes', 'conflicts', 'Feature', 'SlashFeature', 'RangeFeature', 'SLASH', 'TYPE', 'FeatStructReader']