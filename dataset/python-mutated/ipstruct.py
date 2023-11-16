"""A dict subclass that supports attribute style access.

Authors:

* Fernando Perez (original)
* Brian Granger (refactoring to a dict subclass)
"""
__all__ = ['Struct']

class Struct(dict):
    """A dict subclass with attribute style access.

    This dict subclass has a a few extra features:

    * Attribute style access.
    * Protection of class members (like keys, items) when using attribute
      style access.
    * The ability to restrict assignment to only existing keys.
    * Intelligent merging.
    * Overloaded operators.
    """
    _allownew = True

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        "Initialize with a dictionary, another Struct, or data.\n\n        Parameters\n        ----------\n        *args : dict, Struct\n            Initialize with one dict or Struct\n        **kw : dict\n            Initialize with key, value pairs.\n\n        Examples\n        --------\n        >>> s = Struct(a=10,b=30)\n        >>> s.a\n        10\n        >>> s.b\n        30\n        >>> s2 = Struct(s,c=30)\n        >>> sorted(s2.keys())\n        ['a', 'b', 'c']\n        "
        object.__setattr__(self, '_allownew', True)
        dict.__init__(self, *args, **kw)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        "Set an item with check for allownew.\n\n        Examples\n        --------\n        >>> s = Struct()\n        >>> s['a'] = 10\n        >>> s.allow_new_attr(False)\n        >>> s['a'] = 10\n        >>> s['a']\n        10\n        >>> try:\n        ...     s['b'] = 20\n        ... except KeyError:\n        ...     print('this is not allowed')\n        ...\n        this is not allowed\n        "
        if not self._allownew and key not in self:
            raise KeyError("can't create new attribute %s when allow_new_attr(False)" % key)
        dict.__setitem__(self, key, value)

    def __setattr__(self, key, value):
        if False:
            i = 10
            return i + 15
        'Set an attr with protection of class members.\n\n        This calls :meth:`self.__setitem__` but convert :exc:`KeyError` to\n        :exc:`AttributeError`.\n\n        Examples\n        --------\n        >>> s = Struct()\n        >>> s.a = 10\n        >>> s.a\n        10\n        >>> try:\n        ...     s.get = 10\n        ... except AttributeError:\n        ...     print("you can\'t set a class member")\n        ...\n        you can\'t set a class member\n        '
        if isinstance(key, str):
            if key in self.__dict__ or hasattr(Struct, key):
                raise AttributeError('attr %s is a protected member of class Struct.' % key)
        try:
            self.__setitem__(key, value)
        except KeyError as e:
            raise AttributeError(e) from e

    def __getattr__(self, key):
        if False:
            return 10
        'Get an attr by calling :meth:`dict.__getitem__`.\n\n        Like :meth:`__setattr__`, this method converts :exc:`KeyError` to\n        :exc:`AttributeError`.\n\n        Examples\n        --------\n        >>> s = Struct(a=10)\n        >>> s.a\n        10\n        >>> type(s.get)\n        <...method\'>\n        >>> try:\n        ...     s.b\n        ... except AttributeError:\n        ...     print("I don\'t have that key")\n        ...\n        I don\'t have that key\n        '
        try:
            result = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        else:
            return result

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        "s += s2 is a shorthand for s.merge(s2).\n\n        Examples\n        --------\n        >>> s = Struct(a=10,b=30)\n        >>> s2 = Struct(a=20,c=40)\n        >>> s += s2\n        >>> sorted(s.keys())\n        ['a', 'b', 'c']\n        "
        self.merge(other)
        return self

    def __add__(self, other):
        if False:
            print('Hello World!')
        "s + s2 -> New Struct made from s.merge(s2).\n\n        Examples\n        --------\n        >>> s1 = Struct(a=10,b=30)\n        >>> s2 = Struct(a=20,c=40)\n        >>> s = s1 + s2\n        >>> sorted(s.keys())\n        ['a', 'b', 'c']\n        "
        sout = self.copy()
        sout.merge(other)
        return sout

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        "s1 - s2 -> remove keys in s2 from s1.\n\n        Examples\n        --------\n        >>> s1 = Struct(a=10,b=30)\n        >>> s2 = Struct(a=40)\n        >>> s = s1 - s2\n        >>> s\n        {'b': 30}\n        "
        sout = self.copy()
        sout -= other
        return sout

    def __isub__(self, other):
        if False:
            while True:
                i = 10
        "Inplace remove keys from self that are in other.\n\n        Examples\n        --------\n        >>> s1 = Struct(a=10,b=30)\n        >>> s2 = Struct(a=40)\n        >>> s1 -= s2\n        >>> s1\n        {'b': 30}\n        "
        for k in other.keys():
            if k in self:
                del self[k]
        return self

    def __dict_invert(self, data):
        if False:
            print('Hello World!')
        'Helper function for merge.\n\n        Takes a dictionary whose values are lists and returns a dict with\n        the elements of each list as keys and the original keys as values.\n        '
        outdict = {}
        for (k, lst) in data.items():
            if isinstance(lst, str):
                lst = lst.split()
            for entry in lst:
                outdict[entry] = k
        return outdict

    def dict(self):
        if False:
            print('Hello World!')
        return self

    def copy(self):
        if False:
            while True:
                i = 10
        'Return a copy as a Struct.\n\n        Examples\n        --------\n        >>> s = Struct(a=10,b=30)\n        >>> s2 = s.copy()\n        >>> type(s2) is Struct\n        True\n        '
        return Struct(dict.copy(self))

    def hasattr(self, key):
        if False:
            return 10
        "hasattr function available as a method.\n\n        Implemented like has_key.\n\n        Examples\n        --------\n        >>> s = Struct(a=10)\n        >>> s.hasattr('a')\n        True\n        >>> s.hasattr('b')\n        False\n        >>> s.hasattr('get')\n        False\n        "
        return key in self

    def allow_new_attr(self, allow=True):
        if False:
            i = 10
            return i + 15
        'Set whether new attributes can be created in this Struct.\n\n        This can be used to catch typos by verifying that the attribute user\n        tries to change already exists in this Struct.\n        '
        object.__setattr__(self, '_allownew', allow)

    def merge(self, __loc_data__=None, __conflict_solve=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        "Merge two Structs with customizable conflict resolution.\n\n        This is similar to :meth:`update`, but much more flexible. First, a\n        dict is made from data+key=value pairs. When merging this dict with\n        the Struct S, the optional dictionary 'conflict' is used to decide\n        what to do.\n\n        If conflict is not given, the default behavior is to preserve any keys\n        with their current value (the opposite of the :meth:`update` method's\n        behavior).\n\n        Parameters\n        ----------\n        __loc_data__ : dict, Struct\n            The data to merge into self\n        __conflict_solve : dict\n            The conflict policy dict.  The keys are binary functions used to\n            resolve the conflict and the values are lists of strings naming\n            the keys the conflict resolution function applies to.  Instead of\n            a list of strings a space separated string can be used, like\n            'a b c'.\n        **kw : dict\n            Additional key, value pairs to merge in\n\n        Notes\n        -----\n        The `__conflict_solve` dict is a dictionary of binary functions which will be used to\n        solve key conflicts.  Here is an example::\n\n            __conflict_solve = dict(\n                func1=['a','b','c'],\n                func2=['d','e']\n            )\n\n        In this case, the function :func:`func1` will be used to resolve\n        keys 'a', 'b' and 'c' and the function :func:`func2` will be used for\n        keys 'd' and 'e'.  This could also be written as::\n\n            __conflict_solve = dict(func1='a b c',func2='d e')\n\n        These functions will be called for each key they apply to with the\n        form::\n\n            func1(self['a'], other['a'])\n\n        The return value is used as the final merged value.\n\n        As a convenience, merge() provides five (the most commonly needed)\n        pre-defined policies: preserve, update, add, add_flip and add_s. The\n        easiest explanation is their implementation::\n\n            preserve = lambda old,new: old\n            update   = lambda old,new: new\n            add      = lambda old,new: old + new\n            add_flip = lambda old,new: new + old  # note change of order!\n            add_s    = lambda old,new: old + ' ' + new  # only for str!\n\n        You can use those four words (as strings) as keys instead\n        of defining them as functions, and the merge method will substitute\n        the appropriate functions for you.\n\n        For more complicated conflict resolution policies, you still need to\n        construct your own functions.\n\n        Examples\n        --------\n        This show the default policy:\n\n        >>> s = Struct(a=10,b=30)\n        >>> s2 = Struct(a=20,c=40)\n        >>> s.merge(s2)\n        >>> sorted(s.items())\n        [('a', 10), ('b', 30), ('c', 40)]\n\n        Now, show how to specify a conflict dict:\n\n        >>> s = Struct(a=10,b=30)\n        >>> s2 = Struct(a=20,b=40)\n        >>> conflict = {'update':'a','add':'b'}\n        >>> s.merge(s2,conflict)\n        >>> sorted(s.items())\n        [('a', 20), ('b', 70)]\n        "
        data_dict = dict(__loc_data__, **kw)
        preserve = lambda old, new: old
        update = lambda old, new: new
        add = lambda old, new: old + new
        add_flip = lambda old, new: new + old
        add_s = lambda old, new: old + ' ' + new
        conflict_solve = dict.fromkeys(self, preserve)
        if __conflict_solve:
            inv_conflict_solve_user = __conflict_solve.copy()
            for (name, func) in [('preserve', preserve), ('update', update), ('add', add), ('add_flip', add_flip), ('add_s', add_s)]:
                if name in inv_conflict_solve_user.keys():
                    inv_conflict_solve_user[func] = inv_conflict_solve_user[name]
                    del inv_conflict_solve_user[name]
            conflict_solve.update(self.__dict_invert(inv_conflict_solve_user))
        for key in data_dict:
            if key not in self:
                self[key] = data_dict[key]
            else:
                self[key] = conflict_solve[key](self[key], data_dict[key])