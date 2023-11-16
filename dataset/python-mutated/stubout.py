class StubOutForTesting:
    """Sample Usage:
     You want os.path.exists() to always return true during testing.

     stubs = StubOutForTesting()
     stubs.Set(os.path, 'exists', lambda x: 1)
       ...
     stubs.UnsetAll()

     The above changes os.path.exists into a lambda that returns 1.  Once
     the ... part of the code finishes, the UnsetAll() looks up the old value
     of os.path.exists and restores it.

  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.cache = []
        self.stubs = []

    def __del__(self):
        if False:
            return 10
        self.SmartUnsetAll()
        self.UnsetAll()

    def SmartSet(self, obj, attr_name, new_attr):
        if False:
            while True:
                i = 10
        'Replace obj.attr_name with new_attr. This method is smart and works\n       at the module, class, and instance level while preserving proper\n       inheritance. It will not stub out C types however unless that has been\n       explicitly allowed by the type.\n\n       This method supports the case where attr_name is a staticmethod or a\n       classmethod of obj.\n\n       Notes:\n      - If obj is an instance, then it is its class that will actually be\n        stubbed. Note that the method Set() does not do that: if obj is\n        an instance, it (and not its class) will be stubbed.\n      - The stubbing is using the builtin getattr and setattr. So, the __get__\n        and __set__ will be called when stubbing (TODO: A better idea would\n        probably be to manipulate obj.__dict__ instead of getattr() and\n        setattr()).\n\n       Raises AttributeError if the attribute cannot be found.\n    '
        if inspect.ismodule(obj) or (not inspect.isclass(obj) and obj.__dict__.has_key(attr_name)):
            orig_obj = obj
            orig_attr = getattr(obj, attr_name)
        else:
            if not inspect.isclass(obj):
                mro = list(inspect.getmro(obj.__class__))
            else:
                mro = list(inspect.getmro(obj))
            mro.reverse()
            orig_attr = None
            for cls in mro:
                try:
                    orig_obj = cls
                    orig_attr = getattr(obj, attr_name)
                except AttributeError:
                    continue
        if orig_attr is None:
            raise AttributeError('Attribute not found.')
        old_attribute = obj.__dict__.get(attr_name)
        if old_attribute is not None and isinstance(old_attribute, staticmethod):
            orig_attr = staticmethod(orig_attr)
        self.stubs.append((orig_obj, attr_name, orig_attr))
        setattr(orig_obj, attr_name, new_attr)

    def SmartUnsetAll(self):
        if False:
            for i in range(10):
                print('nop')
        'Reverses all the SmartSet() calls, restoring things to their original\n    definition.  Its okay to call SmartUnsetAll() repeatedly, as later calls\n    have no effect if no SmartSet() calls have been made.\n\n    '
        self.stubs.reverse()
        for args in self.stubs:
            setattr(*args)
        self.stubs = []

    def Set(self, parent, child_name, new_child):
        if False:
            i = 10
            return i + 15
        "Replace child_name's old definition with new_child, in the context\n    of the given parent.  The parent could be a module when the child is a\n    function at module scope.  Or the parent could be a class when a class'\n    method is being replaced.  The named child is set to new_child, while\n    the prior definition is saved away for later, when UnsetAll() is called.\n\n    This method supports the case where child_name is a staticmethod or a\n    classmethod of parent.\n    "
        old_child = getattr(parent, child_name)
        old_attribute = parent.__dict__.get(child_name)
        if old_attribute is not None and isinstance(old_attribute, staticmethod):
            old_child = staticmethod(old_child)
        self.cache.append((parent, old_child, child_name))
        setattr(parent, child_name, new_child)

    def UnsetAll(self):
        if False:
            i = 10
            return i + 15
        'Reverses all the Set() calls, restoring things to their original\n    definition.  Its okay to call UnsetAll() repeatedly, as later calls have\n    no effect if no Set() calls have been made.\n\n    '
        self.cache.reverse()
        for (parent, old_child, child_name) in self.cache:
            setattr(parent, child_name, old_child)
        self.cache = []