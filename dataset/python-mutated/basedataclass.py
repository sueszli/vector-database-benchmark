from copy import deepcopy

class BaseDataClass(dict):
    """
    BaseDataClass

    Custom lightweight alternative to dataclasses, that work in older Python versions from before dataclasses.
    * Intentionally does not support positional arguments (because of https://stackoverflow.com/q/51575931/633921).
    * Does not need decorators to declare class props.
    * Ensures type consistency for properties declared in class
    * Unlike dataclasses, new props (set at runtime, but undeclared in the class) become part of the data.
    * Implemented using the AttrDict pattern, but it avoids self.__dict__ = self

    # Example use:
    class Person(BaseDataClass):
        # All props you declare need a default value (not None)
        first_name = ""
        last_name = ""
        age = 0
        metadata = {}  # Note: This will be cloned when you create a new instance, so you can ignore linter warnings

        def full_name(self):
            return self.first_name + " " + self.last_name

    print(Person(first_name=John, last_name="Wayne").full_name()) # John Wayne
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        for cls in reversed(self.__class__.__mro__):
            if cls in BaseDataClass.__mro__:
                continue
            defaults = {k: deepcopy(v) for (k, v) in vars(cls).items() if not k.startswith('__') and (not callable(getattr(cls, k)))}
            self.update(defaults)
        self.update(*args, **kwargs)

    def __dir__(self):
        if False:
            print('Hello World!')
        return dir(type(self)) + list(self.keys())

    def __delattr__(self, name):
        if False:
            return 10
        del self[name]

    def __getattribute__(self, key):
        if False:
            while True:
                i = 10
        try:
            return self[key]
        except KeyError:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if False:
            return 10
        self[key] = value

    def __setitem__(self, key, value, validate_type=True):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self.__class__, key):
            if key.startswith('__'):
                msg = f'Invalid property "{key}". Must not override class property.'
                raise KeyError(msg)
            class_val = getattr(self.__class__, key)
            if callable(class_val):
                msg = f'Invalid property "{key}". Must not override class method.'
                raise KeyError(msg)
            if validate_type and (not isinstance(value, type(class_val))):
                msg = f'"{key}" must be of type {type(class_val).__name__}, {type(value).__name__} given.'
                raise KeyError(msg)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        for (k, v) in dict(*args, **kwargs).items():
            self[k] = v