import importlib
import random
import re
from ..utils import age_restricted, bug_reports_message, classproperty, variadic, write_string
ALLOWED_CLASSMETHODS = {'extract_from_webpage', 'get_testcases', 'get_webpage_testcases'}
_WARNED = False

class LazyLoadMetaClass(type):

    def __getattr__(cls, name):
        if False:
            print('Hello World!')
        global _WARNED
        if '_real_class' not in cls.__dict__ and name not in ALLOWED_CLASSMETHODS and (not _WARNED):
            _WARNED = True
            write_string(f'WARNING: Falling back to normal extractor since lazy extractor {cls.__name__} does not have attribute {name}{bug_reports_message()}\n')
        return getattr(cls.real_class, name)

class LazyLoadExtractor(metaclass=LazyLoadMetaClass):

    @classproperty
    def real_class(cls):
        if False:
            i = 10
            return i + 15
        if '_real_class' not in cls.__dict__:
            cls._real_class = getattr(importlib.import_module(cls._module), cls.__name__)
        return cls._real_class

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        instance = cls.real_class.__new__(cls.real_class)
        instance.__init__(*args, **kwargs)
        return instance