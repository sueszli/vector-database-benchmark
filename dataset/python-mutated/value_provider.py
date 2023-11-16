"""A ValueProvider abstracts the notion of fetching a value that may or
may not be currently available.

This can be used to parameterize transforms that only read values in at
runtime, for example.
"""
from functools import wraps
from typing import Set
from apache_beam import error
__all__ = ['ValueProvider', 'StaticValueProvider', 'RuntimeValueProvider', 'NestedValueProvider', 'check_accessible']

class ValueProvider(object):
    """Base class that all other ValueProviders must implement.
  """

    def is_accessible(self):
        if False:
            i = 10
            return i + 15
        'Whether the contents of this ValueProvider is available to routines\n    that run at graph construction time.\n    '
        raise NotImplementedError('ValueProvider.is_accessible implemented in derived classes')

    def get(self):
        if False:
            while True:
                i = 10
        'Return the value wrapped by this ValueProvider.\n    '
        raise NotImplementedError('ValueProvider.get implemented in derived classes')

class StaticValueProvider(ValueProvider):
    """StaticValueProvider is an implementation of ValueProvider that allows
  for a static value to be provided.
  """

    def __init__(self, value_type, value):
        if False:
            return 10
        '\n    Args:\n        value_type: Type of the static value\n        value: Static value\n    '
        self.value_type = value_type
        self.value = value_type(value)

    def is_accessible(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def get(self):
        if False:
            return 10
        return self.value

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.value)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.value == other:
            return True
        if isinstance(other, StaticValueProvider):
            if self.value_type == other.value_type and self.value == other.value:
                return True
        return False

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((type(self), self.value_type, self.value))

class RuntimeValueProvider(ValueProvider):
    """RuntimeValueProvider is an implementation of ValueProvider that
  allows for a value to be provided at execution time rather than
  at graph construction time.
  """
    runtime_options = None
    experiments = set()

    def __init__(self, option_name, value_type, default_value):
        if False:
            print('Hello World!')
        self.option_name = option_name
        self.default_value = default_value
        self.value_type = value_type

    def is_accessible(self):
        if False:
            while True:
                i = 10
        return RuntimeValueProvider.runtime_options is not None

    @classmethod
    def get_value(cls, option_name, value_type, default_value):
        if False:
            return 10
        if not RuntimeValueProvider.runtime_options:
            return default_value
        candidate = RuntimeValueProvider.runtime_options.get(option_name)
        if candidate:
            return value_type(candidate)
        else:
            return default_value

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        if RuntimeValueProvider.runtime_options is None:
            raise error.RuntimeValueProviderError('%s.get() not called from a runtime context' % self)
        return RuntimeValueProvider.get_value(self.option_name, self.value_type, self.default_value)

    @classmethod
    def set_runtime_options(cls, pipeline_options):
        if False:
            while True:
                i = 10
        RuntimeValueProvider.runtime_options = pipeline_options
        RuntimeValueProvider.experiments = RuntimeValueProvider.get_value('experiments', set, set())

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s(option: %s, type: %s, default_value: %s)' % (self.__class__.__name__, self.option_name, self.value_type.__name__, repr(self.default_value))

class NestedValueProvider(ValueProvider):
    """NestedValueProvider is an implementation of ValueProvider that allows
  for wrapping another ValueProvider object.
  """

    def __init__(self, value, translator):
        if False:
            i = 10
            return i + 15
        'Creates a NestedValueProvider that wraps the provided ValueProvider.\n\n    Args:\n      value: ValueProvider object to wrap\n      translator: function that is applied to the ValueProvider\n    Raises:\n      ``RuntimeValueProviderError``: if any of the provided objects are not\n        accessible.\n    '
        self.value = value
        self.translator = translator

    def is_accessible(self):
        if False:
            print('Hello World!')
        return self.value.is_accessible()

    def get(self):
        if False:
            while True:
                i = 10
        try:
            return self.cached_value
        except AttributeError:
            self.cached_value = self.translator(self.value.get())
            return self.cached_value

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s(value: %s, translator: %s)' % (self.__class__.__name__, self.value, self.translator.__name__)

def check_accessible(value_provider_list):
    if False:
        while True:
            i = 10
    'A decorator that checks accessibility of a list of ValueProvider objects.\n\n  Args:\n    value_provider_list: list of ValueProvider objects\n  Raises:\n    ``RuntimeValueProviderError``: if any of the provided objects are not\n      accessible.\n  '
    assert isinstance(value_provider_list, list)

    def _check_accessible(fnc):
        if False:
            return 10

        @wraps(fnc)
        def _f(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            for obj in [getattr(self, vp) for vp in value_provider_list]:
                if not obj.is_accessible():
                    raise error.RuntimeValueProviderError('%s not accessible' % obj)
            return fnc(self, *args, **kwargs)
        return _f
    return _check_accessible