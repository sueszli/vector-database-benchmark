"""Base TFDecorator class and utility functions for working with decorators.

There are two ways to create decorators that TensorFlow can introspect into.
This is important for documentation generation purposes, so that function
signatures aren't obscured by the (*args, **kwds) signature that decorators
often provide.

1. Call `tf_decorator.make_decorator` on your wrapper function. If your
decorator is stateless, or can capture all of the variables it needs to work
with through lexical closure, this is the simplest option. Create your wrapper
function as usual, but instead of returning it, return
`tf_decorator.make_decorator(target, your_wrapper)`. This will attach some
decorator introspection metadata onto your wrapper and return it.

Example:

  def print_hello_before_calling(target):
    def wrapper(*args, **kwargs):
      print('hello')
      return target(*args, **kwargs)
    return tf_decorator.make_decorator(target, wrapper)

2. Derive from TFDecorator. If your decorator needs to be stateful, you can
implement it in terms of a TFDecorator. Store whatever state you need in your
derived class, and implement the `__call__` method to do your work before
calling into your target. You can retrieve the target via
`super(MyDecoratorClass, self).decorated_target`, and call it with whatever
parameters it needs.

Example:

  class CallCounter(tf_decorator.TFDecorator):
    def __init__(self, target):
      super(CallCounter, self).__init__('count_calls', target)
      self.call_count = 0

    def __call__(self, *args, **kwargs):
      self.call_count += 1
      return super(CallCounter, self).decorated_target(*args, **kwargs)

  def count_calls(target):
    return CallCounter(target)
"""
import inspect
from typing import Dict, Any

def _make_default_values(fullargspec: inspect.FullArgSpec) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    "Returns default values from the function's fullargspec."
    if fullargspec.defaults is not None:
        defaults = {name: value for (name, value) in zip(fullargspec.args[-len(fullargspec.defaults):], fullargspec.defaults)}
    else:
        defaults = {}
    if fullargspec.kwonlydefaults is not None:
        defaults.update(fullargspec.kwonlydefaults)
    return defaults

def fullargspec_to_signature(fullargspec: inspect.FullArgSpec) -> inspect.Signature:
    if False:
        print('Hello World!')
    'Repackages fullargspec information into an equivalent inspect.Signature.'
    defaults = _make_default_values(fullargspec)
    parameters = []
    for arg in fullargspec.args:
        parameters.append(inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=defaults.get(arg, inspect.Parameter.empty)))
    if fullargspec.varargs is not None:
        parameters.append(inspect.Parameter(fullargspec.varargs, inspect.Parameter.VAR_POSITIONAL))
    for kwarg in fullargspec.kwonlyargs:
        parameters.append(inspect.Parameter(kwarg, inspect.Parameter.KEYWORD_ONLY, default=defaults.get(kwarg, inspect.Parameter.empty)))
    if fullargspec.varkw is not None:
        parameters.append(inspect.Parameter(fullargspec.varkw, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)

def make_decorator(target, decorator_func, decorator_name=None, decorator_doc='', decorator_argspec=None):
    if False:
        return 10
    'Make a decorator from a wrapper and a target.\n\n  Args:\n    target: The final callable to be wrapped.\n    decorator_func: The wrapper function.\n    decorator_name: The name of the decorator. If `None`, the name of the\n      function calling make_decorator.\n    decorator_doc: Documentation specific to this application of\n      `decorator_func` to `target`.\n    decorator_argspec: Override the signature using FullArgSpec.\n\n  Returns:\n    The `decorator_func` argument with new metadata attached.\n  '
    if decorator_name is None:
        decorator_name = inspect.currentframe().f_back.f_code.co_name
    decorator = TFDecorator(decorator_name, target, decorator_doc, decorator_argspec)
    setattr(decorator_func, '_tf_decorator', decorator)
    if hasattr(target, '__name__'):
        decorator_func.__name__ = target.__name__
    if hasattr(target, '__qualname__'):
        decorator_func.__qualname__ = target.__qualname__
    if hasattr(target, '__module__'):
        decorator_func.__module__ = target.__module__
    if hasattr(target, '__dict__'):
        for name in target.__dict__:
            if name not in decorator_func.__dict__:
                decorator_func.__dict__[name] = target.__dict__[name]
    if hasattr(target, '__doc__'):
        decorator_func.__doc__ = decorator.__doc__
    decorator_func.__wrapped__ = target
    decorator_func.__original_wrapped__ = target
    if decorator_argspec:
        decorator_func.__signature__ = fullargspec_to_signature(decorator_argspec)
    elif callable(target):
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            pass
        else:
            bound_instance = _get_bound_instance(target)
            if bound_instance and 'self' in signature.parameters:
                signature = inspect.Signature(list(signature.parameters.values())[1:])
                decorator_func.__self__ = bound_instance
            decorator_func.__signature__ = signature
    return decorator_func

def _get_bound_instance(target):
    if False:
        while True:
            i = 10
    'Returns the instance any of the targets is attached to.'
    (decorators, target) = unwrap(target)
    for decorator in decorators:
        if inspect.ismethod(decorator.decorated_target):
            return decorator.decorated_target.__self__

def _has_tf_decorator_attr(obj):
    if False:
        print('Hello World!')
    'Checks if object has _tf_decorator attribute.\n\n  This check would work for mocked object as well since it would\n  check if returned attribute has the right type.\n\n  Args:\n    obj: Python object.\n  '
    return hasattr(obj, '_tf_decorator') and isinstance(getattr(obj, '_tf_decorator'), TFDecorator)

def rewrap(decorator_func, previous_target, new_target):
    if False:
        while True:
            i = 10
    'Injects a new target into a function built by make_decorator.\n\n  This function allows replacing a function wrapped by `decorator_func`,\n  assuming the decorator that wraps the function is written as described below.\n\n  The decorator function must use `<decorator name>.__wrapped__` instead of the\n  wrapped function that is normally used:\n\n  Example:\n\n      # Instead of this:\n      def simple_parametrized_wrapper(*args, **kwds):\n        return wrapped_fn(*args, **kwds)\n\n      tf_decorator.make_decorator(simple_parametrized_wrapper, wrapped_fn)\n\n      # Write this:\n      def simple_parametrized_wrapper(*args, **kwds):\n        return simple_parametrized_wrapper.__wrapped__(*args, **kwds)\n\n      tf_decorator.make_decorator(simple_parametrized_wrapper, wrapped_fn)\n\n  Note that this process modifies decorator_func.\n\n  Args:\n    decorator_func: Callable returned by `wrap`.\n    previous_target: Callable that needs to be replaced.\n    new_target: Callable to replace previous_target with.\n\n  Returns:\n    The updated decorator. If decorator_func is not a tf_decorator, new_target\n    is returned.\n  '
    cur = decorator_func
    innermost_decorator = None
    target = None
    while _has_tf_decorator_attr(cur):
        innermost_decorator = cur
        target = getattr(cur, '_tf_decorator')
        if target.decorated_target is previous_target:
            break
        cur = target.decorated_target
        assert cur is not None
    if innermost_decorator is None:
        assert decorator_func is previous_target
        return new_target
    target.decorated_target = new_target
    if inspect.ismethod(innermost_decorator):
        if hasattr(innermost_decorator, '__func__'):
            innermost_decorator.__func__.__wrapped__ = new_target
        elif hasattr(innermost_decorator, 'im_func'):
            innermost_decorator.im_func.__wrapped__ = new_target
        else:
            innermost_decorator.__wrapped__ = new_target
    else:
        innermost_decorator.__wrapped__ = new_target
    return decorator_func

def unwrap(maybe_tf_decorator):
    if False:
        return 10
    'Unwraps an object into a list of TFDecorators and a final target.\n\n  Args:\n    maybe_tf_decorator: Any callable object.\n\n  Returns:\n    A tuple whose first element is an list of TFDecorator-derived objects that\n    were applied to the final callable target, and whose second element is the\n    final undecorated callable target. If the `maybe_tf_decorator` parameter is\n    not decorated by any TFDecorators, the first tuple element will be an empty\n    list. The `TFDecorator` list is ordered from outermost to innermost\n    decorators.\n  '
    decorators = []
    cur = maybe_tf_decorator
    while True:
        if isinstance(cur, TFDecorator):
            decorators.append(cur)
        elif _has_tf_decorator_attr(cur):
            decorators.append(getattr(cur, '_tf_decorator'))
        else:
            break
        if not hasattr(decorators[-1], 'decorated_target'):
            break
        cur = decorators[-1].decorated_target
    return (decorators, cur)

class TFDecorator(object):
    """Base class for all TensorFlow decorators.

  TFDecorator captures and exposes the wrapped target, and provides details
  about the current decorator.
  """

    def __init__(self, decorator_name, target, decorator_doc='', decorator_argspec=None):
        if False:
            i = 10
            return i + 15
        self._decorated_target = target
        self._decorator_name = decorator_name
        self._decorator_doc = decorator_doc
        self._decorator_argspec = decorator_argspec
        if hasattr(target, '__name__'):
            self.__name__ = target.__name__
        if hasattr(target, '__qualname__'):
            self.__qualname__ = target.__qualname__
        if self._decorator_doc:
            self.__doc__ = self._decorator_doc
        elif hasattr(target, '__doc__') and target.__doc__:
            self.__doc__ = target.__doc__
        else:
            self.__doc__ = ''
        if decorator_argspec:
            self.__signature__ = fullargspec_to_signature(decorator_argspec)
        elif callable(target):
            try:
                self.__signature__ = inspect.signature(target)
            except (TypeError, ValueError):
                pass

    def __get__(self, instance, owner):
        if False:
            print('Hello World!')
        return self._decorated_target.__get__(instance, owner)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        return self._decorated_target(*args, **kwargs)

    @property
    def decorated_target(self):
        if False:
            for i in range(10):
                print('nop')
        return self._decorated_target

    @decorated_target.setter
    def decorated_target(self, decorated_target):
        if False:
            i = 10
            return i + 15
        self._decorated_target = decorated_target

    @property
    def decorator_name(self):
        if False:
            while True:
                i = 10
        return self._decorator_name

    @property
    def decorator_doc(self):
        if False:
            i = 10
            return i + 15
        return self._decorator_doc

    @property
    def decorator_argspec(self):
        if False:
            print('Hello World!')
        return self._decorator_argspec