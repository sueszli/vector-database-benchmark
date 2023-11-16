"""Facilities for creating multiple test combinations.

Here is a simple example for testing various optimizers in Eager and Graph:

class AdditionExample(test.TestCase, parameterized.TestCase):
  @combinations.generate(
     combinations.combine(mode=["graph", "eager"],
                          optimizer=[AdamOptimizer(),
                                     GradientDescentOptimizer()]))
  def testOptimizer(self, optimizer):
    ... f(optimizer)...

This will run `testOptimizer` 4 times with the specified optimizers: 2 in
Eager and 2 in Graph mode.
The test is going to accept the same parameters as the ones used in `combine()`.
The parameters need to match by name between the `combine()` call and the test
signature.  It is necessary to accept all parameters. See `OptionalParameter`
for a way to implement optional parameters.

`combine()` function is available for creating a cross product of various
options.  `times()` function exists for creating a product of N `combine()`-ed
results.

The execution of generated tests can be customized in a number of ways:
-  The test can be skipped if it is not running in the correct environment.
-  The arguments that are passed to the test can be additionally transformed.
-  The test can be run with specific Python context managers.
These behaviors can be customized by providing instances of `TestCombination` to
`generate()`.
"""
from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.test.combinations.TestCombination', v1=[])
class TestCombination:
    """Customize the behavior of `generate()` and the tests that it executes.

  Here is sequence of steps for executing a test combination:
    1. The test combination is evaluated for whether it should be executed in
       the given environment by calling `should_execute_combination`.
    2. If the test combination is going to be executed, then the arguments for
       all combined parameters are validated.  Some arguments can be handled in
       a special way.  This is achieved by implementing that logic in
       `ParameterModifier` instances that returned from `parameter_modifiers`.
    3. Before executing the test, `context_managers` are installed
       around it.
  """

    def should_execute_combination(self, kwargs):
        if False:
            i = 10
            return i + 15
        "Indicates whether the combination of test arguments should be executed.\n\n    If the environment doesn't satisfy the dependencies of the test\n    combination, then it can be skipped.\n\n    Args:\n      kwargs:  Arguments that are passed to the test combination.\n\n    Returns:\n      A tuple boolean and an optional string.  The boolean False indicates\n    that the test should be skipped.  The string would indicate a textual\n    description of the reason.  If the test is going to be executed, then\n    this method returns `None` instead of the string.\n    "
        del kwargs
        return (True, None)

    def parameter_modifiers(self):
        if False:
            return 10
        'Returns `ParameterModifier` instances that customize the arguments.'
        return []

    def context_managers(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Return context managers for running the test combination.\n\n    The test combination will run under all context managers that all\n    `TestCombination` instances return.\n\n    Args:\n      kwargs:  Arguments and their values that are passed to the test\n        combination.\n\n    Returns:\n      A list of instantiated context managers.\n    '
        del kwargs
        return []

@tf_export('__internal__.test.combinations.ParameterModifier', v1=[])
class ParameterModifier:
    """Customizes the behavior of a particular parameter.

  Users should override `modified_arguments()` to modify the parameter they
  want, eg: change the value of certain parameter or filter it from the params
  passed to the test case.

  See the sample usage below, it will change any negative parameters to zero
  before it gets passed to test case.
  ```
  class NonNegativeParameterModifier(ParameterModifier):

    def modified_arguments(self, kwargs, requested_parameters):
      updates = {}
      for name, value in kwargs.items():
        if value < 0:
          updates[name] = 0
      return updates
  ```
  """
    DO_NOT_PASS_TO_THE_TEST = object()

    def __init__(self, parameter_name=None):
        if False:
            return 10
        'Construct a parameter modifier that may be specific to a parameter.\n\n    Args:\n      parameter_name:  A `ParameterModifier` instance may operate on a class of\n        parameters or on a parameter with a particular name.  Only\n        `ParameterModifier` instances that are of a unique type or were\n        initialized with a unique `parameter_name` will be executed.\n        See `__eq__` and `__hash__`.\n    '
        self._parameter_name = parameter_name

    def modified_arguments(self, kwargs, requested_parameters):
        if False:
            i = 10
            return i + 15
        'Replace user-provided arguments before they are passed to a test.\n\n    This makes it possible to adjust user-provided arguments before passing\n    them to the test method.\n\n    Args:\n      kwargs:  The combined arguments for the test.\n      requested_parameters: The set of parameters that are defined in the\n        signature of the test method.\n\n    Returns:\n      A dictionary with updates to `kwargs`.  Keys with values set to\n      `ParameterModifier.DO_NOT_PASS_TO_THE_TEST` are going to be deleted and\n      not passed to the test.\n    '
        del kwargs, requested_parameters
        return {}

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        'Compare `ParameterModifier` by type and `parameter_name`.'
        if self is other:
            return True
        elif type(self) is type(other):
            return self._parameter_name == other._parameter_name
        else:
            return False

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

    def __hash__(self):
        if False:
            while True:
                i = 10
        'Compare `ParameterModifier` by type or `parameter_name`.'
        if self._parameter_name:
            return hash(self._parameter_name)
        else:
            return id(self.__class__)

@tf_export('__internal__.test.combinations.OptionalParameter', v1=[])
class OptionalParameter(ParameterModifier):
    """A parameter that is optional in `combine()` and in the test signature.

  `OptionalParameter` is usually used with `TestCombination` in the
  `parameter_modifiers()`. It allows `TestCombination` to skip certain
  parameters when passing them to `combine()`, since the `TestCombination` might
  consume the param and create some context based on the value it gets.

  See the sample usage below:

  ```
  class EagerGraphCombination(TestCombination):

    def context_managers(self, kwargs):
      mode = kwargs.pop("mode", None)
      if mode is None:
        return []
      elif mode == "eager":
        return [context.eager_mode()]
      elif mode == "graph":
        return [ops.Graph().as_default(), context.graph_mode()]
      else:
        raise ValueError(
            "'mode' has to be either 'eager' or 'graph', got {}".format(mode))

    def parameter_modifiers(self):
      return [test_combinations.OptionalParameter("mode")]
  ```

  When the test case is generated, the param "mode" will not be passed to the
  test method, since it is consumed by the `EagerGraphCombination`.
  """

    def modified_arguments(self, kwargs, requested_parameters):
        if False:
            print('Hello World!')
        if self._parameter_name in requested_parameters:
            return {}
        else:
            return {self._parameter_name: ParameterModifier.DO_NOT_PASS_TO_THE_TEST}

def generate(combinations, test_combinations=()):
    if False:
        i = 10
        return i + 15
    'A decorator for generating combinations of a test method or a test class.\n\n  Parameters of the test method must match by name to get the corresponding\n  value of the combination.  Tests must accept all parameters that are passed\n  other than the ones that are `OptionalParameter`.\n\n  Args:\n    combinations: a list of dictionaries created using combine() and times().\n    test_combinations: a tuple of `TestCombination` instances that customize\n      the execution of generated tests.\n\n  Returns:\n    a decorator that will cause the test method or the test class to be run\n    under the specified conditions.\n\n  Raises:\n    ValueError: if any parameters were not accepted by the test method\n  '

    def decorator(test_method_or_class):
        if False:
            return 10
        'The decorator to be returned.'
        named_combinations = []
        for combination in combinations:
            assert isinstance(combination, OrderedDict)
            name = ''.join(['_{}_{}'.format(''.join(filter(str.isalnum, key)), ''.join(filter(str.isalnum, _get_name(value, i)))) for (i, (key, value)) in enumerate(combination.items())])
            named_combinations.append(OrderedDict(list(combination.items()) + [('testcase_name', '_test{}'.format(name))]))
        if isinstance(test_method_or_class, type):
            class_object = test_method_or_class
            class_object._test_method_ids = test_method_ids = {}
            for (name, test_method) in class_object.__dict__.copy().items():
                if name.startswith(unittest.TestLoader.testMethodPrefix) and isinstance(test_method, types.FunctionType):
                    delattr(class_object, name)
                    methods = {}
                    parameterized._update_class_dict_for_param_test_case(class_object.__name__, methods, test_method_ids, name, parameterized._ParameterizedTestIter(_augment_with_special_arguments(test_method, test_combinations=test_combinations), named_combinations, parameterized._NAMED, name))
                    for (method_name, method) in methods.items():
                        setattr(class_object, method_name, method)
            return class_object
        else:
            test_method = _augment_with_special_arguments(test_method_or_class, test_combinations=test_combinations)
            return parameterized.named_parameters(*named_combinations)(test_method)
    return decorator

def _augment_with_special_arguments(test_method, test_combinations):
    if False:
        while True:
            i = 10

    def decorated(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'A wrapped test method that can treat some arguments in a special way.'
        original_kwargs = kwargs.copy()
        reasons_to_skip = []
        for combination in test_combinations:
            (should_execute, reason) = combination.should_execute_combination(original_kwargs.copy())
            if not should_execute:
                reasons_to_skip.append(' - ' + reason)
        if reasons_to_skip:
            self.skipTest('\n'.join(reasons_to_skip))
        customized_parameters = []
        for combination in test_combinations:
            customized_parameters.extend(combination.parameter_modifiers())
        customized_parameters = set(customized_parameters)

        def execute_test_method():
            if False:
                return 10
            requested_parameters = tf_inspect.getfullargspec(test_method).args
            for customized_parameter in customized_parameters:
                for (argument, value) in customized_parameter.modified_arguments(original_kwargs.copy(), requested_parameters).items():
                    if value is ParameterModifier.DO_NOT_PASS_TO_THE_TEST:
                        kwargs.pop(argument, None)
                    else:
                        kwargs[argument] = value
            omitted_arguments = set(requested_parameters).difference(set(list(kwargs.keys()) + ['self']))
            if omitted_arguments:
                raise ValueError('The test requires parameters whose arguments were not passed: {} .'.format(omitted_arguments))
            missing_arguments = set(list(kwargs.keys()) + ['self']).difference(set(requested_parameters))
            if missing_arguments:
                raise ValueError('The test does not take parameters that were passed : {} .'.format(missing_arguments))
            kwargs_to_pass = {}
            for parameter in requested_parameters:
                if parameter == 'self':
                    kwargs_to_pass[parameter] = self
                else:
                    kwargs_to_pass[parameter] = kwargs[parameter]
            test_method(**kwargs_to_pass)
        context_managers = []
        for combination in test_combinations:
            for manager in combination.context_managers(original_kwargs.copy()):
                context_managers.append(manager)
        if hasattr(contextlib, 'nested'):
            with contextlib.nested(*context_managers):
                execute_test_method()
        else:
            with contextlib.ExitStack() as context_stack:
                for manager in context_managers:
                    context_stack.enter_context(manager)
                execute_test_method()
    return decorated

@tf_export('__internal__.test.combinations.combine', v1=[])
def combine(**kwargs):
    if False:
        while True:
            i = 10
    'Generate combinations based on its keyword arguments.\n\n  Two sets of returned combinations can be concatenated using +.  Their product\n  can be computed using `times()`.\n\n  Args:\n    **kwargs: keyword arguments of form `option=[possibilities, ...]`\n         or `option=the_only_possibility`.\n\n  Returns:\n    a list of dictionaries for each combination. Keys in the dictionaries are\n    the keyword argument names.  Each key has one value - one of the\n    corresponding keyword argument values.\n  '
    if not kwargs:
        return [OrderedDict()]
    sort_by_key = lambda k: k[0]
    kwargs = OrderedDict(sorted(kwargs.items(), key=sort_by_key))
    first = list(kwargs.items())[0]
    rest = dict(list(kwargs.items())[1:])
    rest_combined = combine(**rest)
    key = first[0]
    values = first[1]
    if not isinstance(values, list):
        values = [values]
    return [OrderedDict(sorted(list(combined.items()) + [(key, v)], key=sort_by_key)) for v in values for combined in rest_combined]

@tf_export('__internal__.test.combinations.times', v1=[])
def times(*combined):
    if False:
        for i in range(10):
            print('nop')
    'Generate a product of N sets of combinations.\n\n  times(combine(a=[1,2]), combine(b=[3,4])) == combine(a=[1,2], b=[3,4])\n\n  Args:\n    *combined: N lists of dictionaries that specify combinations.\n\n  Returns:\n    a list of dictionaries for each combination.\n\n  Raises:\n    ValueError: if some of the inputs have overlapping keys.\n  '
    assert combined
    if len(combined) == 1:
        return combined[0]
    first = combined[0]
    rest_combined = times(*combined[1:])
    combined_results = []
    for a in first:
        for b in rest_combined:
            if set(a.keys()).intersection(set(b.keys())):
                raise ValueError('Keys need to not overlap: {} vs {}'.format(a.keys(), b.keys()))
            combined_results.append(OrderedDict(list(a.items()) + list(b.items())))
    return combined_results

@tf_export('__internal__.test.combinations.NamedObject', v1=[])
class NamedObject:
    """A class that translates an object into a good test name."""

    def __init__(self, name, obj):
        if False:
            return 10
        self._name = name
        self._obj = obj

    def __getattr__(self, name):
        if False:
            return 10
        return getattr(self._obj, name)

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._obj(*args, **kwargs)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self._obj.__iter__()

    def __repr__(self):
        if False:
            return 10
        return self._name

def _get_name(value, index):
    if False:
        for i in range(10):
            print('nop')
    return re.sub('0[xX][0-9a-fA-F]+', str(index), str(value))