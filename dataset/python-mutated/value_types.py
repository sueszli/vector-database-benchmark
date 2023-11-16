"""Types of values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from fire import inspectutils
import six
VALUE_TYPES = (bool, six.string_types, six.integer_types, float, complex, type(Ellipsis), type(None), type(NotImplemented))

def IsGroup(component):
    if False:
        for i in range(10):
            print('nop')
    return not IsCommand(component) and (not IsValue(component))

def IsCommand(component):
    if False:
        return 10
    return inspect.isroutine(component) or inspect.isclass(component)

def IsValue(component):
    if False:
        print('Hello World!')
    return isinstance(component, VALUE_TYPES) or HasCustomStr(component)

def IsSimpleGroup(component):
    if False:
        for i in range(10):
            print('nop')
    'If a group is simple enough, then we treat it as a value in PrintResult.\n\n  Only if a group contains all value types do we consider it simple enough to\n  print as a value.\n\n  Args:\n    component: The group to check for value-group status.\n  Returns:\n    A boolean indicating if the group should be treated as a value for printing\n    purposes.\n  '
    assert isinstance(component, dict)
    for (unused_key, value) in component.items():
        if not IsValue(value) and (not isinstance(value, (list, dict))):
            return False
    return True

def HasCustomStr(component):
    if False:
        return 10
    "Determines if a component has a custom __str__ method.\n\n  Uses inspect.classify_class_attrs to determine the origin of the object's\n  __str__ method, if one is present. If it defined by `object` itself, then\n  it is not considered custom. Otherwise it is. This means that the __str__\n  methods of primitives like ints and floats are considered custom.\n\n  Objects with custom __str__ methods are treated as values and can be\n  serialized in places where more complex objects would have their help screen\n  shown instead.\n\n  Args:\n    component: The object to check for a custom __str__ method.\n  Returns:\n    Whether `component` has a custom __str__ method.\n  "
    if hasattr(component, '__str__'):
        class_attrs = inspectutils.GetClassAttrsDict(type(component)) or {}
        str_attr = class_attrs.get('__str__')
        if str_attr and str_attr.defining_class is not object:
            return True
    return False