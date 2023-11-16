""" Nuitka templates can have more checks that the normal '%' operation.

This wraps strings with a class derived from "str" that does more checks.
"""
from nuitka import Options
from nuitka.__past__ import iterItems
from nuitka.Tracing import optimization_logger

class TemplateWrapper(object):
    """Wrapper around templates.

    To better trace and control template usage.

    """

    def __init__(self, name, value):
        if False:
            print('Hello World!')
        self.name = name
        self.value = value

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.value

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self.__class__(self.name + '+' + other.name, self.value + other.value)

    def __mod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        assert type(other) is dict, self.name
        for key in other.keys():
            if '%%(%s)' % key not in self.value:
                optimization_logger.warning('Extra value %r provided to template %r.' % (key, self.name))
        try:
            return self.value % other
        except KeyError as e:
            raise KeyError(self.name, *e.args)

    def split(self, sep):
        if False:
            for i in range(10):
                print('nop')
        return self.value.split(sep)

def enableDebug(globals_dict):
    if False:
        return 10
    templates = dict(globals_dict)
    for (template_name, template_value) in iterItems(templates):
        if template_name.startswith('_'):
            continue
        if type(template_value) is str and '{%' not in template_value:
            globals_dict[template_name] = TemplateWrapper(template_name, template_value)

def checkDebug(globals_dict):
    if False:
        for i in range(10):
            print('nop')
    if Options.is_debug:
        enableDebug(globals_dict)