""" Support for toolset definition.
"""
import sys
import feature, property, generators, property_set
import b2.util.set
import bjam
from b2.util import cached, qualify_jam_action, is_iterable_typed, is_iterable
from b2.util.utility import *
from b2.util import bjam_signature, sequence
from b2.manager import get_manager
__re_split_last_segment = re.compile('^(.+)\\.([^\\.])*')
__re_two_ampersands = re.compile('(&&)')
__re_first_segment = re.compile('([^.]*).*')
__re_first_group = re.compile('[^.]*\\.(.*)')
_ignore_toolset_requirements = '--ignore-toolset-requirements' not in sys.argv

class Flag:

    def __init__(self, variable_name, values, condition, rule=None):
        if False:
            i = 10
            return i + 15
        assert isinstance(variable_name, basestring)
        assert is_iterable(values) and all((isinstance(v, (basestring, type(None))) for v in values))
        assert is_iterable_typed(condition, property_set.PropertySet)
        assert isinstance(rule, (basestring, type(None)))
        self.variable_name = variable_name
        self.values = values
        self.condition = condition
        self.rule = rule

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Flag(' + str(self.variable_name) + ', ' + str(self.values) + ', ' + str(self.condition) + ', ' + str(self.rule) + ')'

def reset():
    if False:
        for i in range(10):
            print('nop')
    ' Clear the module state. This is mainly for testing purposes.\n    '
    global __module_flags, __flags, __stv
    __module_flags = {}
    __flags = {}
    __stv = {}
reset()

def using(toolset_module, *args):
    if False:
        while True:
            i = 10
    if isinstance(toolset_module, (list, tuple)):
        toolset_module = toolset_module[0]
    loaded_toolset_module = get_manager().projects().load_module(toolset_module, [os.getcwd()])
    loaded_toolset_module.init(*args)

@bjam_signature((['rule_or_module', 'variable_name', 'condition', '*'], ['values', '*']))
def flags(rule_or_module, variable_name, condition, values=[]):
    if False:
        print('Hello World!')
    ' Specifies the flags (variables) that must be set on targets under certain\n        conditions, described by arguments.\n        rule_or_module:   If contains dot, should be a rule name.\n                          The flags will be applied when that rule is\n                          used to set up build actions.\n\n                          If does not contain dot, should be a module name.\n                          The flags will be applied for all rules in that\n                          module.\n                          If module for rule is different from the calling\n                          module, an error is issued.\n\n         variable_name:   Variable that should be set on target\n\n         condition        A condition when this flag should be applied.\n                          Should be set of property sets. If one of\n                          those property sets is contained in build\n                          properties, the flag will be used.\n                          Implied values are not allowed:\n                          "<toolset>gcc" should be used, not just\n                          "gcc". Subfeatures, like in "<toolset>gcc-3.2"\n                          are allowed. If left empty, the flag will\n                          always used.\n\n                          Propery sets may use value-less properties\n                          (\'<a>\'  vs. \'<a>value\') to match absent\n                          properties. This allows to separately match\n\n                             <architecture>/<address-model>64\n                             <architecture>ia64/<address-model>\n\n                          Where both features are optional. Without this\n                          syntax we\'d be forced to define "default" value.\n\n         values:          The value to add to variable. If <feature>\n                          is specified, then the value of \'feature\'\n                          will be added.\n    '
    assert isinstance(rule_or_module, basestring)
    assert isinstance(variable_name, basestring)
    assert is_iterable_typed(condition, basestring)
    assert is_iterable(values) and all((isinstance(v, (basestring, type(None))) for v in values))
    caller = bjam.caller()
    if not '.' in rule_or_module and caller and caller[:-1].startswith('Jamfile'):
        rule_or_module = qualify_jam_action(rule_or_module, caller)
    else:
        pass
    if condition and (not replace_grist(condition, '')):
        values = [condition]
        condition = None
    if condition:
        transformed = []
        for c in condition:
            pl = [property.create_from_string(s, False, True) for s in c.split('/')]
            pl = feature.expand_subfeatures(pl)
            transformed.append(property_set.create(pl))
        condition = transformed
        property.validate_property_sets(condition)
    __add_flag(rule_or_module, variable_name, condition, values)

def set_target_variables(manager, rule_or_module, targets, ps):
    if False:
        while True:
            i = 10
    '\n    '
    assert isinstance(rule_or_module, basestring)
    assert is_iterable_typed(targets, basestring)
    assert isinstance(ps, property_set.PropertySet)
    settings = __set_target_variables_aux(manager, rule_or_module, ps)
    if settings:
        for s in settings:
            for target in targets:
                manager.engine().set_target_variable(target, s[0], s[1], True)

def find_satisfied_condition(conditions, ps):
    if False:
        for i in range(10):
            print('nop')
    "Returns the first element of 'property-sets' which is a subset of\n    'properties', or an empty list if no such element exists."
    assert is_iterable_typed(conditions, property_set.PropertySet)
    assert isinstance(ps, property_set.PropertySet)
    for condition in conditions:
        found_all = True
        for i in condition.all():
            if i.value:
                found = i.value in ps.get(i.feature)
            else:
                found = not ps.get(i.feature)
            found_all = found_all and found
        if found_all:
            return condition
    return None

def register(toolset):
    if False:
        while True:
            i = 10
    ' Registers a new toolset.\n    '
    assert isinstance(toolset, basestring)
    feature.extend('toolset', [toolset])

def inherit_generators(toolset, properties, base, generators_to_ignore=[]):
    if False:
        while True:
            i = 10
    assert isinstance(toolset, basestring)
    assert is_iterable_typed(properties, basestring)
    assert isinstance(base, basestring)
    assert is_iterable_typed(generators_to_ignore, basestring)
    if not properties:
        properties = [replace_grist(toolset, '<toolset>')]
    base_generators = generators.generators_for_toolset(base)
    for g in base_generators:
        id = g.id()
        if not id in generators_to_ignore:
            (base, suffix) = split_action_id(id)
            new_id = toolset + '.' + suffix
            generators.register(g.clone(new_id, properties))

def inherit_flags(toolset, base, prohibited_properties=[]):
    if False:
        return 10
    "Brings all flag definitions from the 'base' toolset into the 'toolset'\n    toolset. Flag definitions whose conditions make use of properties in\n    'prohibited-properties' are ignored. Don't confuse property and feature, for\n    example <debug-symbols>on and <debug-symbols>off, so blocking one of them does\n    not block the other one.\n\n    The flag conditions are not altered at all, so if a condition includes a name,\n    or version of a base toolset, it won't ever match the inheriting toolset. When\n    such flag settings must be inherited, define a rule in base toolset module and\n    call it as needed."
    assert isinstance(toolset, basestring)
    assert isinstance(base, basestring)
    assert is_iterable_typed(prohibited_properties, basestring)
    for f in __module_flags.get(base, []):
        if not f.condition or b2.util.set.difference(f.condition, prohibited_properties):
            match = __re_first_group.match(f.rule)
            rule_ = None
            if match:
                rule_ = match.group(1)
            new_rule_or_module = ''
            if rule_:
                new_rule_or_module = toolset + '.' + rule_
            else:
                new_rule_or_module = toolset
            __add_flag(new_rule_or_module, f.variable_name, f.condition, f.values)

def inherit_rules(toolset, base):
    if False:
        print('Hello World!')
    engine = get_manager().engine()
    new_actions = {}
    for (action_name, action) in engine.actions.iteritems():
        (module, id) = split_action_id(action_name)
        if module == base:
            new_action_name = toolset + '.' + id
            if new_action_name not in engine.actions:
                new_actions[new_action_name] = action
    engine.actions.update(new_actions)

@cached
def __set_target_variables_aux(manager, rule_or_module, ps):
    if False:
        return 10
    ' Given a rule name and a property set, returns a list of tuples of\n        variables names and values, which must be set on targets for that\n        rule/properties combination.\n    '
    assert isinstance(rule_or_module, basestring)
    assert isinstance(ps, property_set.PropertySet)
    result = []
    for f in __flags.get(rule_or_module, []):
        if not f.condition or find_satisfied_condition(f.condition, ps):
            processed = []
            for v in f.values:
                processed += __handle_flag_value(manager, v, ps)
            for r in processed:
                result.append((f.variable_name, r))
    next = __re_split_last_segment.match(rule_or_module)
    if next:
        result.extend(__set_target_variables_aux(manager, next.group(1), ps))
    return result

def __handle_flag_value(manager, value, ps):
    if False:
        return 10
    assert isinstance(value, basestring)
    assert isinstance(ps, property_set.PropertySet)
    result = []
    if get_grist(value):
        f = feature.get(value)
        values = ps.get(f)
        for value in values:
            if f.dependency:
                result.append(value.actualize())
            elif f.path or f.free:
                if not __re_two_ampersands.search(value):
                    result.append(value)
                else:
                    result.extend(value.split('&&'))
            else:
                result.append(value)
    else:
        result.append(value)
    return sequence.unique(result, stable=True)

def __add_flag(rule_or_module, variable_name, condition, values):
    if False:
        print('Hello World!')
    ' Adds a new flag setting with the specified values.\n        Does no checking.\n    '
    assert isinstance(rule_or_module, basestring)
    assert isinstance(variable_name, basestring)
    assert is_iterable_typed(condition, property_set.PropertySet)
    assert is_iterable(values) and all((isinstance(v, (basestring, type(None))) for v in values))
    f = Flag(variable_name, values, condition, rule_or_module)
    m = __re_first_segment.match(rule_or_module)
    assert m
    module = m.group(1)
    __module_flags.setdefault(module, []).append(f)
    __flags.setdefault(rule_or_module, []).append(f)
__requirements = []

def requirements():
    if False:
        while True:
            i = 10
    "Return the list of global 'toolset requirements'.\n    Those requirements will be automatically added to the requirements of any main target."
    return __requirements

def add_requirements(requirements):
    if False:
        for i in range(10):
            print('nop')
    "Adds elements to the list of global 'toolset requirements'. The requirements\n    will be automatically added to the requirements for all main targets, as if\n    they were specified literally. For best results, all requirements added should\n    be conditional or indirect conditional."
    assert is_iterable_typed(requirements, basestring)
    if _ignore_toolset_requirements:
        __requirements.extend(requirements)

def inherit(toolset, base):
    if False:
        i = 10
        return i + 15
    assert isinstance(toolset, basestring)
    assert isinstance(base, basestring)
    get_manager().projects().load_module(base, ['.'])
    inherit_generators(toolset, [], base)
    inherit_flags(toolset, base)
    inherit_rules(toolset, base)