from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component

def generate_class_string(typename, props, description, namespace, prop_reorder_exceptions=None, max_props=None):
    if False:
        print('Hello World!')
    'Dynamically generate class strings to have nicely formatted docstrings,\n    keyword arguments, and repr.\n    Inspired by http://jameso.be/2013/08/06/namedtuple.html\n    Parameters\n    ----------\n    typename\n    props\n    description\n    namespace\n    prop_reorder_exceptions\n    Returns\n    -------\n    string\n    '
    c = 'class {typename}(Component):\n    """{docstring}"""\n    _children_props = {children_props}\n    _base_nodes = {base_nodes}\n    _namespace = \'{namespace}\'\n    _type = \'{typename}\'\n    @_explicitize_args\n    def __init__(self, {default_argtext}):\n        self._prop_names = {list_of_valid_keys}\n        self._valid_wildcard_attributes =            {list_of_valid_wildcard_attr_prefixes}\n        self.available_properties = {list_of_valid_keys}\n        self.available_wildcard_properties =            {list_of_valid_wildcard_attr_prefixes}\n        _explicit_args = kwargs.pop(\'_explicit_args\')\n        _locals = locals()\n        _locals.update(kwargs)  # For wildcard attrs and excess named props\n        args = {args}\n        {required_validation}\n        super({typename}, self).__init__({argtext})\n'
    filtered_props = filter_props(props) if prop_reorder_exceptions is not None and typename in prop_reorder_exceptions or (prop_reorder_exceptions is not None and 'ALL' in prop_reorder_exceptions) else reorder_props(filter_props(props))
    wildcard_prefixes = repr(parse_wildcards(props))
    list_of_valid_keys = repr(list(map(str, filtered_props.keys())))
    docstring = create_docstring(component_name=typename, props=filtered_props, description=description, prop_reorder_exceptions=prop_reorder_exceptions).replace('\r\n', '\n')
    required_args = required_props(filtered_props)
    is_children_required = 'children' in required_args
    required_args = [arg for arg in required_args if arg != 'children']
    prohibit_events(props)
    prop_keys = list(props.keys())
    if 'children' in props and 'children' in list_of_valid_keys:
        prop_keys.remove('children')
        default_argtext = 'children=None, '
        args = "{k: _locals[k] for k in _explicit_args if k != 'children'}"
        argtext = 'children=children, **args'
    else:
        default_argtext = ''
        args = '{k: _locals[k] for k in _explicit_args}'
        argtext = '**args'
    if len(required_args) == 0:
        required_validation = ''
    else:
        required_validation = f"\n        for k in {required_args}:\n            if k not in args:\n                raise TypeError(\n                    'Required argument `' + k + '` was not specified.')\n        "
    if is_children_required:
        required_validation += "\n        if 'children' not in _explicit_args:\n            raise TypeError('Required argument children was not specified.')\n        "
    default_arglist = [f'{p:s}=Component.REQUIRED' if props[p]['required'] else f'{p:s}=Component.UNDEFINED' for p in prop_keys if not p.endswith('-*') and p not in python_keywords and (p != 'setProps')]
    if max_props:
        final_max_props = max_props - (1 if 'children' in props else 0)
        if len(default_arglist) > final_max_props:
            default_arglist = default_arglist[:final_max_props]
            docstring += '\n\nNote: due to the large number of props for this component,\nnot all of them appear in the constructor signature, but\nthey may still be used as keyword arguments.'
    default_argtext += ', '.join(default_arglist + ['**kwargs'])
    nodes = collect_nodes({k: v for (k, v) in props.items() if k != 'children'})
    return dedent(c.format(typename=typename, namespace=namespace, filtered_props=filtered_props, list_of_valid_wildcard_attr_prefixes=wildcard_prefixes, list_of_valid_keys=list_of_valid_keys, docstring=docstring, default_argtext=default_argtext, args=args, argtext=argtext, required_validation=required_validation, children_props=nodes, base_nodes=filter_base_nodes(nodes) + ['children']))

def generate_class_file(typename, props, description, namespace, prop_reorder_exceptions=None, max_props=None):
    if False:
        return 10
    'Generate a Python class file (.py) given a class string.\n    Parameters\n    ----------\n    typename\n    props\n    description\n    namespace\n    prop_reorder_exceptions\n    Returns\n    -------\n    '
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n' + 'from dash.development.base_component import ' + 'Component, _explicitize_args\n\n\n'
    class_string = generate_class_string(typename, props, description, namespace, prop_reorder_exceptions, max_props)
    file_name = f'{typename:s}.py'
    file_path = os.path.join(namespace, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(import_string)
        f.write(class_string)
    print(f'Generated {file_name}')

def generate_imports(project_shortname, components):
    if False:
        print('Hello World!')
    with open(os.path.join(project_shortname, '_imports_.py'), 'w', encoding='utf-8') as f:
        component_imports = '\n'.join((f'from .{x} import {x}' for x in components))
        all_list = ',\n'.join((f'    "{x}"' for x in components))
        imports_string = f'{component_imports}\n\n__all__ = [\n{all_list}\n]'
        f.write(imports_string)

def generate_classes_files(project_shortname, metadata, *component_generators):
    if False:
        return 10
    components = []
    for (component_path, component_data) in metadata.items():
        component_name = component_path.split('/')[-1].split('.')[0]
        components.append(component_name)
        for generator in component_generators:
            generator(component_name, component_data['props'], component_data['description'], project_shortname)
    return components

def generate_class(typename, props, description, namespace, prop_reorder_exceptions=None):
    if False:
        print('Hello World!')
    'Generate a Python class object given a class string.\n    Parameters\n    ----------\n    typename\n    props\n    description\n    namespace\n    Returns\n    -------\n    '
    string = generate_class_string(typename, props, description, namespace, prop_reorder_exceptions)
    scope = {'Component': Component, '_explicitize_args': _explicitize_args}
    exec(string, scope)
    result = scope[typename]
    return result

def required_props(props):
    if False:
        i = 10
        return i + 15
    'Pull names of required props from the props object.\n    Parameters\n    ----------\n    props: dict\n    Returns\n    -------\n    list\n        List of prop names (str) that are required for the Component\n    '
    return [prop_name for (prop_name, prop) in list(props.items()) if prop['required']]

def create_docstring(component_name, props, description, prop_reorder_exceptions=None):
    if False:
        return 10
    'Create the Dash component docstring.\n    Parameters\n    ----------\n    component_name: str\n        Component name\n    props: dict\n        Dictionary with {propName: propMetadata} structure\n    description: str\n        Component description\n    Returns\n    -------\n    str\n        Dash component docstring\n    '
    props = props if prop_reorder_exceptions is not None and component_name in prop_reorder_exceptions or (prop_reorder_exceptions is not None and 'ALL' in prop_reorder_exceptions) else reorder_props(props)
    n = 'n' if component_name[0].lower() in 'aeiou' else ''
    args = '\n'.join((create_prop_docstring(prop_name=p, type_object=prop['type'] if 'type' in prop else prop['flowType'], required=prop['required'], description=prop['description'], default=prop.get('defaultValue'), indent_num=0, is_flow_type='flowType' in prop and 'type' not in prop) for (p, prop) in filter_props(props).items()))
    return f'A{n} {component_name} component.\n{description}\n\nKeyword arguments:\n{args}'

def prohibit_events(props):
    if False:
        print('Hello World!')
    'Events have been removed. Raise an error if we see dashEvents or\n    fireEvents.\n    Parameters\n    ----------\n    props: dict\n        Dictionary with {propName: propMetadata} structure\n    Raises\n    -------\n    ?\n    '
    if 'dashEvents' in props or 'fireEvents' in props:
        raise NonExistentEventException('Events are no longer supported by dash. Use properties instead, eg `n_clicks` instead of a `click` event.')

def parse_wildcards(props):
    if False:
        print('Hello World!')
    'Pull out the wildcard attributes from the Component props.\n    Parameters\n    ----------\n    props: dict\n        Dictionary with {propName: propMetadata} structure\n    Returns\n    -------\n    list\n        List of Dash valid wildcard prefixes\n    '
    list_of_valid_wildcard_attr_prefixes = []
    for wildcard_attr in ['data-*', 'aria-*']:
        if wildcard_attr in props:
            list_of_valid_wildcard_attr_prefixes.append(wildcard_attr[:-1])
    return list_of_valid_wildcard_attr_prefixes

def reorder_props(props):
    if False:
        for i in range(10):
            print('nop')
    'If "children" is in props, then move it to the front to respect dash\n    convention, then \'id\', then the remaining props sorted by prop name\n    Parameters\n    ----------\n    props: dict\n        Dictionary with {propName: propMetadata} structure\n    Returns\n    -------\n    dict\n        Dictionary with {propName: propMetadata} structure\n    '
    props1 = [('children', '')] if 'children' in props else []
    props2 = [('id', '')] if 'id' in props else []
    return OrderedDict(props1 + props2 + sorted(list(props.items())))

def filter_props(props):
    if False:
        print('Hello World!')
    'Filter props from the Component arguments to exclude:\n        - Those without a "type" or a "flowType" field\n        - Those with arg.type.name in {\'func\', \'symbol\', \'instanceOf\'}\n    Parameters\n    ----------\n    props: dict\n        Dictionary with {propName: propMetadata} structure\n    Returns\n    -------\n    dict\n        Filtered dictionary with {propName: propMetadata} structure\n    Examples\n    --------\n    ```python\n    prop_args = {\n        \'prop1\': {\n            \'type\': {\'name\': \'bool\'},\n            \'required\': False,\n            \'description\': \'A description\',\n            \'flowType\': {},\n            \'defaultValue\': {\'value\': \'false\', \'computed\': False},\n        },\n        \'prop2\': {\'description\': \'A prop without a type\'},\n        \'prop3\': {\n            \'type\': {\'name\': \'func\'},\n            \'description\': \'A function prop\',\n        },\n    }\n    # filtered_prop_args is now\n    # {\n    #    \'prop1\': {\n    #        \'type\': {\'name\': \'bool\'},\n    #        \'required\': False,\n    #        \'description\': \'A description\',\n    #        \'flowType\': {},\n    #        \'defaultValue\': {\'value\': \'false\', \'computed\': False},\n    #    },\n    # }\n    filtered_prop_args = filter_props(prop_args)\n    ```\n    '
    filtered_props = copy.deepcopy(props)
    for (arg_name, arg) in list(filtered_props.items()):
        if 'type' not in arg and 'flowType' not in arg:
            filtered_props.pop(arg_name)
            continue
        if 'type' in arg:
            arg_type = arg['type']['name']
            if arg_type in {'func', 'symbol', 'instanceOf'}:
                filtered_props.pop(arg_name)
        elif 'flowType' in arg:
            arg_type_name = arg['flowType']['name']
            if arg_type_name == 'signature':
                if 'type' not in arg['flowType'] or arg['flowType']['type'] != 'object':
                    filtered_props.pop(arg_name)
        else:
            raise ValueError
    return filtered_props

def fix_keywords(txt):
    if False:
        i = 10
        return i + 15
    '\n    replaces javascript keywords true, false, null with Python keywords\n    '
    fix_word = {'true': 'True', 'false': 'False', 'null': 'None'}
    for (js_keyword, python_keyword) in fix_word.items():
        txt = txt.replace(js_keyword, python_keyword)
    return txt

def create_prop_docstring(prop_name, type_object, required, description, default, indent_num, is_flow_type=False):
    if False:
        print('Hello World!')
    "Create the Dash component prop docstring.\n    Parameters\n    ----------\n    prop_name: str\n        Name of the Dash component prop\n    type_object: dict\n        react-docgen-generated prop type dictionary\n    required: bool\n        Component is required?\n    description: str\n        Dash component description\n    default: dict\n        Either None if a default value is not defined, or\n        dict containing the key 'value' that defines a\n        default value for the prop\n    indent_num: int\n        Number of indents to use for the context block\n        (creates 2 spaces for every indent)\n    is_flow_type: bool\n        Does the prop use Flow types? Otherwise, uses PropTypes\n    Returns\n    -------\n    str\n        Dash component prop docstring\n    "
    py_type_name = js_to_py_type(type_object=type_object, is_flow_type=is_flow_type, indent_num=indent_num)
    indent_spacing = '  ' * indent_num
    default = default['value'] if default else ''
    default = fix_keywords(default)
    is_required = 'optional'
    if required:
        is_required = 'required'
    elif default and default not in ['None', '{}', '[]']:
        is_required = 'default ' + default.replace('\n', '')
    period = '.' if description else ''
    description = description.strip().strip('.').replace('"', '\\"') + period
    desc_indent = indent_spacing + '    '
    description = fill(description, initial_indent=desc_indent, subsequent_indent=desc_indent, break_long_words=False, break_on_hyphens=False)
    description = f'\n{description}' if description else ''
    colon = ':' if description else ''
    description = fix_keywords(description)
    if '\n' in py_type_name:
        dict_or_list = 'list of dicts' if py_type_name.startswith('list') else 'dict'
        (intro1, intro2, dict_descr) = py_type_name.partition('with keys:')
        intro = f'`{prop_name}` is a {intro1}{intro2}'
        intro = fill(intro, initial_indent=desc_indent, subsequent_indent=desc_indent, break_long_words=False, break_on_hyphens=False)
        if '| dict with keys:' in dict_descr:
            (dict_part1, dict_part2) = dict_descr.split(' |', 1)
            dict_part2 = ''.join([desc_indent, 'Or', dict_part2])
            dict_descr = f'{dict_part1}\n\n  {dict_part2}'
        current_indent = dict_descr.lstrip('\n').find('-')
        if current_indent == len(indent_spacing):
            dict_descr = ''.join(('\n\n    ' + line for line in dict_descr.splitlines() if line != ''))
        return f'\n{indent_spacing}- {prop_name} ({dict_or_list}; {is_required}){colon}{description}\n\n{intro}{dict_descr}'
    tn = f'{py_type_name}; ' if py_type_name else ''
    return f'\n{indent_spacing}- {prop_name} ({tn}{is_required}){colon}{description}'

def map_js_to_py_types_prop_types(type_object, indent_num):
    if False:
        while True:
            i = 10
    'Mapping from the PropTypes js type object to the Python type.'

    def shape_or_exact():
        if False:
            while True:
                i = 10
        return 'dict with keys:\n' + '\n'.join((create_prop_docstring(prop_name=prop_name, type_object=prop, required=prop['required'], description=prop.get('description', ''), default=prop.get('defaultValue'), indent_num=indent_num + 2) for (prop_name, prop) in sorted(list(type_object['value'].items()))))

    def array_of():
        if False:
            for i in range(10):
                print('nop')
        inner = js_to_py_type(type_object['value'])
        if inner:
            return 'list of ' + (inner + 's' if inner.split(' ')[0] != 'dict' else inner.replace('dict', 'dicts', 1))
        return 'list'

    def tuple_of():
        if False:
            while True:
                i = 10
        elements = [js_to_py_type(element) for element in type_object['elements']]
        return f"list of {len(elements)} elements: [{', '.join(elements)}]"
    return dict(array=lambda : 'list', bool=lambda : 'boolean', number=lambda : 'number', string=lambda : 'string', object=lambda : 'dict', any=lambda : 'boolean | number | string | dict | list', element=lambda : 'dash component', node=lambda : 'a list of or a singular dash component, string or number', enum=lambda : 'a value equal to: ' + ', '.join((str(t['value']) for t in type_object['value'])), union=lambda : ' | '.join((js_to_py_type(subType) for subType in type_object['value'] if js_to_py_type(subType) != '')), arrayOf=array_of, objectOf=lambda : 'dict with strings as keys and values of type ' + js_to_py_type(type_object['value']), shape=shape_or_exact, exact=shape_or_exact, tuple=tuple_of)

def map_js_to_py_types_flow_types(type_object):
    if False:
        while True:
            i = 10
    'Mapping from the Flow js types to the Python type.'
    return dict(array=lambda : 'list', boolean=lambda : 'boolean', number=lambda : 'number', string=lambda : 'string', Object=lambda : 'dict', any=lambda : 'bool | number | str | dict | list', Element=lambda : 'dash component', Node=lambda : 'a list of or a singular dash component, string or number', union=lambda : ' | '.join((js_to_py_type(subType) for subType in type_object['elements'] if js_to_py_type(subType) != '')), Array=lambda : 'list' + (f" of {js_to_py_type(type_object['elements'][0])}s" if js_to_py_type(type_object['elements'][0]) != '' else ''), signature=lambda indent_num: 'dict with keys:\n' + '\n'.join((create_prop_docstring(prop_name=prop['key'], type_object=prop['value'], required=prop['value']['required'], description=prop['value'].get('description', ''), default=prop.get('defaultValue'), indent_num=indent_num + 2, is_flow_type=True) for prop in type_object['signature']['properties'])))

def js_to_py_type(type_object, is_flow_type=False, indent_num=0):
    if False:
        while True:
            i = 10
    'Convert JS types to Python types for the component definition.\n    Parameters\n    ----------\n    type_object: dict\n        react-docgen-generated prop type dictionary\n    is_flow_type: bool\n        Does the prop use Flow types? Otherwise, uses PropTypes\n    indent_num: int\n        Number of indents to use for the docstring for the prop\n    Returns\n    -------\n    str\n        Python type string\n    '
    js_type_name = type_object['name']
    js_to_py_types = map_js_to_py_types_flow_types(type_object=type_object) if is_flow_type else map_js_to_py_types_prop_types(type_object=type_object, indent_num=indent_num)
    if 'computed' in type_object and type_object['computed'] or type_object.get('type', '') == 'function':
        return ''
    if js_type_name in js_to_py_types:
        if js_type_name == 'signature':
            return js_to_py_types[js_type_name](indent_num)
        return js_to_py_types[js_type_name]()
    return ''