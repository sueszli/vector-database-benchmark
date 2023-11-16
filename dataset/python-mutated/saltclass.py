import glob
import logging
import os
import re
from jinja2 import Environment, FileSystemLoader
import salt.utils.path
import salt.utils.yaml
log = logging.getLogger(__name__)

def render_jinja(_file, salt_data):
    if False:
        for i in range(10):
            print('nop')
    j_env = Environment(loader=FileSystemLoader(os.path.dirname(_file)))
    j_env.globals.update({'__opts__': salt_data['__opts__'], '__salt__': salt_data['__salt__'], '__grains__': salt_data['__grains__'], '__pillar__': salt_data['__pillar__'], 'minion_id': salt_data['minion_id']})
    j_render = j_env.get_template(os.path.basename(_file)).render()
    return j_render

def render_yaml(_file, salt_data):
    if False:
        while True:
            i = 10
    return salt.utils.yaml.safe_load(render_jinja(_file, salt_data))

def get_class(_class, salt_data):
    if False:
        i = 10
        return i + 15
    l_files = []
    saltclass_path = salt_data['path']
    (straight, sub_init, sub_straight) = get_class_paths(_class, saltclass_path)
    for (root, dirs, files) in salt.utils.path.os_walk(os.path.join(saltclass_path, 'classes'), followlinks=True):
        for l_file in files:
            l_files.append(os.path.join(root, l_file))
    if straight in l_files:
        return render_yaml(straight, salt_data)
    if sub_straight in l_files:
        return render_yaml(sub_straight, salt_data)
    if sub_init in l_files:
        return render_yaml(sub_init, salt_data)
    log.warning('%s: Class definition not found', _class)
    return {}

def get_class_paths(_class, saltclass_path):
    if False:
        i = 10
        return i + 15
    '\n    Converts the dotted notation of a saltclass class to its possible file counterparts.\n\n    :param str _class: Dotted notation of the class\n    :param str saltclass_path: Root to saltclass storage\n    :return: 3-tuple of possible file counterparts\n    :rtype: tuple(str)\n    '
    straight = os.path.join(saltclass_path, 'classes', '{}.yml'.format(_class))
    sub_straight = os.path.join(saltclass_path, 'classes', '{}.yml'.format(_class.replace('.', os.sep)))
    sub_init = os.path.join(saltclass_path, 'classes', _class.replace('.', os.sep), 'init.yml')
    return (straight, sub_init, sub_straight)

def get_class_from_file(_file, saltclass_path):
    if False:
        for i in range(10):
            print('nop')
    "\n    Converts the absolute path to a saltclass file back to the dotted notation.\n\n    .. code-block:: python\n\n       print(get_class_from_file('/srv/saltclass/classes/services/nginx/init.yml', '/srv/saltclass'))\n       # services.nginx\n\n    :param str _file: Absolute path to file\n    :param str saltclass_path: Root to saltclass storage\n    :return: class name in dotted notation\n    :rtype: str\n    "
    _file = _file[len(os.path.join(saltclass_path, 'classes')) + len(os.sep):]
    _file = _file[:-4]
    _file = _file.replace(os.sep, '.')
    if _file.endswith('.init'):
        _file = _file[:-5]
    return _file

def get_env_from_dict(exp_dict_list):
    if False:
        for i in range(10):
            print('nop')
    environment = ''
    for s_class in exp_dict_list:
        if 'environment' in s_class:
            environment = s_class['environment']
    return environment

def dict_merge(a, b, path=None):
    if False:
        i = 10
        return i + 15
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], list) and isinstance(b[key], list):
                if b[key][0] == '^':
                    b[key].pop(0)
                    a[key] = b[key]
                else:
                    a[key].extend(b[key])
            elif isinstance(a[key], dict) and isinstance(b[key], dict):
                dict_merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def dict_search_and_replace(d, old, new, expanded):
    if False:
        i = 10
        return i + 15
    for (k, v) in d.items():
        if isinstance(v, dict):
            dict_search_and_replace(d[k], old, new, expanded)
        if isinstance(v, list):
            x = 0
            for i in v:
                if isinstance(i, dict):
                    dict_search_and_replace(v[x], old, new, expanded)
                if isinstance(i, str):
                    if i == old:
                        v[x] = new
                x = x + 1
        if v == old:
            d[k] = new
    return d

def find_value_to_expand(x, v):
    if False:
        while True:
            i = 10
    a = x
    for i in v[2:-1].split(':'):
        if a is None:
            return v
        if i in a:
            a = a.get(i)
        else:
            return v
    return a

def find_and_process_re(_str, v, k, b, expanded):
    if False:
        return 10
    vre = re.finditer('(^|.)\\$\\{.*?\\}', _str)
    if vre:
        for re_v in vre:
            re_str = str(re_v.group())
            if re_str.startswith('\\'):
                v_new = _str.replace(re_str, re_str.lstrip('\\'))
                b = dict_search_and_replace(b, _str, v_new, expanded)
                expanded.append(k)
            elif not re_str.startswith('$'):
                v_expanded = find_value_to_expand(b, re_str[1:])
                v_new = _str.replace(re_str[1:], v_expanded)
                b = dict_search_and_replace(b, _str, v_new, expanded)
                _str = v_new
                expanded.append(k)
            else:
                v_expanded = find_value_to_expand(b, re_str)
                if isinstance(v, str):
                    v_new = v.replace(re_str, v_expanded)
                else:
                    v_new = _str.replace(re_str, v_expanded)
                b = dict_search_and_replace(b, _str, v_new, expanded)
                _str = v_new
                v = v_new
                expanded.append(k)
    return b

def expand_variables(a, b, expanded, path=None):
    if False:
        while True:
            i = 10
    if path is None:
        b = a.copy()
        path = []
    for (k, v) in a.items():
        if isinstance(v, dict):
            expand_variables(v, b, expanded, path + [str(k)])
        else:
            if isinstance(v, list):
                for i in v:
                    if isinstance(i, dict):
                        expand_variables(i, b, expanded, path + [str(k)])
                    if isinstance(i, str):
                        b = find_and_process_re(i, v, k, b, expanded)
            if isinstance(v, str):
                b = find_and_process_re(v, v, k, b, expanded)
    return b

def match_class_glob(_class, saltclass_path):
    if False:
        while True:
            i = 10
    "\n    Takes a class name possibly including `*` or `?` wildcards (or any other wildcards supportet by `glob.glob`) and\n    returns a list of expanded class names without wildcards.\n\n    .. code-block:: python\n\n       classes = match_class_glob('services.*', '/srv/saltclass')\n       print(classes)\n       # services.mariadb\n       # services.nginx...\n\n\n    :param str _class: dotted class name, globbing allowed.\n    :param str saltclass_path: path to the saltclass root directory.\n\n    :return: The list of expanded class matches.\n    :rtype: list(str)\n    "
    (straight, sub_init, sub_straight) = get_class_paths(_class, saltclass_path)
    classes = []
    matches = []
    matches.extend(glob.glob(straight))
    matches.extend(glob.glob(sub_straight))
    matches.extend(glob.glob(sub_init))
    if not matches:
        log.warning('%s: Class globbing did not yield any results', _class)
    for match in matches:
        classes.append(get_class_from_file(match, saltclass_path))
    return classes

def expand_classes_glob(classes, salt_data):
    if False:
        print('Hello World!')
    '\n    Expand the list of `classes` to no longer include any globbing.\n\n    :param iterable(str) classes: Iterable of classes\n    :param dict salt_data: configuration data\n    :return: Expanded list of classes with resolved globbing\n    :rtype: list(str)\n    '
    all_classes = []
    expanded_classes = []
    saltclass_path = salt_data['path']
    for _class in classes:
        all_classes.extend(match_class_glob(_class, saltclass_path))
    for _class in all_classes:
        if _class not in expanded_classes:
            expanded_classes.append(_class)
    return expanded_classes

def expand_classes_in_order(minion_dict, salt_data, seen_classes, expanded_classes, classes_to_expand):
    if False:
        print('Hello World!')
    if not classes_to_expand and 'classes' in minion_dict:
        classes_to_expand = minion_dict['classes']
    classes_to_expand = expand_classes_glob(classes_to_expand, salt_data)
    for klass in classes_to_expand:
        if klass not in seen_classes:
            seen_classes.append(klass)
            expanded_classes[klass] = get_class(klass, salt_data)
            if expanded_classes[klass] is None:
                expanded_classes[klass] = {}
            new_pillars = expanded_classes[klass].get('pillars', {})
            if new_pillars:
                dict_merge(salt_data['__pillar__'], new_pillars)
            if expanded_classes[klass].get('classes'):
                l_id = classes_to_expand.index(klass)
                classes_to_expand[l_id:l_id] = expanded_classes[klass]['classes']
                expand_classes_in_order(minion_dict, salt_data, seen_classes, expanded_classes, classes_to_expand)
            else:
                expand_classes_in_order(minion_dict, salt_data, seen_classes, expanded_classes, classes_to_expand)
    tmp = []
    for t_element in classes_to_expand:
        if t_element not in tmp:
            tmp.append(t_element)
    classes_to_expand = tmp
    ord_expanded_classes = []
    ord_expanded_states = []
    for ord_klass in classes_to_expand:
        ord_expanded_classes.append(expanded_classes[ord_klass])
        if 'states' in expanded_classes[ord_klass] and expanded_classes[ord_klass]['states'] is None:
            expanded_classes[ord_klass]['states'] = {}
        if 'states' in expanded_classes[ord_klass]:
            ord_expanded_states.extend(expanded_classes[ord_klass]['states'])
    if 'states' in minion_dict and minion_dict['states'] is None:
        minion_dict['states'] = []
    if 'states' in minion_dict:
        ord_expanded_states.extend(minion_dict['states'])
    ord_expanded_classes.append(minion_dict)
    return (ord_expanded_classes, classes_to_expand, ord_expanded_states)

def expanded_dict_from_minion(minion_id, salt_data):
    if False:
        print('Hello World!')
    _file = ''
    saltclass_path = salt_data['path']
    for (root, dirs, files) in salt.utils.path.os_walk(os.path.join(saltclass_path, 'nodes'), followlinks=True):
        for minion_file in files:
            if minion_file == '{}.yml'.format(minion_id):
                _file = os.path.join(root, minion_file)
    node_dict = {}
    if _file:
        node_dict[minion_id] = render_yaml(_file, salt_data)
    else:
        log.warning('%s: Node definition not found', minion_id)
        node_dict[minion_id] = {}
    dict_merge(salt_data['__pillar__'], node_dict[minion_id].get('pillars', {}))
    (expanded_classes, classes_list, states_list) = expand_classes_in_order(node_dict[minion_id], salt_data, [], {}, [])
    pillars_dict = {}
    for exp_dict in expanded_classes:
        if 'pillars' in exp_dict:
            dict_merge(pillars_dict, exp_dict)
    return (expanded_classes, pillars_dict, classes_list, states_list)

def get_pillars(minion_id, salt_data):
    if False:
        while True:
            i = 10
    (expanded_classes, pillars_dict, classes_list, states_list) = expanded_dict_from_minion(minion_id, salt_data)
    environment = get_env_from_dict(expanded_classes)
    if 'pillars' in pillars_dict:
        pillars_dict_expanded = expand_variables(pillars_dict['pillars'], {}, [])
    else:
        pillars_dict_expanded = expand_variables({}, {}, [])
    pillars_dict = {}
    pillars_dict['__saltclass__'] = {}
    pillars_dict['__saltclass__']['states'] = states_list
    pillars_dict['__saltclass__']['classes'] = classes_list
    pillars_dict['__saltclass__']['environment'] = environment
    pillars_dict['__saltclass__']['nodename'] = minion_id
    pillars_dict.update(pillars_dict_expanded)
    return pillars_dict

def get_tops(minion_id, salt_data):
    if False:
        i = 10
        return i + 15
    (expanded_classes, pillars_dict, classes_list, states_list) = expanded_dict_from_minion(minion_id, salt_data)
    environment = get_env_from_dict(expanded_classes)
    tops_dict = {}
    tops_dict[environment] = states_list
    return tops_dict