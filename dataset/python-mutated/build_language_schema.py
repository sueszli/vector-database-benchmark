import inspect
import json
import argparse
import os
import glob
import re
import voluptuous as vol
import esphome.schema_extractors as ejs
ejs.EnableSchemaExtraction = True
S_CONFIG_VAR = 'config_var'
S_CONFIG_VARS = 'config_vars'
S_CONFIG_SCHEMA = 'CONFIG_SCHEMA'
S_COMPONENT = 'component'
S_COMPONENTS = 'components'
S_PLATFORMS = 'platforms'
S_SCHEMA = 'schema'
S_SCHEMAS = 'schemas'
S_EXTENDS = 'extends'
S_TYPE = 'type'
S_NAME = 'name'
parser = argparse.ArgumentParser()
parser.add_argument('--output-path', default='.', help='Output path', type=os.path.abspath)
args = parser.parse_args()
DUMP_RAW = False
DUMP_UNKNOWN = False
DUMP_PATH = False
JSON_DUMP_PRETTY = True
components = {}
schema_core = {}
output = {'core': schema_core}
schema_full = {'components': output}
known_schemas = {}
solve_registry = []

def get_component_names():
    if False:
        for i in range(10):
            print('nop')
    from esphome.loader import CORE_COMPONENTS_PATH
    component_names = ['esphome', 'sensor', 'esp32', 'esp8266']
    for d in os.listdir(CORE_COMPONENTS_PATH):
        if not d.startswith('__') and os.path.isdir(os.path.join(CORE_COMPONENTS_PATH, d)):
            if d not in component_names:
                component_names.append(d)
    return component_names

def load_components():
    if False:
        print('Hello World!')
    from esphome.config import get_component
    for domain in get_component_names():
        components[domain] = get_component(domain)
from esphome.const import CONF_TYPE, KEY_CORE
from esphome.core import CORE
CORE.data[KEY_CORE] = {}
load_components()
import esphome.core as esphome_core
import esphome.config_validation as cv
from esphome import automation
from esphome import pins
from esphome.components import remote_base
from esphome.loader import get_platform, CORE_COMPONENTS_PATH
from esphome.helpers import write_file_if_changed
from esphome.util import Registry

def write_file(name, obj):
    if False:
        while True:
            i = 10
    full_path = os.path.join(args.output_path, name + '.json')
    if JSON_DUMP_PRETTY:
        json_str = json.dumps(obj, indent=2)
    else:
        json_str = json.dumps(obj, separators=(',', ':'))
    write_file_if_changed(full_path, json_str)
    print(f'Wrote {full_path}')

def delete_extra_files(keep_names):
    if False:
        while True:
            i = 10
    for d in os.listdir(args.output_path):
        if d.endswith('.json') and (not d[:-5] in keep_names):
            os.remove(os.path.join(args.output_path, d))
            print(f'Deleted {d}')

def register_module_schemas(key, module, manifest=None):
    if False:
        for i in range(10):
            print('nop')
    for (name, schema) in module_schemas(module):
        register_known_schema(key, name, schema)
    if manifest:
        if manifest.multi_conf and S_CONFIG_SCHEMA in output[key][S_SCHEMAS]:
            output[key][S_SCHEMAS][S_CONFIG_SCHEMA]['is_list'] = True

def register_known_schema(module, name, schema):
    if False:
        i = 10
        return i + 15
    if module not in output:
        output[module] = {S_SCHEMAS: {}}
    config = convert_config(schema, f'{module}/{name}')
    if S_TYPE not in config:
        print(f'Config var without type: {module}.{name}')
    output[module][S_SCHEMAS][name] = config
    repr_schema = repr(schema)
    if repr_schema in known_schemas:
        schema_info = known_schemas[repr_schema]
        schema_info.append((schema, f'{module}.{name}'))
    else:
        known_schemas[repr_schema] = [(schema, f'{module}.{name}')]

def module_schemas(module):
    if False:
        while True:
            i = 10
    try:
        module_str = inspect.getsource(module)
    except TypeError:
        module_str = ''
    except OSError:
        module_str = ''
    schemas = {}
    for m_attr_name in dir(module):
        m_attr_obj = getattr(module, m_attr_name)
        if is_convertible_schema(m_attr_obj):
            schemas[module_str.find(m_attr_name)] = [m_attr_name, m_attr_obj]
    for pos in sorted(schemas.keys()):
        yield schemas[pos]
found_registries = {}
pin_validators = {}

def add_pin_validators():
    if False:
        return 10
    for m_attr_name in dir(pins):
        if 'gpio' in m_attr_name:
            s = pin_validators[repr(getattr(pins, m_attr_name))] = {}
            if 'schema' in m_attr_name:
                s['schema'] = True
            if 'internal' in m_attr_name:
                s['internal'] = True
            if 'input' in m_attr_name:
                s['modes'] = ['input']
            elif 'output' in m_attr_name:
                s['modes'] = ['output']
            else:
                s['modes'] = []
            if 'pullup' in m_attr_name:
                s['modes'].append('pullup')
    from esphome.components.adc import sensor as adc_sensor
    pin_validators[repr(adc_sensor.validate_adc_pin)] = {'internal': True, 'modes': ['input']}

def add_module_registries(domain, module):
    if False:
        print('Hello World!')
    for attr_name in dir(module):
        attr_obj = getattr(module, attr_name)
        if isinstance(attr_obj, Registry):
            if attr_obj == automation.ACTION_REGISTRY:
                reg_type = 'action'
                reg_domain = 'core'
                found_registries[repr(attr_obj)] = reg_type
            elif attr_obj == automation.CONDITION_REGISTRY:
                reg_type = 'condition'
                reg_domain = 'core'
                found_registries[repr(attr_obj)] = reg_type
            else:
                reg_domain = domain
                reg_type = attr_name.partition('_')[0].lower()
                found_registries[repr(attr_obj)] = f'{domain}.{reg_type}'
            for name in attr_obj.keys():
                if '.' not in name:
                    reg_entry_name = name
                else:
                    parts = name.split('.')
                    if len(parts) == 2:
                        reg_domain = parts[0]
                        reg_entry_name = parts[1]
                    else:
                        reg_domain = '.'.join([parts[1], parts[0]])
                        reg_entry_name = parts[2]
                if reg_domain not in output:
                    output[reg_domain] = {}
                if reg_type not in output[reg_domain]:
                    output[reg_domain][reg_type] = {}
                output[reg_domain][reg_type][reg_entry_name] = convert_config(attr_obj[name].schema, f'{reg_domain}/{reg_type}/{reg_entry_name}')

def do_pins():
    if False:
        print('Hello World!')
    pins_providers = schema_core['pins'] = []
    for pin_registry in pins.PIN_SCHEMA_REGISTRY:
        s = convert_config(pins.PIN_SCHEMA_REGISTRY[pin_registry][1], f'pins/{pin_registry}')
        if pin_registry not in output:
            output[pin_registry] = {}
        output[pin_registry]['pin'] = s
        pins_providers.append(pin_registry)

def setBoards(obj, boards):
    if False:
        while True:
            i = 10
    obj[S_TYPE] = 'enum'
    obj['values'] = {}
    for (k, v) in boards.items():
        obj['values'][k] = {'docs': v['name']}

def do_esp32():
    if False:
        return 10
    import esphome.components.esp32.boards as esp32_boards
    setBoards(output['esp32']['schemas']['CONFIG_SCHEMA']['schema']['config_vars']['board'], esp32_boards.BOARDS)

def do_esp8266():
    if False:
        for i in range(10):
            print('nop')
    import esphome.components.esp8266.boards as esp8266_boards
    setBoards(output['esp8266']['schemas']['CONFIG_SCHEMA']['schema']['config_vars']['board'], esp8266_boards.BOARDS)

def fix_remote_receiver():
    if False:
        for i in range(10):
            print('nop')
    if 'remote_receiver.binary_sensor' not in output:
        return
    remote_receiver_schema = output['remote_receiver.binary_sensor']['schemas']
    remote_receiver_schema['CONFIG_SCHEMA'] = {'type': 'schema', 'schema': {'extends': ['binary_sensor.BINARY_SENSOR_SCHEMA', 'core.COMPONENT_SCHEMA'], 'config_vars': output['remote_base'].pop('binary')}}
    remote_receiver_schema['CONFIG_SCHEMA']['schema']['config_vars']['receiver_id'] = {'key': 'GeneratedID', 'use_id_type': 'remote_base::RemoteReceiverBase', 'type': 'use_id'}

def fix_script():
    if False:
        while True:
            i = 10
    if 'script' not in output:
        return
    output['script'][S_SCHEMAS][S_CONFIG_SCHEMA][S_TYPE] = S_SCHEMA
    config_schema = output['script'][S_SCHEMAS][S_CONFIG_SCHEMA]
    config_schema[S_SCHEMA][S_CONFIG_VARS]['id']['id_type'] = {'class': 'script::Script'}
    config_schema['is_list'] = True

def fix_font():
    if False:
        i = 10
        return i + 15
    if 'font' not in output:
        return
    output['font'][S_SCHEMAS]['FILE_SCHEMA'] = output['font'][S_SCHEMAS].pop('TYPED_FILE_SCHEMA')

def fix_menu():
    if False:
        print('Hello World!')
    if 'display_menu_base' not in output:
        return
    schemas = output['display_menu_base'][S_SCHEMAS]
    schemas['MENU_TYPES'] = {S_TYPE: S_SCHEMA, S_SCHEMA: {S_CONFIG_VARS: {'items': schemas['DISPLAY_MENU_BASE_SCHEMA'][S_SCHEMA][S_CONFIG_VARS]['items']}}}
    schemas['DISPLAY_MENU_BASE_SCHEMA'][S_SCHEMA][S_CONFIG_VARS].pop('items')
    schemas['DISPLAY_MENU_BASE_SCHEMA'][S_SCHEMA][S_EXTENDS].append('display_menu_base.MENU_TYPES')
    menu = schemas['MENU_TYPES'][S_SCHEMA][S_CONFIG_VARS]['items']['types']['menu']
    menu[S_CONFIG_VARS].pop('items')
    menu[S_EXTENDS] = ['display_menu_base.MENU_TYPES']

def get_logger_tags():
    if False:
        for i in range(10):
            print('nop')
    pattern = re.compile('^static const char \\*const TAG = "(\\w.*)";', re.MULTILINE)
    tags = ['app', 'component', 'entity_base', 'scheduler', 'api.service']
    for x in os.walk(CORE_COMPONENTS_PATH):
        for y in glob.glob(os.path.join(x[0], '*.cpp')):
            with open(y, encoding='utf-8') as file:
                data = file.read()
                match = pattern.search(data)
                if match:
                    tags.append(match.group(1))
    return tags

def add_logger_tags():
    if False:
        return 10
    if 'logger' not in output or 'schemas' not in output['logger']:
        return
    tags = get_logger_tags()
    logs = output['logger']['schemas']['CONFIG_SCHEMA']['schema']['config_vars']['logs']['schema']['config_vars']
    for t in tags:
        logs[t] = logs['string'].copy()
    logs.pop('string')

def add_referenced_recursive(referenced_schemas, config_var, path, eat_schema=False):
    if False:
        return 10
    assert S_CONFIG_VARS not in config_var and S_EXTENDS not in config_var
    if config_var.get(S_TYPE) in ['schema', 'trigger', 'maybe'] and S_SCHEMA in config_var:
        schema = config_var[S_SCHEMA]
        for (k, v) in schema.get(S_CONFIG_VARS, {}).items():
            if eat_schema:
                new_path = path + [S_CONFIG_VARS, k]
            else:
                new_path = path + ['schema', S_CONFIG_VARS, k]
            add_referenced_recursive(referenced_schemas, v, new_path)
        for k in schema.get(S_EXTENDS, []):
            if k not in referenced_schemas:
                referenced_schemas[k] = [path]
            elif path not in referenced_schemas[k]:
                referenced_schemas[k].append(path)
            s1 = get_str_path_schema(k)
            p = k.split('.')
            if len(p) == 3 and path[0] == f'{p[0]}.{p[1]}':
                add_referenced_recursive(referenced_schemas, s1, [path[0], 'schemas', p[2]])
            else:
                add_referenced_recursive(referenced_schemas, s1, [p[0], 'schemas', p[1]])
    elif config_var.get(S_TYPE) == 'typed':
        for (tk, tv) in config_var.get('types').items():
            add_referenced_recursive(referenced_schemas, {S_TYPE: S_SCHEMA, S_SCHEMA: tv}, path + ['types', tk], eat_schema=True)

def get_str_path_schema(strPath):
    if False:
        while True:
            i = 10
    parts = strPath.split('.')
    if len(parts) > 2:
        parts[0] += '.' + parts[1]
        parts[1] = parts[2]
    s1 = output.get(parts[0], {}).get(S_SCHEMAS, {}).get(parts[1], {})
    return s1

def pop_str_path_schema(strPath):
    if False:
        while True:
            i = 10
    parts = strPath.split('.')
    if len(parts) > 2:
        parts[0] += '.' + parts[1]
        parts[1] = parts[2]
    output.get(parts[0], {}).get(S_SCHEMAS, {}).pop(parts[1])

def get_arr_path_schema(path):
    if False:
        print('Hello World!')
    s = output
    for x in path:
        s = s[x]
    return s

def merge(source, destination):
    if False:
        return 10
    "\n    run me with nosetests --with-doctest file.py\n\n    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }\n    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }\n    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }\n    True\n    "
    for (key, value) in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value
    return destination

def is_platform_schema(schema_name):
    if False:
        for i in range(10):
            print('nop')
    component = schema_name.split('.')[0]
    return component in components and components[component].is_platform_component

def shrink():
    if False:
        return 10
    'Shrink the extending schemas which has just an end type, e.g. at this point\n    ota / port is type schema with extended pointing to core.port, this should instead be\n    type number. core.port is number\n\n    This also fixes enums, as they are another schema and they are instead put in the same cv\n    '
    pass_again = True
    while pass_again:
        pass_again = False
        referenced_schemas = {}
        for (k, v) in output.items():
            for (kv, vv) in v.items():
                if kv != 'pin' and isinstance(vv, dict):
                    for (kvv, vvv) in vv.items():
                        add_referenced_recursive(referenced_schemas, vvv, [k, kv, kvv])
        for (x, paths) in referenced_schemas.items():
            if len(paths) == 1 and (not is_platform_schema(x)):
                key_s = get_str_path_schema(x)
                arr_s = get_arr_path_schema(paths[0])
                pass_again = True
                if S_SCHEMA in arr_s:
                    if S_EXTENDS in arr_s[S_SCHEMA]:
                        arr_s[S_SCHEMA].pop(S_EXTENDS)
                    else:
                        print('expected extends here!' + x)
                    arr_s = merge(key_s, arr_s)
                    if arr_s[S_TYPE] in ['enum', 'typed']:
                        arr_s.pop(S_SCHEMA)
                else:
                    arr_s.pop(S_EXTENDS)
                    arr_s |= key_s[S_SCHEMA]
                    print(x)
    for (x, paths) in referenced_schemas.items():
        key_s = get_str_path_schema(x)
        if key_s and key_s[S_TYPE] in ['enum', 'registry', 'integer', 'string']:
            if key_s[S_TYPE] == 'registry':
                print('Spreading registry: ' + x)
            for target in paths:
                target_s = get_arr_path_schema(target)
                assert target_s[S_SCHEMA][S_EXTENDS] == [x]
                target_s.pop(S_SCHEMA)
                target_s |= key_s
                if key_s[S_TYPE] in ['integer', 'string']:
                    target_s['data_type'] = x.split('.')[1]
            pop_str_path_schema(x)
        elif not key_s:
            for target in paths:
                target_s = get_arr_path_schema(target)
                if S_SCHEMA not in target_s:
                    target_s[S_EXTENDS].remove(x)
                    continue
                assert target_s[S_SCHEMA][S_EXTENDS] == [x]
                target_s.pop(S_SCHEMA)
                target_s.pop(S_TYPE)
                target_s['data_type'] = x.split('.')[1]
            pop_str_path_schema(x)
    for (domain, domain_schemas) in output.items():
        for schema_name in list(domain_schemas.get(S_SCHEMAS, {}).keys()):
            s = f'{domain}.{schema_name}'
            if not s.endswith('.' + S_CONFIG_SCHEMA) and s not in referenced_schemas.keys() and (not is_platform_schema(s)):
                print(f'Removing {s}')
                output[domain][S_SCHEMAS].pop(schema_name)

def build_schema():
    if False:
        i = 10
        return i + 15
    print('Building schema')
    if len(ejs.extended_schemas) == 0:
        raise Exception('no data collected. Did you globally import an ESPHome component?')
    schema_core[S_SCHEMAS] = {}
    register_module_schemas('core', cv)
    platforms = {}
    schema_core[S_PLATFORMS] = platforms
    core_components = {}
    schema_core[S_COMPONENTS] = core_components
    add_pin_validators()
    for (domain, manifest) in components.items():
        if manifest.is_platform_component:
            output[domain] = {S_COMPONENTS: {}, S_SCHEMAS: {}}
            platforms[domain] = {}
        elif manifest.config_schema is not None:
            output[domain] = {S_SCHEMAS: {S_CONFIG_SCHEMA: {}}}
    for domain in platforms:
        c = components[domain]
        register_module_schemas(domain, c.module)
    for (domain, manifest) in components.items():
        if domain not in platforms:
            if manifest.config_schema is not None:
                core_components[domain] = {}
                if len(manifest.dependencies) > 0:
                    core_components[domain]['dependencies'] = manifest.dependencies
            register_module_schemas(domain, manifest.module, manifest)
        for platform in platforms:
            platform_manifest = get_platform(domain=platform, platform=domain)
            if platform_manifest is not None:
                output[platform][S_COMPONENTS][domain] = {}
                if len(platform_manifest.dependencies) > 0:
                    output[platform][S_COMPONENTS][domain]['dependencies'] = platform_manifest.dependencies
                register_module_schemas(f'{domain}.{platform}', platform_manifest.module, platform_manifest)
    add_module_registries('core', automation)
    for (domain, manifest) in components.items():
        add_module_registries(domain, manifest.module)
    add_module_registries('remote_base', remote_base)
    for reg_config_var in solve_registry:
        (registry, config_var) = reg_config_var
        config_var[S_TYPE] = 'registry'
        config_var['registry'] = found_registries[repr(registry)]
    do_pins()
    do_esp8266()
    do_esp32()
    fix_remote_receiver()
    fix_script()
    fix_font()
    add_logger_tags()
    shrink()
    fix_menu()
    data = {}
    for (component, component_schemas) in output.items():
        if '.' in component:
            key = component.partition('.')[0]
            if key not in data:
                data[key] = {}
            data[key][component] = component_schemas
        else:
            if component not in data:
                data[component] = {}
            data[component] |= {component: component_schemas}
    data['esphome']['core'] = data.pop('core')['core']
    for (c, s) in data.items():
        write_file(c, s)
    delete_extra_files(data.keys())

def is_convertible_schema(schema):
    if False:
        while True:
            i = 10
    if schema is None:
        return False
    if isinstance(schema, (cv.Schema, cv.All, cv.Any)):
        return True
    if repr(schema) in ejs.hidden_schemas:
        return True
    if repr(schema) in ejs.typed_schemas:
        return True
    if repr(schema) in ejs.list_schemas:
        return True
    if repr(schema) in ejs.registry_schemas:
        return True
    if isinstance(schema, dict):
        for k in schema.keys():
            if isinstance(k, (cv.Required, cv.Optional)):
                return True
    return False

def convert_config(schema, path):
    if False:
        print('Hello World!')
    converted = {}
    convert(schema, converted, path)
    return converted

def convert(schema, config_var, path):
    if False:
        return 10
    'config_var can be a config_var or a schema: both are dicts\n    config_var has a S_TYPE property, if this is S_SCHEMA, then it has a S_SCHEMA property\n    schema does not have a type property, schema can have optionally both S_CONFIG_VARS and S_EXTENDS\n    '
    repr_schema = repr(schema)
    if path.startswith('ads1115.sensor') and path.endswith('gain'):
        print(path)
    if repr_schema in known_schemas:
        schema_info = known_schemas[repr_schema]
        for (schema_instance, name) in schema_info:
            if schema_instance is schema:
                assert S_CONFIG_VARS not in config_var
                assert S_EXTENDS not in config_var
                if not S_TYPE in config_var:
                    config_var[S_TYPE] = S_SCHEMA
                if S_SCHEMA not in config_var:
                    config_var[S_SCHEMA] = {}
                if S_EXTENDS not in config_var[S_SCHEMA]:
                    config_var[S_SCHEMA][S_EXTENDS] = [name]
                elif name not in config_var[S_SCHEMA][S_EXTENDS]:
                    config_var[S_SCHEMA][S_EXTENDS].append(name)
                return
    if repr_schema in ejs.extended_schemas:
        extended = ejs.extended_schemas.get(repr_schema)
        if repr_schema == repr(extended[1]):
            assert path.startswith('midea_ac/')
            return
        assert len(extended) == 2
        convert(extended[0], config_var, path + '/extL')
        convert(extended[1], config_var, path + '/extR')
        return
    if isinstance(schema, cv.All):
        i = 0
        for inner in schema.validators:
            i = i + 1
            convert(inner, config_var, path + f'/val {i}')
        return
    if hasattr(schema, 'validators'):
        i = 0
        for inner in schema.validators:
            i = i + 1
            convert(inner, config_var, path + f'/val {i}')
    if isinstance(schema, cv.Schema):
        convert(schema.schema, config_var, path + '/all')
        return
    if isinstance(schema, dict):
        convert_keys(config_var, schema, path)
        return
    if repr_schema in ejs.list_schemas:
        config_var['is_list'] = True
        items_schema = ejs.list_schemas[repr_schema][0]
        convert(items_schema, config_var, path + '/list')
        return
    if DUMP_RAW:
        config_var['raw'] = repr_schema
    if schema == cv.boolean:
        config_var[S_TYPE] = 'boolean'
    elif schema == automation.validate_potentially_and_condition:
        config_var[S_TYPE] = 'registry'
        config_var['registry'] = 'condition'
    elif schema == cv.int_ or schema == cv.int_range:
        config_var[S_TYPE] = 'integer'
    elif schema == cv.string or schema == cv.string_strict or schema == cv.valid_name:
        config_var[S_TYPE] = 'string'
    elif isinstance(schema, vol.Schema):
        config_var[S_TYPE] = 'schema'
        config_var['schema'] = convert_config(schema.schema, path + '/s')['schema']
    elif repr_schema in pin_validators:
        config_var |= pin_validators[repr_schema]
        config_var[S_TYPE] = 'pin'
    elif repr_schema in ejs.hidden_schemas:
        schema_type = ejs.hidden_schemas[repr_schema]
        data = schema(ejs.SCHEMA_EXTRACT)
        if schema_type == 'one_of':
            config_var[S_TYPE] = 'enum'
            config_var['values'] = dict.fromkeys(list(data))
        elif schema_type == 'enum':
            config_var[S_TYPE] = 'enum'
            config_var['values'] = dict.fromkeys(list(data.keys()))
        elif schema_type == 'maybe':
            config_var[S_TYPE] = S_SCHEMA
            config_var['maybe'] = data[1]
            config_var['schema'] = convert_config(data[0], path + '/maybe')['schema']
        elif schema_type == 'automation':
            extra_schema = None
            config_var[S_TYPE] = 'trigger'
            if automation.AUTOMATION_SCHEMA == ejs.extended_schemas[repr(data)][0]:
                extra_schema = ejs.extended_schemas[repr(data)][1]
            if extra_schema is not None and len(extra_schema) > 1:
                config = convert_config(extra_schema, path + '/extra')
                if 'schema' in config:
                    automation_schema = config['schema']
                    if not (len(automation_schema['config_vars']) == 1 and 'trigger_id' in automation_schema['config_vars']):
                        automation_schema['config_vars']['then'] = {S_TYPE: 'trigger'}
                        if 'trigger_id' in automation_schema['config_vars']:
                            automation_schema['config_vars'].pop('trigger_id')
                        config_var[S_TYPE] = 'trigger'
                        config_var['schema'] = automation_schema
                        try:
                            schema({'delay': '1s'})
                        except cv.Invalid:
                            config_var['has_required_var'] = True
                else:
                    print('figure out ' + path)
        elif schema_type == 'effects':
            config_var[S_TYPE] = 'registry'
            config_var['registry'] = 'light.effects'
            config_var['filter'] = data[0]
        elif schema_type == 'templatable':
            config_var['templatable'] = True
            convert(data, config_var, path + '/templat')
        elif schema_type == 'triggers':
            convert(data, config_var, path + '/trigger')
        elif schema_type == 'sensor':
            schema = data
            convert(data, config_var, path + '/trigger')
        elif schema_type == 'declare_id':
            parents = data._parents
            config_var['id_type'] = {'class': str(data.base), 'parents': [str(x.base) for x in parents] if isinstance(parents, list) else None}
        elif schema_type == 'use_id':
            if inspect.ismodule(data):
                m_attr_obj = getattr(data, 'CONFIG_SCHEMA')
                use_schema = known_schemas.get(repr(m_attr_obj))
                if use_schema:
                    [output_module, output_name] = use_schema[0][1].split('.')
                    use_id_config = output[output_module][S_SCHEMAS][output_name]
                    config_var['use_id_type'] = use_id_config['schema']['config_vars']['id']['id_type']['class']
                    config_var[S_TYPE] = 'use_id'
                else:
                    print('TODO deferred?')
            elif isinstance(data, str):
                config_var['use_id_type'] = data
            else:
                config_var['use_id_type'] = str(data.base)
                config_var[S_TYPE] = 'use_id'
        else:
            raise Exception('Unknown extracted schema type')
    elif config_var.get('key') == 'GeneratedID':
        if path.startswith('i2c/CONFIG_SCHEMA/') and path.endswith('/id'):
            config_var['id_type'] = {'class': 'i2c::I2CBus', 'parents': ['Component']}
        elif path == 'uart/CONFIG_SCHEMA/val 1/extL/all/id':
            config_var['id_type'] = {'class': 'uart::UARTComponent', 'parents': ['Component']}
        elif path == 'pins/esp32/val 1/id':
            config_var['id_type'] = 'pin'
        else:
            raise Exception('Cannot determine id_type for ' + path)
    elif repr_schema in ejs.registry_schemas:
        solve_registry.append((ejs.registry_schemas[repr_schema], config_var))
    elif repr_schema in ejs.typed_schemas:
        config_var[S_TYPE] = 'typed'
        types = config_var['types'] = {}
        typed_schema = ejs.typed_schemas[repr_schema]
        if len(typed_schema) > 1:
            config_var['typed_key'] = typed_schema[1].get('key', CONF_TYPE)
        for (schema_key, schema_type) in typed_schema[0][0].items():
            config = convert_config(schema_type, path + '/type_' + schema_key)
            types[schema_key] = config['schema']
    elif DUMP_UNKNOWN:
        if S_TYPE not in config_var:
            config_var['unknown'] = repr_schema
    if DUMP_PATH:
        config_var['path'] = path
    if S_TYPE not in config_var:
        pass

def get_overridden_config(key, converted):
    if False:
        while True:
            i = 10
    assert S_CONFIG_VARS not in converted and S_EXTENDS not in converted
    config = converted.get(S_SCHEMA, {})
    return get_overridden_key_inner(key, config, {})

def get_overridden_key_inner(key, config, ret):
    if False:
        for i in range(10):
            print('nop')
    if S_EXTENDS not in config:
        return ret
    for s in config[S_EXTENDS]:
        p = s.partition('.')
        s1 = output.get(p[0], {}).get(S_SCHEMAS, {}).get(p[2], {}).get(S_SCHEMA)
        if s1:
            if key in s1.get(S_CONFIG_VARS, {}):
                for (k, v) in s1.get(S_CONFIG_VARS)[key].items():
                    if k not in ret:
                        ret[k] = v
            get_overridden_key_inner(key, s1, ret)
    return ret

def convert_keys(converted, schema, path):
    if False:
        return 10
    for (k, v) in schema.items():
        if repr(v).startswith('<function invalid'):
            continue
        result = {}
        if isinstance(k, cv.GenerateID):
            result['key'] = 'GeneratedID'
        elif isinstance(k, cv.Required):
            result['key'] = 'Required'
        elif isinstance(k, cv.Optional) or isinstance(k, cv.Inclusive) or isinstance(k, cv.Exclusive):
            result['key'] = 'Optional'
        else:
            converted['key'] = 'String'
            key_string_match = re.search('<function (\\w*) at \\w*>', str(k), re.IGNORECASE)
            if key_string_match:
                converted['key_type'] = key_string_match.group(1)
            else:
                converted['key_type'] = str(k)
        esphome_core.CORE.data = {esphome_core.KEY_CORE: {esphome_core.KEY_TARGET_PLATFORM: 'esp8266'}}
        if hasattr(k, 'default') and str(k.default) != '...':
            default_value = k.default()
            if default_value is not None:
                result['default'] = str(default_value)
        convert(v, result, path + f'/{str(k)}')
        if 'schema' not in converted:
            converted[S_TYPE] = 'schema'
            converted['schema'] = {S_CONFIG_VARS: {}}
        if S_CONFIG_VARS not in converted['schema']:
            converted['schema'][S_CONFIG_VARS] = {}
        for (base_k, base_v) in get_overridden_config(k, converted).items():
            if base_k in result and base_v == result[base_k]:
                result.pop(base_k)
        converted['schema'][S_CONFIG_VARS][str(k)] = result
        if 'key' in converted and converted['key'] == 'String':
            config_vars = converted['schema']['config_vars']
            assert len(config_vars) == 1
            key = list(config_vars.keys())[0]
            assert key.startswith('<')
            config_vars['string'] = config_vars.pop(key)
build_schema()