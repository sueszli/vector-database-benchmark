import argparse
import ast
import importlib
import inspect
import logging
import os
import pkgutil
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Any, Union
try:
    import msrestazure
except:
    import subprocess
    subprocess.call(sys.executable + ' -m pip install msrestazure', shell=True)
try:
    from jinja2 import Template, FileSystemLoader, Environment
except:
    import subprocess
    subprocess.call(sys.executable + ' -m pip install jinja2', shell=True)
    from jinja2 import Template, FileSystemLoader, Environment
try:
    import azure.common
except:
    sys.path.append(str((Path(__file__).parents[1] / 'sdk' / 'core' / 'azure-common').resolve()))
    import azure.common
import pkg_resources
pkg_resources.declare_namespace('azure')
_LOGGER = logging.getLogger(__name__)

def parse_input(input_parameter):
    if False:
        i = 10
        return i + 15
    'From a syntax like package_name#submodule, build a package name\n    and complete module name.\n    '
    split_package_name = input_parameter.split('#')
    package_name = split_package_name[0]
    module_name = package_name.replace('-', '.')
    if len(split_package_name) >= 2:
        module_name = '.'.join([module_name, split_package_name[1]])
    return (package_name, module_name)

def resolve_package_directory(package_name, sdk_root=None):
    if False:
        print('Hello World!')
    packages = [p.parent for p in list(sdk_root.glob('{}/setup.py'.format(package_name))) + list(sdk_root.glob('sdk/*/{}/setup.py'.format(package_name)))]
    if len(packages) > 1:
        print('There should only be a single package matched in either repository structure. The following were found: {}'.format(packages))
        sys.exit(1)
    return str(packages[0].relative_to(sdk_root))

def get_versioned_modules(package_name: str, module_name: str, sdk_root: Path=None) -> List[Tuple[str, Any]]:
    if False:
        print('Hello World!')
    'Get (label, submodule) where label starts with "v20" and submodule is the corresponding imported module.\n    '
    if not sdk_root:
        sdk_root = Path(__file__).parents[1]
    path_to_package = resolve_package_directory(package_name, sdk_root)
    azure.__path__.append(str((sdk_root / path_to_package / 'azure').resolve()))
    module_to_generate = importlib.import_module(module_name)
    return {label: importlib.import_module('.' + label, module_to_generate.__name__) for (_, label, ispkg) in pkgutil.iter_modules(module_to_generate.__path__) if label.startswith('v20') and ispkg}

class ApiVersionExtractor(ast.NodeVisitor):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.api_version = None
        super(ApiVersionExtractor, self).__init__(*args, **kwargs)

    def visit_Assign(self, node):
        if False:
            for i in range(10):
                print('nop')
        try:
            if node.targets[0].id == 'api_version':
                self.api_version = node.value.s
        except Exception:
            pass

def extract_api_version_from_code(function):
    if False:
        i = 10
        return i + 15
    'Will extract from __code__ the API version. Should be use if you use this is an operation group with no constant api_version.\n    '
    try:
        srccode = inspect.getsource(function)
        try:
            ast_tree = ast.parse(srccode)
        except IndentationError:
            ast_tree = ast.parse('with 0:\n' + srccode)
        api_version_visitor = ApiVersionExtractor()
        api_version_visitor.visit(ast_tree)
        return api_version_visitor.api_version
    except Exception:
        raise

def get_client_class_name_from_module(module):
    if False:
        print('Hello World!')
    'Being a module that is an Autorest generation, get the client name.'
    return module.__all__[0]

def build_operation_meta(versioned_modules):
    if False:
        for i in range(10):
            print('nop')
    "Introspect the client:\n\n    version_dict => {\n        'application_gateways': [\n            ('v2018_05_01', 'ApplicationGatewaysOperations')\n        ]\n    }\n    mod_to_api_version => {'v2018_05_01': '2018-05-01'}\n    "
    version_dict = {}
    mod_to_api_version = {}
    for (versionned_label, versionned_mod) in versioned_modules.items():
        extracted_api_versions = set()
        client_doc = versionned_mod.__dict__[get_client_class_name_from_module(versionned_mod)].__doc__
        operations = list(re.finditer(':ivar (?P<attr>[a-z_0-9]+): \\w+ operations\\n\\s+:vartype (?P=attr): .*.operations.(?P<clsname>\\w+)\\n', client_doc))
        for operation in operations:
            (attr, clsname) = operation.groups()
            _LOGGER.debug('Class name: %s', clsname)
            version_dict.setdefault(attr, []).append((versionned_label, clsname))
            extracted_api_version = None
            try:
                extracted_api_version = versionned_mod.operations.__dict__[clsname](None, None, None, None).api_version
                _LOGGER.debug('Found an obvious API version: %s', extracted_api_version)
                if extracted_api_version:
                    extracted_api_versions.add(extracted_api_version)
            except Exception:
                _LOGGER.debug('Should not happen. I guess it mixed operation groups like VMSS Network...')
                for (func_name, function) in versionned_mod.operations.__dict__[clsname].__dict__.items():
                    if not func_name.startswith('__'):
                        _LOGGER.debug('Try to extract API version from: %s', func_name)
                        extracted_api_version = extract_api_version_from_code(function)
                        _LOGGER.debug('Extracted API version: %s', extracted_api_version)
                        if extracted_api_version:
                            extracted_api_versions.add(extracted_api_version)
        if not extracted_api_versions:
            sys.exit('Was not able to extract api_version of {}'.format(versionned_label))
        if len(extracted_api_versions) >= 2:
            final_api_version = None
            _LOGGER.warning('Found too much API version: {} in label {}'.format(extracted_api_versions, versionned_label))
            for candidate_api_version in extracted_api_versions:
                if 'v{}'.format(candidate_api_version.replace('-', '_')) == versionned_label:
                    final_api_version = candidate_api_version
                    _LOGGER.warning('Guessing you want {} based on label {}'.format(final_api_version, versionned_label))
                    break
            else:
                sys.exit('Unble to match {} to label {}'.format(extracted_api_versions, versionned_label))
            extracted_api_versions = {final_api_version}
        mod_to_api_version[versionned_label] = extracted_api_versions.pop()
    return (version_dict, mod_to_api_version)

def build_operation_mixin_meta(versioned_modules):
    if False:
        return 10
    "Introspect the client:\n\n    version_dict => {\n        'check_dns_name_availability': {\n            'doc': 'docstring',\n            'signature': '(self, p1, p2, **operation_config),\n            'call': 'p1, p2',\n            'available_apis': [\n                'v2018_05_01'\n            ]\n        }\n    }\n    "
    mixin_operations = {}
    for (versionned_label, versionned_mod) in sorted(versioned_modules.items()):
        client_name = get_client_class_name_from_module(versionned_mod)
        client_class = versionned_mod.__dict__[client_name]
        operations_mixin = next((c for c in client_class.__mro__ if 'OperationsMixin' in c.__name__), None)
        if not operations_mixin:
            continue
        for (func_name, func) in operations_mixin.__dict__.items():
            if func_name.startswith('_'):
                continue
            signature = inspect.signature(func)
            mixin_operations.setdefault(func_name, {}).setdefault('available_apis', []).append(versionned_label)
            mixin_operations[func_name]['doc'] = func.__doc__
            mixin_operations[func_name]['signature'] = str(signature)
            mixin_operations[func_name]['call'] = ', '.join(list(signature.parameters)[1:-1])
    return mixin_operations

def build_last_rt_list(versioned_operations_dict, mixin_operations, last_api_version, preview_mode):
    if False:
        while True:
            i = 10
    'Build the a mapping RT => API version if RT doesn\'t exist in latest detected API version.\n\n    Example:\n    last_rt_list = {\n       \'check_dns_name_availability\': \'2018-05-01\'\n    }\n\n    There is one subtle scenario if PREVIEW mode is disabled:\n    - RT1 available on 2019-05-01 and 2019-06-01-preview\n    - RT2 available on 2019-06-01-preview\n    - RT3 available on 2019-07-01-preview\n\n    Then, if I put "RT2: 2019-06-01-preview" in the list, this means I have to make\n    "2019-06-01-preview" the default for models loading (otherwise "RT2: 2019-06-01-preview" won\'t work).\n    But this likely breaks RT1 default operations at "2019-05-01", with default models at "2019-06-01-preview"\n    since "models" are shared for the entire set of operations groups (I wished models would be split by operation groups, but meh, that\'s not the case)\n\n    So, until we have a smarter Autorest to deal with that, only preview RTs which do not share models with a stable RT can be added to this map.\n    In this case, RT2 is out, RT3 is in.\n    '

    def there_is_a_rt_that_contains_api_version(rt_dict, api_version):
        if False:
            i = 10
            return i + 15
        'Test in the given api_version is is one of those RT.'
        for rt_api_version in rt_dict.values():
            if api_version in rt_api_version:
                return True
        return False
    last_rt_list = {}
    versioned_dict = {operation_name: [meta[0] for meta in operation_metadata] for (operation_name, operation_metadata) in versioned_operations_dict.items()}
    versioned_dict.update({operation_name: operation_metadata['available_apis'] for (operation_name, operation_metadata) in mixin_operations.items()})
    for (operation, api_versions_list) in versioned_dict.items():
        local_last_api_version = get_floating_latest(api_versions_list, preview_mode)
        if local_last_api_version == last_api_version:
            continue
        if there_is_a_rt_that_contains_api_version(versioned_dict, local_last_api_version) and local_last_api_version > last_api_version:
            continue
        last_rt_list[operation] = local_last_api_version
    return last_rt_list

def get_floating_latest(api_versions_list, preview_mode):
    if False:
        for i in range(10):
            print('nop')
    'Get the floating latest, from a random list of API versions.\n    '
    api_versions_list = list(api_versions_list)
    absolute_latest = sorted(api_versions_list)[-1]
    trimmed_preview = [version for version in api_versions_list if 'preview' not in version]
    if not trimmed_preview:
        return absolute_latest
    if preview_mode:
        return absolute_latest
    return sorted(trimmed_preview)[-1]

def find_module_folder(package_name, module_name):
    if False:
        print('Hello World!')
    sdk_root = Path(__file__).parents[1]
    _LOGGER.debug('SDK root is: %s', sdk_root)
    path_to_package = resolve_package_directory(package_name, sdk_root)
    module_path = sdk_root / Path(path_to_package) / Path(module_name.replace('.', os.sep))
    _LOGGER.debug('Module path is: %s', module_path)
    return module_path

def find_client_file(package_name, module_name):
    if False:
        return 10
    module_path = find_module_folder(package_name, module_name)
    return next(module_path.glob('*_client.py'))

def patch_import(file_path: Union[str, Path]) -> None:
    if False:
        for i in range(10):
            print('nop')
    "If multi-client package, we need to patch import to be\n    from ..version\n    and not\n    from .version\n\n    That should probably means those files should become a template, but since right now\n    it's literally one dot, let's do it the raw way.\n    "
    with open(file_path, 'rb') as read_fd:
        conf_bytes = read_fd.read()
    conf_bytes = conf_bytes.replace(b' .version', b' ..version')
    with open(file_path, 'wb') as write_fd:
        write_fd.write(conf_bytes)

def has_subscription_id(client_class):
    if False:
        print('Hello World!')
    return 'subscription_id' in inspect.signature(client_class).parameters

def main(input_str, default_api=None):
    if False:
        return 10
    preview_mode = default_api and 'preview' in default_api
    is_multi_client_package = '#' in input_str
    (package_name, module_name) = parse_input(input_str)
    versioned_modules = get_versioned_modules(package_name, module_name)
    (versioned_operations_dict, mod_to_api_version) = build_operation_meta(versioned_modules)
    client_folder = find_module_folder(package_name, module_name)
    last_api_version = get_floating_latest(mod_to_api_version.keys(), preview_mode)
    if default_api and (not default_api.startswith('v')):
        last_api_version = [mod_api for (mod_api, real_api) in mod_to_api_version.items() if real_api == default_api][0]
        _LOGGER.info('Default API version will be: %s', last_api_version)
    last_api_path = client_folder / last_api_version
    shutil.rmtree(str(client_folder / 'operations'), ignore_errors=True)
    shutil.rmtree(str(client_folder / 'models'), ignore_errors=True)
    shutil.copy(str(client_folder / last_api_version / '_configuration.py'), str(client_folder / '_configuration.py'))
    shutil.copy(str(client_folder / last_api_version / '__init__.py'), str(client_folder / '__init__.py'))
    if is_multi_client_package:
        _LOGGER.warning('Patching multi-api client basic files')
        patch_import(client_folder / '_configuration.py')
        patch_import(client_folder / '__init__.py')
    versionned_mod = versioned_modules[last_api_version]
    client_name = get_client_class_name_from_module(versionned_mod)
    client_class = versionned_mod.__dict__[client_name]
    mixin_operations = build_operation_mixin_meta(versioned_modules)
    client_file_name = next(last_api_path.glob('*_client.py')).name
    last_rt_list = build_last_rt_list(versioned_operations_dict, mixin_operations, last_api_version, preview_mode)
    conf = {'client_name': client_name, 'has_subscription_id': has_subscription_id(client_class), 'module_name': module_name, 'operations': versioned_operations_dict, 'mixin_operations': mixin_operations, 'mod_to_api_version': mod_to_api_version, 'last_api_version': mod_to_api_version[last_api_version], 'client_doc': client_class.__doc__.split('\n')[0], 'last_rt_list': last_rt_list, 'default_models': sorted({last_api_version} | {versions for (_, versions) in last_rt_list.items()})}
    env = Environment(loader=FileSystemLoader(str(Path(__file__).parents[0] / 'templates')), keep_trailing_newline=True)
    for template_name in env.list_templates():
        if template_name == '_operations_mixin.py' and (not mixin_operations):
            continue
        if template_name == '_multiapi_client.py':
            output_filename = client_file_name
        else:
            output_filename = template_name
        future_filepath = client_folder / output_filename
        template = env.get_template(template_name)
        result = template.render(**conf)
        with future_filepath.open('w') as fd:
            fd.write(result)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-API client generation for Azure SDK for Python')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Verbosity in DEBUG mode')
    parser.add_argument('--default-api-version', dest='default_api', default=None, help='Force default API version, do not detect it. [default: %(default)s]')
    parser.add_argument('package_name', help='The package name.')
    args = parser.parse_args()
    main_logger = logging.getLogger()
    logging.basicConfig()
    main_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args.package_name, default_api=args.default_api)