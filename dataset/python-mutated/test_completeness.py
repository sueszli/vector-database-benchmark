from __future__ import absolute_import, division, print_function
__metaclass__ = type
from awx.main.tests.functional.conftest import _request
from ansible.module_utils.six import string_types
import yaml
import os
import re
import glob
read_only_endpoints_with_modules = ['settings', 'role', 'project_update', 'workflow_approval']
no_module_for_endpoint = ['constructed_inventory']
no_endpoint_for_module = ['import', 'controller_meta', 'export', 'inventory_source_update', 'job_launch', 'job_wait', 'job_list', 'license', 'ping', 'receive', 'send', 'workflow_launch', 'workflow_node_wait', 'job_cancel', 'workflow_template', 'ad_hoc_command_wait', 'ad_hoc_command_cancel', 'subscriptions']
extra_endpoints = {'bulk_job_launch': '/api/v2/bulk/job_launch/', 'bulk_host_create': '/api/v2/bulk/host_create/'}
ignore_parameters = ['state', 'new_name', 'update_secrets', 'copy_from']
no_api_parameter_ok = {'project': ['wait', 'interval', 'update_project'], 'token': ['existing_token', 'existing_token_id'], 'job_template': ['survey_spec', 'organization'], 'inventory_source': ['organization'], 'workflow_job_template_node': ['organization', 'approval_node', 'lookup_organization'], 'workflow_job_template': ['survey_spec', 'destroy_current_nodes'], 'schedule': ['organization'], 'ad_hoc_command': ['interval', 'timeout', 'wait'], 'group': ['preserve_existing_children', 'preserve_existing_hosts'], 'user': ['new_username', 'organization'], 'workflow_approval': ['action', 'interval', 'timeout', 'workflow_job_id'], 'bulk_job_launch': ['interval', 'wait']}
needs_development = ['inventory_script', 'instance']
needs_param_development = {'host': ['instance_id'], 'workflow_approval': ['description', 'execution_environment']}
return_value = 0
read_only_endpoint = []

def cause_error(msg):
    if False:
        while True:
            i = 10
    global return_value
    return_value = 255
    return msg

def test_meta_runtime():
    if False:
        print('Hello World!')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    meta_filename = 'meta/runtime.yml'
    module_dir = 'plugins/modules'
    print('\nMeta check:')
    with open('{0}/{1}'.format(base_dir, meta_filename), 'r') as f:
        meta_data_string = f.read()
    meta_data = yaml.load(meta_data_string, Loader=yaml.Loader)
    needs_grouping = []
    for file_name in glob.glob('{0}/{1}/*'.format(base_dir, module_dir)):
        if not os.path.isfile(file_name) or os.path.islink(file_name):
            continue
        with open(file_name, 'r') as f:
            if 'extends_documentation_fragment: awx.awx.auth' in f.read():
                needs_grouping.append(os.path.splitext(os.path.basename(file_name))[0])
    needs_to_be_removed = list(set(meta_data['action_groups']['controller']) - set(needs_grouping))
    needs_to_be_added = list(set(needs_grouping) - set(meta_data['action_groups']['controller']))
    needs_to_be_removed.sort()
    needs_to_be_added.sort()
    group = 'action-groups.controller'
    if needs_to_be_removed:
        print(cause_error('The following items should be removed from the {0} {1}:\n    {2}'.format(meta_filename, group, '\n    '.join(needs_to_be_removed))))
    if needs_to_be_added:
        print(cause_error('The following items should be added to the {0} {1}:\n    {2}'.format(meta_filename, group, '\n    '.join(needs_to_be_added))))

def determine_state(module_id, endpoint, module, parameter, api_option, module_option):
    if False:
        return 10
    if module_id in needs_development and module == 'N/A':
        return 'Failed (non-blocking), module needs development'
    if module_id in read_only_endpoint:
        if module == 'N/A':
            return 'OK, this endpoint is read-only and should not have a module'
        elif module_id in read_only_endpoints_with_modules:
            return 'OK, module params can not be checked to read-only'
        else:
            return cause_error('Failed, read-only endpoint should not have an associated module')
    if module_id in no_module_for_endpoint and module == 'N/A':
        return 'OK, this endpoint should not have a module'
    if module_id in no_endpoint_for_module and endpoint == 'N/A':
        return 'OK, this module does not require an endpoint'
    if module == 'N/A':
        return cause_error('Failed, missing module')
    if endpoint == 'N/A':
        return cause_error('Failed, why does this module have no endpoint')
    if parameter in ignore_parameters:
        return 'OK, globally ignored parameter'
    if (api_option is None) ^ (module_option is None):
        if api_option is None and parameter in no_api_parameter_ok.get(module, {}):
            return 'OK, no api parameter is ok'
        if module_option is None and parameter in needs_param_development.get(module_id, {}):
            return 'Failed (non-blocking), parameter needs development'
        if module_option and module_option.get('description'):
            description = ''
            if isinstance(module_option.get('description'), string_types):
                description = module_option.get('description')
            else:
                description = ' '.join(module_option.get('description'))
            if 'deprecated' in description.lower():
                if api_option is None:
                    return 'OK, deprecated module option'
                else:
                    return cause_error('Failed, module marks option as deprecated but option still exists in API')
        if not api_option and module_option and (module_option.get('type', 'str') == 'list'):
            return 'OK, Field appears to be relation'
        return cause_error('Failed, option mismatch')
    return 'OK'

def test_completeness(collection_import, request, admin_user, job_template, execution_environment):
    if False:
        i = 10
        return i + 15
    option_comparison = {}
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    module_directory = os.path.join(base_folder, 'plugins', 'modules')
    for (root, dirs, files) in os.walk(module_directory):
        if root == module_directory:
            for filename in files:
                if os.path.islink(os.path.join(root, filename)):
                    continue
                if re.match('^[a-z].*.py$', filename):
                    module_name = filename[:-3]
                    option_comparison[module_name] = {'endpoint': 'N/A', 'api_options': {}, 'module_options': {}, 'module_name': module_name}
                    resource_module = collection_import('plugins.modules.{0}'.format(module_name))
                    option_comparison[module_name]['module_options'] = yaml.load(resource_module.DOCUMENTATION, Loader=yaml.SafeLoader)['options']
    endpoint_response = _request('get')(url='/api/v2/', user=admin_user, expect=None)
    for (key, val) in extra_endpoints.items():
        endpoint_response.data[key] = val
    for endpoint in endpoint_response.data.keys():
        singular_endpoint = '{0}'.format(endpoint)
        if singular_endpoint.endswith('ies'):
            singular_endpoint = singular_endpoint[:-3]
        if singular_endpoint != 'settings' and singular_endpoint.endswith('s'):
            singular_endpoint = singular_endpoint[:-1]
        module_name = '{0}'.format(singular_endpoint)
        endpoint_url = endpoint_response.data.get(endpoint)
        if module_name not in option_comparison:
            option_comparison[module_name] = {}
            option_comparison[module_name]['module_name'] = 'N/A'
            option_comparison[module_name]['module_options'] = {}
        option_comparison[module_name]['endpoint'] = endpoint_url
        option_comparison[module_name]['api_options'] = {}
        options_response = _request('options')(url=endpoint_url, user=admin_user, expect=None)
        if 'POST' in options_response.data.get('actions', {}):
            option_comparison[module_name]['api_options'] = options_response.data.get('actions').get('POST')
        else:
            read_only_endpoint.append(module_name)
    longest_module_name = 0
    longest_option_name = 0
    longest_endpoint = 0
    for (module, module_value) in option_comparison.items():
        if len(module_value['module_name']) > longest_module_name:
            longest_module_name = len(module_value['module_name'])
        if len(module_value['endpoint']) > longest_endpoint:
            longest_endpoint = len(module_value['endpoint'])
        for option in (module_value['api_options'], module_value['module_options']):
            if len(option) > longest_option_name:
                longest_option_name = len(option)
    print(''.join(['End Point', ' ' * (longest_endpoint - len('End Point')), ' | Module Name', ' ' * (longest_module_name - len('Module Name')), ' | Option', ' ' * (longest_option_name - len('Option')), ' | API | Module | State']))
    print('-|-'.join(['-' * longest_endpoint, '-' * longest_module_name, '-' * longest_option_name, '---', '------', '---------------------------------------------']))
    for module in sorted(option_comparison):
        module_data = option_comparison[module]
        all_param_names = list(set(module_data['api_options']) | set(module_data['module_options']))
        for parameter in sorted(all_param_names):
            print(''.join([module_data['endpoint'], ' ' * (longest_endpoint - len(module_data['endpoint'])), ' | ', module_data['module_name'], ' ' * (longest_module_name - len(module_data['module_name'])), ' | ', parameter, ' ' * (longest_option_name - len(parameter)), ' | ', ' X ' if parameter in module_data['api_options'] else '   ', ' | ', '  X   ' if parameter in module_data['module_options'] else '      ', ' | ', determine_state(module, module_data['endpoint'], module_data['module_name'], parameter, module_data['api_options'][parameter] if parameter in module_data['api_options'] else None, module_data['module_options'][parameter] if parameter in module_data['module_options'] else None)]))
        if len(all_param_names) == 0:
            print(''.join([module_data['endpoint'], ' ' * (longest_endpoint - len(module_data['endpoint'])), ' | ', module_data['module_name'], ' ' * (longest_module_name - len(module_data['module_name'])), ' | ', 'N/A', ' ' * (longest_option_name - len('N/A')), ' | ', '   ', ' | ', '      ', ' | ', determine_state(module, module_data['endpoint'], module_data['module_name'], 'N/A', None, None)]))
    test_meta_runtime()
    if return_value != 0:
        raise Exception('One or more failures caused issues')