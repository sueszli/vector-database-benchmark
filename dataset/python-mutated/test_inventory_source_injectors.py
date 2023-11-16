import pytest
from unittest import mock
import os
import json
import re
from collections import namedtuple
from awx.main.tasks.jobs import RunInventoryUpdate
from awx.main.models import InventorySource, Credential, CredentialType, UnifiedJob, ExecutionEnvironment
from awx.main.constants import CLOUD_PROVIDERS, STANDARD_INVENTORY_UPDATE_ENV
from awx.main.tests import data
from awx.main.utils.execution_environments import to_container_path
from django.conf import settings
DATA = os.path.join(os.path.dirname(data.__file__), 'inventory')

def generate_fake_var(element):
    if False:
        for i in range(10):
            print('nop')
    'Given a credential type field element, makes up something acceptable.'
    if element['type'] == 'string':
        if element.get('format', None) == 'ssh_private_key':
            return '\n'.join(['-----BEGIN ENCRYPTED PRIVATE KEY-----MIIBpjBABgkqhkiG9w0BBQ0wMzAbBgkqhkiG9w0BBQwwDgQI5yNCu9T5SnsCAggAMBQGCCqGSIb3DQMHBAhJISTgOAxtYwSCAWDXK/a1lxHIbRZHud1tfRMR4ROqkmr4kVGAnfqTyGptZUt3ZtBgrYlFAaZ1z0wxnhmhn3KIbqebI4w0cIL/3tmQ6eBD1Ad1nSEjUxZCuzTkimXQ88wZLzIS9KHc8GhINiUu5rKWbyvWA13Ykc0w65Ot5MSw3cQcw1LEDJjTculyDcRQgiRfKH5376qTzukileeTrNebNq+wbhY1kEPAHojercB7d10E+QcbjJX1Tb1Zangom1qH9t/pepmV0Hn4EMzDs6DS2SWTffTddTY4dQzvksmLkP+Ji8hkFIZwUkWpT9/k7MeklgtTiy0lR/Jj9CxAIQVxP8alLWbIqwCNRApleSmqtittZ+NdsuNeTm3iUaPGYSw237tjLyVE6pr0EJqLv7VUClvJvBnH2qhQEtWYB9gvE1dSBioGu40pXVfjiLqhEKVVVEoHpI32oMkojhCGJs8Oow4bAxkzQFCtuWB1-----END ENCRYPTED PRIVATE KEY-----'])
        if element['id'] == 'host':
            return 'https://foo.invalid'
        return 'fooo'
    elif element['type'] == 'boolean':
        return False
    raise Exception('No generator written for {} type'.format(element.get('type', 'unknown')))

def credential_kind(source):
    if False:
        while True:
            i = 10
    'Given the inventory source kind, return expected credential kind'
    return source.replace('ec2', 'aws')

@pytest.fixture
def fake_credential_factory():
    if False:
        while True:
            i = 10

    def wrap(source):
        if False:
            print('Hello World!')
        ct = CredentialType.defaults[credential_kind(source)]()
        ct.save()
        inputs = {}
        var_specs = {}
        for element in ct.inputs.get('fields'):
            var_specs[element['id']] = element
        for var in var_specs.keys():
            inputs[var] = generate_fake_var(var_specs[var])
        if source == 'controller':
            inputs.pop('oauth_token')
        return Credential.objects.create(credential_type=ct, inputs=inputs)
    return wrap

def read_content(private_data_dir, raw_env, inventory_update):
    if False:
        return 10
    'Read the environmental data laid down by the task system\n    template out private and secret data so they will be readable and predictable\n    return a dictionary `content` with file contents, keyed off environment variable\n        that references the file\n    '
    env = {}
    exclude_keys = set(('PATH', 'INVENTORY_SOURCE_ID', 'INVENTORY_UPDATE_ID'))
    for key in dir(settings):
        if key.startswith('ANSIBLE_'):
            exclude_keys.add(key)
    for (k, v) in raw_env.items():
        if k in STANDARD_INVENTORY_UPDATE_ENV or k in exclude_keys:
            continue
        if k not in os.environ or v != os.environ[k]:
            env[k] = v
    inverse_env = {}
    for (key, value) in env.items():
        inverse_env.setdefault(value, []).append(key)
    cache_file_regex = re.compile('/tmp/awx_{0}_[a-zA-Z0-9_]+/{1}_cache[a-zA-Z0-9_]+'.format(inventory_update.id, inventory_update.source))
    private_key_regex = re.compile('-----BEGIN ENCRYPTED PRIVATE KEY-----.*-----END ENCRYPTED PRIVATE KEY-----')
    dir_contents = {}
    referenced_paths = set()
    file_aliases = {}
    filename_list = os.listdir(private_data_dir)
    for subdir in ('env', 'inventory'):
        if subdir in filename_list:
            filename_list.remove(subdir)
            for filename in os.listdir(os.path.join(private_data_dir, subdir)):
                filename_list.append(os.path.join(subdir, filename))
    filename_list = sorted(filename_list, key=lambda fn: inverse_env.get(os.path.join(private_data_dir, fn), [fn])[0])
    for filename in filename_list:
        if filename in ('args', 'project'):
            continue
        abs_file_path = os.path.join(private_data_dir, filename)
        file_aliases[abs_file_path] = filename
        runner_path = to_container_path(abs_file_path, private_data_dir)
        if runner_path in inverse_env:
            referenced_paths.add(abs_file_path)
            alias = 'file_reference'
            for i in range(10):
                if alias not in file_aliases.values():
                    break
                alias = 'file_reference_{}'.format(i)
            else:
                raise RuntimeError('Test not able to cope with >10 references by env vars. Something probably went very wrong.')
            file_aliases[abs_file_path] = alias
            for env_key in inverse_env[runner_path]:
                env[env_key] = '{{{{ {} }}}}'.format(alias)
        try:
            with open(abs_file_path, 'r') as f:
                dir_contents[abs_file_path] = f.read()
            if abs_file_path.endswith('.yml') and 'plugin: ' in dir_contents[abs_file_path]:
                referenced_paths.add(abs_file_path)
            elif cache_file_regex.match(abs_file_path):
                file_aliases[abs_file_path] = 'cache_file'
        except IsADirectoryError:
            dir_contents[abs_file_path] = '<directory>'
            if cache_file_regex.match(abs_file_path):
                file_aliases[abs_file_path] = 'cache_dir'
    for (abs_file_path, file_content) in dir_contents.copy().items():
        if cache_file_regex.match(file_content):
            if 'cache_dir' not in file_aliases.values() and 'cache_file' not in file_aliases in file_aliases.values():
                raise AssertionError('A cache file was referenced but never created, files:\n{}'.format(json.dumps(dir_contents, indent=4)))
        for target_path in dir_contents.keys():
            other_alias = file_aliases[target_path]
            if target_path in file_content:
                referenced_paths.add(target_path)
                dir_contents[abs_file_path] = file_content.replace(target_path, '{{ ' + other_alias + ' }}')
    ignore_files = [os.path.join(private_data_dir, 'env', 'settings')]
    content = {}
    for (abs_file_path, file_content) in dir_contents.items():
        if abs_file_path not in referenced_paths and abs_file_path not in ignore_files:
            raise AssertionError('File {} is not referenced. References and files:\n{}\n{}'.format(abs_file_path, json.dumps(env, indent=4), json.dumps(dir_contents, indent=4)))
        file_content = private_key_regex.sub('{{private_key}}', file_content)
        content[file_aliases[abs_file_path]] = file_content
    return (env, content)

def create_reference_data(source_dir, env, content):
    if False:
        while True:
            i = 10
    if not os.path.exists(source_dir):
        os.mkdir(source_dir)
    if content:
        files_dir = os.path.join(source_dir, 'files')
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
        for (env_name, content) in content.items():
            with open(os.path.join(files_dir, env_name), 'w') as f:
                f.write(content)
    if env:
        with open(os.path.join(source_dir, 'env.json'), 'w') as f:
            json.dump(env, f, indent=4, sort_keys=True)

@pytest.mark.django_db
@pytest.mark.parametrize('this_kind', CLOUD_PROVIDERS)
def test_inventory_update_injected_content(this_kind, inventory, fake_credential_factory, mock_me):
    if False:
        i = 10
        return i + 15
    ExecutionEnvironment.objects.create(name='Control Plane EE', managed=True)
    ExecutionEnvironment.objects.create(name='Default Job EE', managed=False)
    injector = InventorySource.injectors[this_kind]
    if injector.plugin_name is None:
        pytest.skip('Use of inventory plugin is not enabled for this source')
    src_vars = dict(base_source_var='value_of_var')
    src_vars['plugin'] = injector.get_proper_name()
    inventory_source = InventorySource.objects.create(inventory=inventory, source=this_kind, source_vars=src_vars)
    inventory_source.credentials.add(fake_credential_factory(this_kind))
    inventory_update = inventory_source.create_unified_job()
    task = RunInventoryUpdate()

    def substitute_run(awx_receptor_job):
        if False:
            while True:
                i = 10
        'This method will replace run_pexpect\n        instead of running, it will read the private data directory contents\n        It will make assertions that the contents are correct\n        If MAKE_INVENTORY_REFERENCE_FILES is set, it will produce reference files\n        '
        envvars = awx_receptor_job.runner_params['envvars']
        private_data_dir = envvars.pop('AWX_PRIVATE_DATA_DIR')
        assert envvars.pop('ANSIBLE_INVENTORY_ENABLED') == 'auto'
        set_files = bool(os.getenv('MAKE_INVENTORY_REFERENCE_FILES', 'false').lower()[0] not in ['f', '0'])
        (env, content) = read_content(private_data_dir, envvars, inventory_update)
        inventory_filename = InventorySource.injectors[inventory_update.source]().filename
        assert len([True for k in content.keys() if k.endswith(inventory_filename)]) > 0, f"'{inventory_filename}' file not found in inventory update runtime files {content.keys()}"
        env.pop('ANSIBLE_COLLECTIONS_PATHS', None)
        base_dir = os.path.join(DATA, 'plugins')
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        source_dir = os.path.join(base_dir, this_kind)
        if set_files:
            create_reference_data(source_dir, env, content)
            pytest.skip('You set MAKE_INVENTORY_REFERENCE_FILES, so this created files, unset to run actual test.')
        else:
            source_dir = os.path.join(base_dir, this_kind)
            if not os.path.exists(source_dir):
                raise FileNotFoundError('Maybe you never made reference files? MAKE_INVENTORY_REFERENCE_FILES=true py.test ...\noriginal: {}')
            files_dir = os.path.join(source_dir, 'files')
            try:
                expected_file_list = os.listdir(files_dir)
            except FileNotFoundError:
                expected_file_list = []
            for f_name in expected_file_list:
                with open(os.path.join(files_dir, f_name), 'r') as f:
                    ref_content = f.read()
                    assert ref_content == content[f_name], f_name
            try:
                with open(os.path.join(source_dir, 'env.json'), 'r') as f:
                    ref_env_text = f.read()
                    ref_env = json.loads(ref_env_text)
            except FileNotFoundError:
                ref_env = {}
            assert ref_env == env
        Res = namedtuple('Result', ['status', 'rc'])
        return Res('successful', 0)
    with mock.patch('awx.main.queue.CallbackQueueDispatcher.dispatch', lambda self, obj: None):
        with mock.patch.object(UnifiedJob, 'websocket_emit_status', mock.Mock()):
            with mock.patch('awx.main.tasks.receptor.AWXReceptorJob.run', substitute_run):
                with mock.patch('awx.main.tasks.jobs.create_partition'):
                    task.run(inventory_update.pk)