from __future__ import annotations
import json
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
_DEVICE_CONFIGS = {}
vyos_provider_spec = {'host': dict(), 'port': dict(type='int'), 'username': dict(fallback=(env_fallback, ['ANSIBLE_NET_USERNAME'])), 'password': dict(fallback=(env_fallback, ['ANSIBLE_NET_PASSWORD']), no_log=True), 'ssh_keyfile': dict(fallback=(env_fallback, ['ANSIBLE_NET_SSH_KEYFILE']), type='path'), 'timeout': dict(type='int')}
vyos_argument_spec = {'provider': dict(type='dict', options=vyos_provider_spec, removed_in_version=2.14)}

def get_provider_argspec():
    if False:
        for i in range(10):
            print('nop')
    return vyos_provider_spec

def get_connection(module):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(module, '_vyos_connection'):
        return module._vyos_connection
    capabilities = get_capabilities(module)
    network_api = capabilities.get('network_api')
    if network_api == 'cliconf':
        module._vyos_connection = Connection(module._socket_path)
    else:
        module.fail_json(msg='Invalid connection type %s' % network_api)
    return module._vyos_connection

def get_capabilities(module):
    if False:
        return 10
    if hasattr(module, '_vyos_capabilities'):
        return module._vyos_capabilities
    try:
        capabilities = Connection(module._socket_path).get_capabilities()
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    module._vyos_capabilities = json.loads(capabilities)
    return module._vyos_capabilities

def get_config(module, flags=None, format=None):
    if False:
        print('Hello World!')
    flags = [] if flags is None else flags
    global _DEVICE_CONFIGS
    if _DEVICE_CONFIGS != {}:
        return _DEVICE_CONFIGS
    else:
        connection = get_connection(module)
        try:
            out = connection.get_config(flags=flags, format=format)
        except ConnectionError as exc:
            module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
        cfg = to_text(out, errors='surrogate_then_replace').strip()
        _DEVICE_CONFIGS = cfg
        return cfg

def run_commands(module, commands, check_rc=True):
    if False:
        print('Hello World!')
    connection = get_connection(module)
    try:
        response = connection.run_commands(commands=commands, check_rc=check_rc)
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    return response

def load_config(module, commands, commit=False, comment=None):
    if False:
        print('Hello World!')
    connection = get_connection(module)
    try:
        response = connection.edit_config(candidate=commands, commit=commit, comment=comment)
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    return response.get('diff')