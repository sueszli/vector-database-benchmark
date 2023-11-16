from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: httptester_kinit\nshort_description: Get Kerberos ticket\ndescription: Get Kerberos ticket using kinit non-interactively.\noptions:\n  username:\n    description: The username to get the ticket for.\n    required: true\n    type: str\n  password:\n    description: The password for I(username).\n    required; true\n    type: str\nauthor:\n- Ansible Project\n'
EXAMPLES = '\n#\n'
RETURN = '\n#\n'
import contextlib
import errno
import os
import subprocess
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

@contextlib.contextmanager
def env_path(name, value, default_value):
    if False:
        return 10
    ' Adds a value to a PATH-like env var and preserve the existing value if present. '
    orig_value = os.environ.get(name, None)
    os.environ[name] = '%s:%s' % (value, orig_value or default_value)
    try:
        yield
    finally:
        if orig_value:
            os.environ[name] = orig_value
        else:
            del os.environ[name]

@contextlib.contextmanager
def krb5_conf(module, config):
    if False:
        print('Hello World!')
    ' Runs with a custom krb5.conf file that extends the existing config if present. '
    if config:
        ini_config = configparser.ConfigParser()
        for (section, entries) in config.items():
            ini_config.add_section(section)
            for (key, value) in entries.items():
                ini_config.set(section, key, value)
        config_path = os.path.join(module.tmpdir, 'krb5.conf')
        with open(config_path, mode='wt') as config_fd:
            ini_config.write(config_fd)
        with env_path('KRB5_CONFIG', config_path, '/etc/krb5.conf'):
            yield
    else:
        yield

def main():
    if False:
        while True:
            i = 10
    module_args = dict(username=dict(type='str', required=True), password=dict(type='str', required=True, no_log=True))
    module = AnsibleModule(argument_spec=module_args, required_together=[('username', 'password')])
    sysname = os.uname()[0]
    prefix = '/usr/local/bin/' if sysname == 'FreeBSD' else ''
    is_heimdal = sysname in ['Darwin', 'FreeBSD']
    try:
        process = subprocess.Popen(['%skrb5-config' % prefix, '--version'], stdout=subprocess.PIPE)
        (stdout, stderr) = process.communicate()
        version = to_text(stdout)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        version = 'Unknown (no krb5-config)'
    kinit_args = ['%skinit' % prefix]
    config = {}
    if is_heimdal:
        kinit_args.append('--password-file=STDIN')
        config['logging'] = {'krb5': 'FILE:/dev/stdout'}
    kinit_args.append(to_text(module.params['username'], errors='surrogate_or_strict'))
    with krb5_conf(module, config):
        kinit_env = os.environ.copy()
        kinit_env['KRB5_TRACE'] = '/dev/stdout'
        process = subprocess.Popen(kinit_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=kinit_env)
        (stdout, stderr) = process.communicate(to_bytes(module.params['password'], errors='surrogate_or_strict') + b'\n')
        rc = process.returncode
    module.exit_json(changed=True, stdout=to_text(stdout), stderr=to_text(stderr), rc=rc, version=version)
if __name__ == '__main__':
    main()