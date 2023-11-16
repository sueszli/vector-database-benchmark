"""
Modify, retrieve, or delete values from OpenStack configuration files.

:maintainer: Jeffrey C. Ollie <jeff@ocjtech.us>
:maturity: new
:depends:
:platform: linux

"""
import shlex
import salt.exceptions
import salt.utils.decorators.path
__func_alias__ = {'set_': 'set'}

def __virtual__():
    if False:
        while True:
            i = 10
    return True

def _fallback(*args, **kw):
    if False:
        i = 10
        return i + 15
    return 'The "openstack-config" command needs to be installed for this function to work.  Typically this is included in the "openstack-utils" package.'

@salt.utils.decorators.path.which('openstack-config')
def set_(filename, section, parameter, value):
    if False:
        return 10
    '\n    Set a value in an OpenStack configuration file.\n\n    filename\n        The full path to the configuration file\n\n    section\n        The section in which the parameter will be set\n\n    parameter\n        The parameter to change\n\n    value\n        The value to set\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call openstack_config.set /etc/keystone/keystone.conf sql connection foo\n    '
    filename = shlex.quote(filename)
    section = shlex.quote(section)
    parameter = shlex.quote(parameter)
    value = shlex.quote(str(value))
    result = __salt__['cmd.run_all']('openstack-config --set {} {} {} {}'.format(filename, section, parameter, value), python_shell=False)
    if result['retcode'] == 0:
        return result['stdout']
    else:
        raise salt.exceptions.CommandExecutionError(result['stderr'])

@salt.utils.decorators.path.which('openstack-config')
def get(filename, section, parameter):
    if False:
        return 10
    '\n    Get a value from an OpenStack configuration file.\n\n    filename\n        The full path to the configuration file\n\n    section\n        The section from which to search for the parameter\n\n    parameter\n        The parameter to return\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call openstack_config.get /etc/keystone/keystone.conf sql connection\n\n    '
    filename = shlex.quote(filename)
    section = shlex.quote(section)
    parameter = shlex.quote(parameter)
    result = __salt__['cmd.run_all'](f'openstack-config --get {filename} {section} {parameter}', python_shell=False)
    if result['retcode'] == 0:
        return result['stdout']
    else:
        raise salt.exceptions.CommandExecutionError(result['stderr'])

@salt.utils.decorators.path.which('openstack-config')
def delete(filename, section, parameter):
    if False:
        print('Hello World!')
    '\n    Delete a value from an OpenStack configuration file.\n\n    filename\n        The full path to the configuration file\n\n    section\n        The section from which to delete the parameter\n\n    parameter\n        The parameter to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-call openstack_config.delete /etc/keystone/keystone.conf sql connection\n    '
    filename = shlex.quote(filename)
    section = shlex.quote(section)
    parameter = shlex.quote(parameter)
    result = __salt__['cmd.run_all'](f'openstack-config --del {filename} {section} {parameter}', python_shell=False)
    if result['retcode'] == 0:
        return result['stdout']
    else:
        raise salt.exceptions.CommandExecutionError(result['stderr'])