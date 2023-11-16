"""
Support for eselect, Gentoo's configuration and management tool.
"""
import logging
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only work on Gentoo systems with eselect installed\n    '
    if __grains__['os'] == 'Gentoo' and salt.utils.path.which('eselect'):
        return 'eselect'
    return (False, 'The eselect execution module cannot be loaded: either the system is not Gentoo or the eselect binary is not in the path.')

def exec_action(module, action, module_parameter=None, action_parameter=None, state_only=False):
    if False:
        while True:
            i = 10
    "\n    Execute an arbitrary action on a module.\n\n    module\n        name of the module to be executed\n\n    action\n        name of the module's action to be run\n\n    module_parameter\n        additional params passed to the defined module\n\n    action_parameter\n        additional params passed to the defined action\n\n    state_only\n        don't return any output but only the success/failure of the operation\n\n    CLI Example (updating the ``php`` implementation used for ``apache2``):\n\n    .. code-block:: bash\n\n        salt '*' eselect.exec_action php update action_parameter='apache2'\n    "
    out = __salt__['cmd.run']('eselect --brief --colour=no {} {} {} {}'.format(module, module_parameter or '', action, action_parameter or ''), python_shell=False)
    out = out.strip().split('\n')
    if out[0].startswith('!!! Error'):
        return False
    if state_only:
        return True
    if not out:
        return False
    if len(out) == 1 and (not out[0].strip()):
        return False
    return out

def get_modules():
    if False:
        while True:
            i = 10
    "\n    List available ``eselect`` modules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' eselect.get_modules\n    "
    modules = []
    module_list = exec_action('modules', 'list', action_parameter='--only-names')
    if not module_list:
        return None
    for module in module_list:
        if module not in ['help', 'usage', 'version']:
            modules.append(module)
    return modules

def get_target_list(module, action_parameter=None):
    if False:
        while True:
            i = 10
    "\n    List available targets for the given module.\n\n    module\n        name of the module to be queried for its targets\n\n    action_parameter\n        additional params passed to the defined action\n\n        .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' eselect.get_target_list kernel\n    "
    exec_output = exec_action(module, 'list', action_parameter=action_parameter)
    if not exec_output:
        return None
    target_list = []
    if isinstance(exec_output, list):
        for item in exec_output:
            target_list.append(item.split(None, 1)[0])
        return target_list
    return None

def get_current_target(module, module_parameter=None, action_parameter=None):
    if False:
        print('Hello World!')
    "\n    Get the currently selected target for the given module.\n\n    module\n        name of the module to be queried for its current target\n\n    module_parameter\n        additional params passed to the defined module\n\n    action_parameter\n        additional params passed to the 'show' action\n\n    CLI Example (current target of system-wide ``java-vm``):\n\n    .. code-block:: bash\n\n        salt '*' eselect.get_current_target java-vm action_parameter='system'\n\n    CLI Example (current target of ``kernel`` symlink):\n\n    .. code-block:: bash\n\n        salt '*' eselect.get_current_target kernel\n    "
    result = exec_action(module, 'show', module_parameter=module_parameter, action_parameter=action_parameter)[0]
    if not result:
        return None
    if result == '(unset)':
        return None
    return result

def set_target(module, target, module_parameter=None, action_parameter=None):
    if False:
        print('Hello World!')
    "\n    Set the target for the given module.\n    Target can be specified by index or name.\n\n    module\n        name of the module for which a target should be set\n\n    target\n        name of the target to be set for this module\n\n    module_parameter\n        additional params passed to the defined module\n\n    action_parameter\n        additional params passed to the defined action\n\n    CLI Example (setting target of system-wide ``java-vm``):\n\n    .. code-block:: bash\n\n        salt '*' eselect.set_target java-vm icedtea-bin-7 action_parameter='system'\n\n    CLI Example (setting target of ``kernel`` symlink):\n\n    .. code-block:: bash\n\n        salt '*' eselect.set_target kernel linux-3.17.5-gentoo\n    "
    if action_parameter:
        action_parameter = '{} {}'.format(action_parameter, target)
    else:
        action_parameter = target
    if module not in get_modules():
        log.error('Module %s not available', module)
        return False
    exec_result = exec_action(module, 'set', module_parameter=module_parameter, action_parameter=action_parameter, state_only=True)
    if exec_result:
        return exec_result
    return False