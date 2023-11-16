"""
Manage LXD containers.

.. versionadded:: 2019.2.0

.. note:

    - :ref:`pylxd` version 2 is required to let this work,
      currently only available via pip.

        To install on Ubuntu:

        $ apt-get install libssl-dev python-pip
        $ pip install -U pylxd

    - you need lxd installed on the minion
      for the init() and version() methods.

    - for the config_get() and config_get() methods
      you need to have lxd-client installed.

.. _pylxd: https://github.com/lxc/pylxd/blob/master/doc/source/installation.rst

:maintainer: René Jochum <rene@jochums.at>
:maturity: new
:depends: python-pylxd
:platform: Linux
"""
from salt.exceptions import CommandExecutionError, SaltInvocationError
__docformat__ = 'restructuredtext en'
__virtualname__ = 'lxd_container'
CONTAINER_STATUS_RUNNING = 103
CONTAINER_STATUS_FROZEN = 110
CONTAINER_STATUS_STOPPED = 102

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the lxd module is available in __salt__\n    '
    if 'lxd.version' in __salt__:
        return __virtualname__
    return (False, 'lxd module could not be loaded')

def present(name, running=None, source=None, profiles=None, config=None, devices=None, architecture='x86_64', ephemeral=False, restart_on_change=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        return 10
    '\n    Create the named container if it does not exist\n\n    name\n        The name of the container to be created\n\n    running : None\n        * If ``True``, ensure that the container is running\n        * If ``False``, ensure that the container is stopped\n        * If ``None``, do nothing with regards to the running state of the\n          container\n\n    source : None\n        Can be either a string containing an image alias:\n\n        .. code-block:: none\n\n             "xenial/amd64"\n\n        or an dict with type "image" with alias:\n\n        .. code-block:: python\n\n            {"type": "image",\n             "alias": "xenial/amd64"}\n\n        or image with "fingerprint":\n\n        .. code-block:: python\n\n            {"type": "image",\n             "fingerprint": "SHA-256"}\n\n        or image with "properties":\n\n        .. code-block:: python\n\n            {"type": "image",\n             "properties": {\n                "os": "ubuntu",\n                "release": "14.04",\n                "architecture": "x86_64"\n             }}\n\n        or none:\n\n        .. code-block:: python\n\n            {"type": "none"}\n\n        or copy:\n\n        .. code-block:: python\n\n            {"type": "copy",\n             "source": "my-old-container"}\n\n    profiles : [\'default\']\n        List of profiles to apply on this container\n\n    config :\n        A config dict or None (None = unset).\n\n        Can also be a list:\n\n        .. code-block:: python\n\n            [{\'key\': \'boot.autostart\', \'value\': 1},\n             {\'key\': \'security.privileged\', \'value\': \'1\'}]\n\n    devices :\n        A device dict or None (None = unset).\n\n    architecture : \'x86_64\'\n        Can be one of the following:\n\n        * unknown\n        * i686\n        * x86_64\n        * armv7l\n        * aarch64\n        * ppc\n        * ppc64\n        * ppc64le\n        * s390x\n\n    ephemeral : False\n        Destroy this container after stop?\n\n    restart_on_change : False\n        Restart the container when we detect changes on the config or\n        its devices?\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    if profiles is None:
        profiles = ['default']
    if source is None:
        source = {}
    ret = {'name': name, 'running': running, 'profiles': profiles, 'source': source, 'config': config, 'devices': devices, 'architecture': architecture, 'ephemeral': ephemeral, 'restart_on_change': restart_on_change, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    container = None
    try:
        container = __salt__['lxd.container_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        pass
    if container is None:
        if __opts__['test']:
            msg = 'Would create the container "{}"'.format(name)
            ret['changes'] = {'created': msg}
            if running is True:
                msg = msg + ' and start it.'
                ret['changes']['started'] = 'Would start the container "{}"'.format(name)
            ret['changes'] = {'created': msg}
            return _unchanged(ret, msg)
        try:
            __salt__['lxd.container_create'](name, source, profiles, config, devices, architecture, ephemeral, True, remote_addr, cert, key, verify_cert)
        except CommandExecutionError as e:
            return _error(ret, str(e))
        msg = 'Created the container "{}"'.format(name)
        ret['changes'] = {'created': msg}
        if running is True:
            try:
                __salt__['lxd.container_start'](name, remote_addr, cert, key, verify_cert)
            except CommandExecutionError as e:
                return _error(ret, str(e))
            msg = msg + ' and started it.'
            ret['changes'] = {'started': 'Started the container "{}"'.format(name)}
        return _success(ret, msg)
    new_profiles = set(map(str, profiles))
    old_profiles = set(map(str, container.profiles))
    container_changed = False
    profile_changes = []
    for k in old_profiles.difference(new_profiles):
        if not __opts__['test']:
            profile_changes.append('Removed profile "{}"'.format(k))
            old_profiles.discard(k)
        else:
            profile_changes.append('Would remove profile "{}"'.format(k))
    for k in new_profiles.difference(old_profiles):
        if not __opts__['test']:
            profile_changes.append('Added profile "{}"'.format(k))
            old_profiles.add(k)
        else:
            profile_changes.append('Would add profile "{}"'.format(k))
    if profile_changes:
        container_changed = True
        ret['changes']['profiles'] = profile_changes
        container.profiles = list(old_profiles)
    (config, devices) = __salt__['lxd.normalize_input_values'](config, devices)
    changes = __salt__['lxd.sync_config_devices'](container, config, devices, __opts__['test'])
    if changes:
        container_changed = True
        ret['changes'].update(changes)
    is_running = container.status_code == CONTAINER_STATUS_RUNNING
    if not __opts__['test']:
        try:
            __salt__['lxd.pylxd_save_object'](container)
        except CommandExecutionError as e:
            return _error(ret, str(e))
    if running != is_running:
        if running is True:
            if __opts__['test']:
                changes['running'] = 'Would start the container'
                return _unchanged(ret, 'Container "{}" would get changed and started.'.format(name))
            else:
                container.start(wait=True)
                changes['running'] = 'Started the container'
        elif running is False:
            if __opts__['test']:
                changes['stopped'] = 'Would stopped the container'
                return _unchanged(ret, 'Container "{}" would get changed and stopped.'.format(name))
            else:
                container.stop(wait=True)
                changes['stopped'] = 'Stopped the container'
    if (running is True or running is None) and is_running and restart_on_change and container_changed:
        if __opts__['test']:
            changes['restarted'] = 'Would restart the container'
            return _unchanged(ret, 'Would restart the container "{}"'.format(name))
        else:
            container.restart(wait=True)
            changes['restarted'] = 'Container "{}" has been restarted'.format(name)
            return _success(ret, 'Container "{}" has been restarted'.format(name))
    if not container_changed:
        return _success(ret, 'No changes')
    if __opts__['test']:
        return _unchanged(ret, 'Container "{}" would get changed.'.format(name))
    return _success(ret, '{} changes'.format(len(ret['changes'].keys())))

def absent(name, stop=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    '\n    Ensure a LXD container is not present, destroying it if present\n\n    name :\n        The name of the container to destroy\n\n    stop :\n        stop before destroying\n        default: false\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    ret = {'name': name, 'stop': stop, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    try:
        container = __salt__['lxd.container_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        return _success(ret, 'Container "{}" not found.'.format(name))
    if __opts__['test']:
        ret['changes'] = {'removed': 'Container "{}" would get deleted.'.format(name)}
        return _unchanged(ret, ret['changes']['removed'])
    if stop and container.status_code == CONTAINER_STATUS_RUNNING:
        container.stop(wait=True)
    container.delete(wait=True)
    ret['changes']['deleted'] = 'Container "{}" has been deleted.'.format(name)
    return _success(ret, ret['changes']['deleted'])

def running(name, restart=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Ensure a LXD container is running and restart it if restart is True\n\n    name :\n        The name of the container to start/restart.\n\n    restart :\n        restart the container if it is already started.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    ret = {'name': name, 'restart': restart, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    try:
        container = __salt__['lxd.container_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        return _error(ret, 'Container "{}" not found'.format(name))
    is_running = container.status_code == CONTAINER_STATUS_RUNNING
    if is_running:
        if not restart:
            return _success(ret, 'The container "{}" is already running'.format(name))
        elif __opts__['test']:
            ret['changes']['restarted'] = 'Would restart the container "{}"'.format(name)
            return _unchanged(ret, ret['changes']['restarted'])
        else:
            container.restart(wait=True)
            ret['changes']['restarted'] = 'Restarted the container "{}"'.format(name)
            return _success(ret, ret['changes']['restarted'])
    if __opts__['test']:
        ret['changes']['started'] = 'Would start the container "{}"'.format(name)
        return _unchanged(ret, ret['changes']['started'])
    container.start(wait=True)
    ret['changes']['started'] = 'Started the container "{}"'.format(name)
    return _success(ret, ret['changes']['started'])

def frozen(name, start=True, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Ensure a LXD container is frozen, start and freeze it if start is true\n\n    name :\n        The name of the container to freeze\n\n    start :\n        start and freeze it\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    ret = {'name': name, 'start': start, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    try:
        container = __salt__['lxd.container_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        return _error(ret, 'Container "{}" not found'.format(name))
    if container.status_code == CONTAINER_STATUS_FROZEN:
        return _success(ret, 'Container "{}" is alredy frozen'.format(name))
    is_running = container.status_code == CONTAINER_STATUS_RUNNING
    if not is_running and (not start):
        return _error(ret, 'Container "{}" is not running and start is False, cannot freeze it'.format(name))
    elif not is_running and start:
        if __opts__['test']:
            ret['changes']['started'] = 'Would start the container "{}" and freeze it after'.format(name)
            return _unchanged(ret, ret['changes']['started'])
        else:
            container.start(wait=True)
            ret['changes']['started'] = 'Start the container "{}"'.format(name)
    if __opts__['test']:
        ret['changes']['frozen'] = 'Would freeze the container "{}"'.format(name)
        return _unchanged(ret, ret['changes']['frozen'])
    container.freeze(wait=True)
    ret['changes']['frozen'] = 'Froze the container "{}"'.format(name)
    return _success(ret, ret['changes']['frozen'])

def stopped(name, kill=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Ensure a LXD container is stopped, kill it if kill is true else stop it\n\n    name :\n        The name of the container to stop\n\n    kill :\n        kill if true\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    ret = {'name': name, 'kill': kill, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    try:
        container = __salt__['lxd.container_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        return _error(ret, 'Container "{}" not found'.format(name))
    if container.status_code == CONTAINER_STATUS_STOPPED:
        return _success(ret, 'Container "{}" is already stopped'.format(name))
    if __opts__['test']:
        ret['changes']['stopped'] = 'Would stop the container "{}"'.format(name)
        return _unchanged(ret, ret['changes']['stopped'])
    container.stop(force=kill, wait=True)
    ret['changes']['stopped'] = 'Stopped the container "{}"'.format(name)
    return _success(ret, ret['changes']['stopped'])

def migrated(name, remote_addr, cert, key, verify_cert, src_remote_addr, stop_and_start=False, src_cert=None, src_key=None, src_verify_cert=None):
    if False:
        for i in range(10):
            print('nop')
    'Ensure a container is migrated to another host\n\n    If the container is running, it either must be shut down\n    first (use stop_and_start=True) or criu must be installed\n    on the source and destination machines.\n\n    For this operation both certs need to be authenticated,\n    use :mod:`lxd.authenticate <salt.states.lxd.authenticate`\n    to authenticate your cert(s).\n\n    name :\n        The container to migrate\n\n    remote_addr :\n        An URL to the destination remote Server\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    src_remote_addr :\n        An URL to the source remote Server\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    stop_and_start:\n        Stop before migrating and start after\n\n    src_cert :\n        PEM Formatted SSL Zertifikate, if None we copy "cert"\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    src_key :\n        PEM Formatted SSL Key, if None we copy "key"\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    src_verify_cert :\n        Wherever to verify the cert, if None we copy "verify_cert"\n    '
    ret = {'name': name, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'src_remote_addr': src_remote_addr, 'src_and_start': stop_and_start, 'src_cert': src_cert, 'src_key': src_key, 'changes': {}}
    dest_container = None
    try:
        dest_container = __salt__['lxd.container_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        pass
    if dest_container is not None:
        return _success(ret, 'Container "{}" exists on the destination'.format(name))
    if src_verify_cert is None:
        src_verify_cert = verify_cert
    try:
        __salt__['lxd.container_get'](name, src_remote_addr, src_cert, src_key, src_verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        return _error(ret, 'Source Container "{}" not found'.format(name))
    if __opts__['test']:
        ret['changes']['migrated'] = 'Would migrate the container "{}" from "{}" to "{}"'.format(name, src_remote_addr, remote_addr)
        return _unchanged(ret, ret['changes']['migrated'])
    try:
        __salt__['lxd.container_migrate'](name, stop_and_start, remote_addr, cert, key, verify_cert, src_remote_addr, src_cert, src_key, src_verify_cert)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    ret['changes']['migrated'] = 'Migrated the container "{}" from "{}" to "{}"'.format(name, src_remote_addr, remote_addr)
    return _success(ret, ret['changes']['migrated'])

def _success(ret, success_msg):
    if False:
        while True:
            i = 10
    ret['result'] = True
    ret['comment'] = success_msg
    if 'changes' not in ret:
        ret['changes'] = {}
    return ret

def _unchanged(ret, msg):
    if False:
        return 10
    ret['result'] = None
    ret['comment'] = msg
    if 'changes' not in ret:
        ret['changes'] = {}
    return ret

def _error(ret, err_msg):
    if False:
        i = 10
        return i + 15
    ret['result'] = False
    ret['comment'] = err_msg
    if 'changes' not in ret:
        ret['changes'] = {}
    return ret