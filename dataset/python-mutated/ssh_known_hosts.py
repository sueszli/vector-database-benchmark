"""
Control of SSH known_hosts entries
==================================

Manage the information stored in the known_hosts files.

.. code-block:: yaml

    github.com:
      ssh_known_hosts:
        - present
        - user: root
        - fingerprint: 16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48
        - fingerprint_hash_type: md5

    example.com:
      ssh_known_hosts:
        - absent
        - user: root
"""
import os
import salt.utils.platform
from salt.exceptions import CommandNotFoundError
__virtualname__ = 'ssh_known_hosts'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Does not work on Windows, requires ssh module functions\n    '
    if salt.utils.platform.is_windows():
        return (False, 'ssh_known_hosts: Does not support Windows')
    return __virtualname__

def present(name, user=None, fingerprint=None, key=None, port=None, enc=None, config=None, hash_known_hosts=True, timeout=5, fingerprint_hash_type=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verifies that the specified host is known by the specified user\n\n    On many systems, specifically those running with openssh 4 or older, the\n    ``enc`` option must be set, only openssh 5 and above can detect the key\n    type.\n\n    name\n        The name of the remote host (e.g. "github.com")\n        Note that only a single hostname is supported, if foo.example.com and\n        bar.example.com have the same host you will need two separate Salt\n        States to represent them.\n\n    user\n        The user who owns the ssh authorized keys file to modify\n\n    fingerprint\n        The fingerprint of the key which must be present in the known_hosts\n        file (optional if key specified)\n\n    key\n        The public key which must be present in the known_hosts file\n        (optional if fingerprint specified)\n\n    port\n        optional parameter, port which will be used to when requesting the\n        public key from the remote host, defaults to port 22.\n\n    enc\n        Defines what type of key is being used, can be ed25519, ecdsa,\n        ssh-rsa, ssh-dss or any other type as of openssh server version 8.7.\n\n    config\n        The location of the authorized keys file relative to the user\'s home\n        directory, defaults to ".ssh/known_hosts". If no user is specified,\n        defaults to "/etc/ssh/ssh_known_hosts". If present, must be an\n        absolute path when a user is not specified.\n\n    hash_known_hosts : True\n        Hash all hostnames and addresses in the known hosts file.\n\n    timeout : int\n        Set the timeout for connection attempts.  If ``timeout`` seconds have\n        elapsed since a connection was initiated to a host or since the last\n        time anything was read from that host, then the connection is closed\n        and the host in question considered unavailable.  Default is 5 seconds.\n\n        .. versionadded:: 2016.3.0\n\n    fingerprint_hash_type\n        The public key fingerprint hash type that the public key fingerprint\n        was originally hashed with. This defaults to ``sha256`` if not specified.\n\n        .. versionadded:: 2016.11.4\n        .. versionchanged:: 2017.7.0\n\n            default changed from ``md5`` to ``sha256``\n\n    '
    ret = {'name': name, 'changes': {}, 'result': None if __opts__['test'] else True, 'comment': ''}
    if not user:
        config = config or '/etc/ssh/ssh_known_hosts'
    else:
        config = config or '.ssh/known_hosts'
    if not user and (not os.path.isabs(config)):
        comment = 'If not specifying a "user", specify an absolute "config".'
        ret['result'] = False
        return dict(ret, comment=comment)
    if __opts__['test']:
        if key and fingerprint:
            comment = 'Specify either "key" or "fingerprint", not both.'
            ret['result'] = False
            return dict(ret, comment=comment)
        elif key and (not enc):
            comment = 'Required argument "enc" if using "key" argument.'
            ret['result'] = False
            return dict(ret, comment=comment)
        try:
            result = __salt__['ssh.check_known_host'](user, name, key=key, fingerprint=fingerprint, config=config, port=port, fingerprint_hash_type=fingerprint_hash_type)
        except CommandNotFoundError as err:
            ret['result'] = False
            ret['comment'] = 'ssh.check_known_host error: {}'.format(err)
            return ret
        if result == 'exists':
            comment = 'Host {} is already in {}'.format(name, config)
            ret['result'] = True
            return dict(ret, comment=comment)
        elif result == 'add':
            comment = 'Key for {} is set to be added to {}'.format(name, config)
            return dict(ret, comment=comment)
        else:
            comment = 'Key for {} is set to be updated in {}'.format(name, config)
            return dict(ret, comment=comment)
    result = __salt__['ssh.set_known_host'](user=user, hostname=name, fingerprint=fingerprint, key=key, port=port, enc=enc, config=config, hash_known_hosts=hash_known_hosts, timeout=timeout, fingerprint_hash_type=fingerprint_hash_type)
    if result['status'] == 'exists':
        return dict(ret, comment='{} already exists in {}'.format(name, config))
    elif result['status'] == 'error':
        return dict(ret, result=False, comment=result['error'])
    elif key:
        new_key = result['new'][0]['key']
        return dict(ret, changes={'old': result['old'], 'new': result['new']}, comment="{}'s key saved to {} (key: {})".format(name, config, new_key))
    else:
        fingerprint = result['new'][0]['fingerprint']
        return dict(ret, changes={'old': result['old'], 'new': result['new']}, comment="{}'s key saved to {} (fingerprint: {})".format(name, config, fingerprint))

def absent(name, user=None, config=None):
    if False:
        while True:
            i = 10
    '\n    Verifies that the specified host is not known by the given user\n\n    name\n        The host name\n        Note that only single host names are supported.  If foo.example.com\n        and bar.example.com are the same machine and you need to exclude both,\n        you will need one Salt state for each.\n\n    user\n        The user who owns the ssh authorized keys file to modify\n\n    config\n        The location of the authorized keys file relative to the user\'s home\n        directory, defaults to ".ssh/known_hosts". If no user is specified,\n        defaults to "/etc/ssh/ssh_known_hosts". If present, must be an\n        absolute path when a user is not specified.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if not user:
        config = config or '/etc/ssh/ssh_known_hosts'
    else:
        config = config or '.ssh/known_hosts'
    if not user and (not os.path.isabs(config)):
        comment = 'If not specifying a "user", specify an absolute "config".'
        ret['result'] = False
        return dict(ret, comment=comment)
    known_host = __salt__['ssh.get_known_host_entries'](user=user, hostname=name, config=config)
    if not known_host:
        return dict(ret, comment='Host is already absent')
    if __opts__['test']:
        comment = 'Key for {} is set to be removed from {}'.format(name, config)
        ret['result'] = None
        return dict(ret, comment=comment)
    rm_result = __salt__['ssh.rm_known_host'](user=user, hostname=name, config=config)
    if rm_result['status'] == 'error':
        return dict(ret, result=False, comment=rm_result['error'])
    else:
        return dict(ret, changes={'old': known_host, 'new': None}, result=True, comment=rm_result['comment'])