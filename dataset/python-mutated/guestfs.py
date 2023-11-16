"""
Interact with virtual machine images via libguestfs

:depends:   - libguestfs
"""
import hashlib
import logging
import os
import tempfile
import time
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if libguestfs python bindings are installed\n    '
    if salt.utils.path.which('guestmount'):
        return 'guestfs'
    return (False, 'The guestfs execution module cannot be loaded: guestmount binary not in path.')

def mount(location, access='rw', root=None):
    if False:
        while True:
            i = 10
    "\n    Mount an image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' guest.mount /srv/images/fedora.qcow\n    "
    if root is None:
        root = os.path.join(tempfile.gettempdir(), 'guest', location.lstrip(os.sep).replace('/', '.'))
        log.debug('Using root %s', root)
    if not os.path.isdir(root):
        try:
            os.makedirs(root)
        except OSError:
            pass
    while True:
        if os.listdir(root):
            hash_type = getattr(hashlib, __opts__.get('hash_type', 'md5'))
            rand = hash_type(os.urandom(32)).hexdigest()
            root = os.path.join(tempfile.gettempdir(), 'guest', location.lstrip(os.sep).replace('/', '.') + rand)
            log.debug('Establishing new root as %s', root)
            if not os.path.isdir(root):
                try:
                    os.makedirs(root)
                except OSError:
                    log.info('Path already existing: %s', root)
        else:
            break
    cmd = 'guestmount -i -a {} --{} {}'.format(location, access, root)
    __salt__['cmd.run'](cmd, python_shell=False)
    return root

def umount(name, disk=None):
    if False:
        return 10
    "\n    Unmount an image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' guestfs.umount /mountpoint disk=/srv/images/fedora.qcow\n    "
    cmd = 'guestunmount -q {}'.format(name)
    __salt__['cmd.run'](cmd)
    loops = 0
    while disk is not None and loops < 5 and (len(__salt__['cmd.run']('lsof {}'.format(disk)).splitlines()) != 0):
        loops = loops + 1
        time.sleep(1)