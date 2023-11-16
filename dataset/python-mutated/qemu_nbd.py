"""
Qemu Command Wrapper

The qemu system comes with powerful tools, such as qemu-img and qemu-nbd which
are used here to build up kvm images.
"""
import glob
import logging
import os
import tempfile
import time
import salt.crypt
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only load if qemu-img and qemu-nbd are installed\n    '
    if salt.utils.path.which('qemu-nbd'):
        return 'qemu_nbd'
    return (False, 'The qemu_nbd execution module cannot be loaded: the qemu-nbd binary is not in the path.')

def connect(image):
    if False:
        while True:
            i = 10
    "\n    Activate nbd for an image file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' qemu_nbd.connect /tmp/image.raw\n    "
    if not os.path.isfile(image):
        log.warning('Could not connect image: %s does not exist', image)
        return ''
    if salt.utils.path.which('sfdisk'):
        fdisk = 'sfdisk -d'
    else:
        fdisk = 'fdisk -l'
    __salt__['cmd.run']('modprobe nbd max_part=63')
    for nbd in glob.glob('/dev/nbd?'):
        if __salt__['cmd.retcode']('{} {}'.format(fdisk, nbd)):
            while True:
                __salt__['cmd.run']('qemu-nbd -c {} {}'.format(nbd, image), python_shell=False)
                if not __salt__['cmd.retcode']('{} {}'.format(fdisk, nbd)):
                    break
            return nbd
    log.warning('Could not connect image: %s', image)
    return ''

def mount(nbd, root=None):
    if False:
        print('Hello World!')
    "\n    Pass in the nbd connection device location, mount all partitions and return\n    a dict of mount points\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' qemu_nbd.mount /dev/nbd0\n    "
    __salt__['cmd.run']('partprobe {}'.format(nbd), python_shell=False)
    ret = {}
    if root is None:
        root = os.path.join(tempfile.gettempdir(), 'nbd', os.path.basename(nbd))
    for part in glob.glob('{}p*'.format(nbd)):
        m_pt = os.path.join(root, os.path.basename(part))
        time.sleep(1)
        mnt = __salt__['mount.mount'](m_pt, part, True)
        if mnt is not True:
            continue
        ret[m_pt] = part
    return ret

def init(image, root=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Mount the named image via qemu-nbd and return the mounted roots\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' qemu_nbd.init /srv/image.qcow2\n    "
    nbd = connect(image)
    if not nbd:
        return ''
    return mount(nbd, root)

def clear(mnt):
    if False:
        i = 10
        return i + 15
    '\n    Pass in the mnt dict returned from nbd_mount to unmount and disconnect\n    the image from nbd. If all of the partitions are unmounted return an\n    empty dict, otherwise return a dict containing the still mounted\n    partitions\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' qemu_nbd.clear \'{"/mnt/foo": "/dev/nbd0p1"}\'\n    '
    ret = {}
    nbds = set()
    for (m_pt, dev) in mnt.items():
        mnt_ret = __salt__['mount.umount'](m_pt)
        if mnt_ret is not True:
            ret[m_pt] = dev
        nbds.add(dev[:dev.rindex('p')])
    if ret:
        return ret
    for nbd in nbds:
        __salt__['cmd.run']('qemu-nbd -d {}'.format(nbd), python_shell=False)
    return ret