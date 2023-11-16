"""
Qemu-img Command Wrapper
========================

The qemu img command is wrapped for specific functions

:depends: qemu-img
"""
import os
import salt.utils.path

def __virtual__():
    if False:
        return 10
    '\n    Only load if qemu-img is installed\n    '
    if salt.utils.path.which('qemu-img'):
        return 'qemu_img'
    return (False, 'The qemu_img execution module cannot be loaded: the qemu-img binary is not in the path.')

def make_image(location, size, fmt):
    if False:
        i = 10
        return i + 15
    "\n    Create a blank virtual machine image file of the specified size in\n    megabytes. The image can be created in any format supported by qemu\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' qemu_img.make_image /tmp/image.qcow 2048 qcow2\n        salt '*' qemu_img.make_image /tmp/image.raw 10240 raw\n    "
    if not os.path.isabs(location):
        return ''
    if not os.path.isdir(os.path.dirname(location)):
        return ''
    if not __salt__['cmd.retcode']('qemu-img create -f {} {} {}M'.format(fmt, location, size), python_shell=False):
        return location
    return ''

def convert(orig, dest, fmt):
    if False:
        return 10
    "\n    Convert an existing disk image to another format using qemu-img\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' qemu_img.convert /path/to/original.img /path/to/new.img qcow2\n    "
    cmd = ('qemu-img', 'convert', '-O', fmt, orig, dest)
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] == 0:
        return True
    else:
        return ret['stderr']