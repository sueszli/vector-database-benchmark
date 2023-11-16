from . import RemovableDrivePlugin
from UM.Logger import Logger
import glob
import os
import subprocess

class LinuxRemovableDrivePlugin(RemovableDrivePlugin.RemovableDrivePlugin):
    """Support for removable devices on Linux.

    TODO: This code uses the most basic interfaces for handling this.
    We should instead use UDisks2 to handle mount/unmount and hotplugging events.
    """

    def checkRemovableDrives(self):
        if False:
            return 10
        drives = {}
        for volume in glob.glob('/media/*'):
            if os.path.ismount(volume):
                drives[volume] = os.path.basename(volume)
            elif volume == '/media/' + os.getenv('USER'):
                for volume in glob.glob('/media/' + os.getenv('USER') + '/*'):
                    if os.path.ismount(volume):
                        drives[volume] = os.path.basename(volume)
        for volume in glob.glob('/run/media/' + os.getenv('USER') + '/*'):
            if os.path.ismount(volume):
                drives[volume] = os.path.basename(volume)
        return drives

    def performEjectDevice(self, device):
        if False:
            for i in range(10):
                print('nop')
        p = subprocess.Popen(['umount', device.getId()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = p.communicate()
        Logger.log('d', 'umount returned: %s.', repr(output))
        return_code = p.wait()
        if return_code != 0:
            return False
        else:
            return True