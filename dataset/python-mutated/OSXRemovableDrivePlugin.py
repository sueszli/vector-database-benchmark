from . import RemovableDrivePlugin
import subprocess
import os
import plistlib

class OSXRemovableDrivePlugin(RemovableDrivePlugin.RemovableDrivePlugin):
    """Support for removable devices on Mac OSX"""

    def checkRemovableDrives(self):
        if False:
            return 10
        drives = {}
        p = subprocess.Popen(['system_profiler', 'SPUSBDataType', '-xml'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        plist = plistlib.loads(p.communicate()[0])
        result = self._recursiveSearch(plist, 'removable_media')
        p = subprocess.Popen(['system_profiler', 'SPCardReaderDataType', '-xml'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        plist = plistlib.loads(p.communicate()[0])
        result.extend(self._recursiveSearch(plist, 'removable_media'))
        for drive in result:
            if drive['removable_media'] != 'yes':
                continue
            if 'volumes' not in drive or not drive['volumes']:
                continue
            for volume in drive['volumes']:
                if not 'mount_point' in volume:
                    continue
                mount_point = volume['mount_point']
                if '_name' in volume:
                    drive_name = volume['_name']
                else:
                    drive_name = os.path.basename(mount_point)
                drives[mount_point] = drive_name
        return drives

    def performEjectDevice(self, device):
        if False:
            while True:
                i = 10
        p = subprocess.Popen(['diskutil', 'eject', device.getId()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()
        return_code = p.wait()
        if return_code != 0:
            return False
        else:
            return True

    def _recursiveSearch(self, plist, key):
        if False:
            return 10
        result = []
        for entry in plist:
            if key in entry:
                result.append(entry)
                continue
            if '_items' in entry:
                result.extend(self._recursiveSearch(entry['_items'], key))
            if 'Media' in entry:
                result.extend(self._recursiveSearch(entry['Media'], key))
        return result