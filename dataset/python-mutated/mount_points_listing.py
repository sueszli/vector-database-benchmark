import os
from abc import ABCMeta, abstractmethod
import six

@six.add_metaclass(ABCMeta)
class MountPointsListing:

    @abstractmethod
    def list_mount_points(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class RealMountPointsListing(MountPointsListing):

    def list_mount_points(self):
        if False:
            for i in range(10):
                print('nop')
        return os_mount_points()

class FakeMountPointsListing(MountPointsListing):

    def __init__(self, mount_points):
        if False:
            print('Hello World!')
        self.mount_points = mount_points

    def set_mount_points(self, mount_points):
        if False:
            for i in range(10):
                print('nop')
        self.mount_points = mount_points

    def list_mount_points(self):
        if False:
            while True:
                i = 10
        return self.mount_points

def os_mount_points():
    if False:
        i = 10
        return i + 15
    import psutil
    fstypes = ['nfs', 'nfs4', 'p9', 'btrfs', 'fuse', 'fuse.glusterfs', 'fuse.mergerfs']
    fstypes += set([p.fstype for p in psutil.disk_partitions()])
    partitions = Partitions(fstypes)
    for p in psutil.disk_partitions(all=True):
        if os.path.isdir(p.mountpoint) and partitions.should_used_by_trashcli(p):
            yield p.mountpoint

class Partitions:

    def __init__(self, physical_fstypes):
        if False:
            while True:
                i = 10
        self.physical_fstypes = physical_fstypes

    def should_used_by_trashcli(self, partition):
        if False:
            i = 10
            return i + 15
        if (partition.device, partition.mountpoint, partition.fstype) == ('tmpfs', '/tmp', 'tmpfs'):
            return True
        return partition.fstype in self.physical_fstypes