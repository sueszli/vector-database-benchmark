import os
from abc import ABCMeta, abstractmethod
import six
from trashcli.fstab.mount_points_listing import MountPointsListing, RealMountPointsListing

@six.add_metaclass(ABCMeta)
class VolumesListing:

    @abstractmethod
    def list_volumes(self, environ):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class RealVolumesListing(VolumesListing):

    def list_volumes(self, environ):
        if False:
            while True:
                i = 10
        return VolumesListingImpl(RealMountPointsListing()).list_volumes(environ)

class VolumesListingImpl:

    def __init__(self, mount_points_listing):
        if False:
            while True:
                i = 10
        self.mount_points_listing = mount_points_listing

    def list_volumes(self, environ):
        if False:
            i = 10
            return i + 15
        if 'TRASH_VOLUMES' in environ and environ['TRASH_VOLUMES']:
            return [vol for vol in environ['TRASH_VOLUMES'].split(':') if vol != '']
        return self.mount_points_listing.list_mount_points()

class NoVolumesListing(VolumesListing):

    def list_volumes(self, environ):
        if False:
            for i in range(10):
                print('nop')
        return []

class RealIsMount:

    def is_mount(self, path):
        if False:
            return 10
        return os.path.ismount(path)