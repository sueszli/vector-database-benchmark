import unittest
from trashcli.fstab.mount_points_listing import FakeMountPointsListing
from trashcli.fstab.volume_listing import VolumesListingImpl

class TestVolumesListingImpl(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.volumes_listing = VolumesListingImpl(FakeMountPointsListing(['/os-vol1', '/os-vol2']))

    def test_os_mount_points(self):
        if False:
            while True:
                i = 10
        result = self.volumes_listing.list_volumes({})
        assert list(result) == ['/os-vol1', '/os-vol2']

    def test_one_vol_from_environ(self):
        if False:
            while True:
                i = 10
        result = self.volumes_listing.list_volumes({'TRASH_VOLUMES': '/fake-vol1'})
        assert list(result) == ['/fake-vol1']

    def test_multiple_vols_from_environ(self):
        if False:
            i = 10
            return i + 15
        result = self.volumes_listing.list_volumes({'TRASH_VOLUMES': '/fake-vol1:/fake-vol2:/fake-vol3'})
        assert list(result) == ['/fake-vol1', '/fake-vol2', '/fake-vol3']

    def test_empty_environ(self):
        if False:
            return 10
        result = self.volumes_listing.list_volumes({'TRASH_VOLUMES': ''})
        assert list(result) == ['/os-vol1', '/os-vol2']

    def test_skip_empty_vol(self):
        if False:
            return 10
        result = self.volumes_listing.list_volumes({'TRASH_VOLUMES': '/vol1::/vol2'})
        assert list(result) == ['/vol1', '/vol2']