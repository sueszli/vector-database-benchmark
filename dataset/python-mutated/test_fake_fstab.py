import unittest
from tests.support.fake_volume_of import fake_volume_of

class TestFakeFstab(unittest.TestCase):

    def test_default(self):
        if False:
            while True:
                i = 10
        self.volumes = fake_volume_of([])
        assert ['/'] == self.filter_only_mount_points('/')

    def test_it_should_accept_fake_mount_points(self):
        if False:
            i = 10
            return i + 15
        self.volumes = fake_volume_of(['/fake'])
        assert ['/', '/fake'] == self.filter_only_mount_points('/', '/fake')

    def test_something(self):
        if False:
            i = 10
            return i + 15
        volumes = fake_volume_of(['/fake'])
        assert '/fake' == volumes.volume_of('/fake/foo')

    def filter_only_mount_points(self, *supposed_mount_points):
        if False:
            for i in range(10):
                print('nop')
        return [mp for mp in supposed_mount_points if mp == self.volumes.volume_of(mp)]