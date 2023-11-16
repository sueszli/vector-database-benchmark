import unittest
from tests.support.fake_volume_of import fake_volume_of

class Test_create_fake_volume_of(unittest.TestCase):

    def test_return_the_containing_volume(self):
        if False:
            return 10
        self.volumes = fake_volume_of(['/fake-vol'])
        assert '/fake-vol' == self.volumes.volume_of('/fake-vol/foo')

    def test_with_file_that_are_outside(self):
        if False:
            for i in range(10):
                print('nop')
        self.volumes = fake_volume_of(['/fake-vol'])
        assert '/' == self.volumes.volume_of('/foo')

    def test_it_work_also_with_relative_mount_point(self):
        if False:
            i = 10
            return i + 15
        self.volumes = fake_volume_of(['relative-fake-vol'])
        assert 'relative-fake-vol' == self.volumes.volume_of('relative-fake-vol/foo')