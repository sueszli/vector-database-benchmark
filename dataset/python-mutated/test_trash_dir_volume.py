import unittest
from mock import Mock
from tests.support.fake_volume_of import fake_volume_of
from trashcli.put.trash_dir_volume_reader import TrashDirVolumeReader

class TestTrashDirVolume(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        volumes = fake_volume_of(['/disk1', '/disk2'])
        fs = Mock()
        fs.realpath = lambda path: path
        self.trash_dir_volume = TrashDirVolumeReader(volumes, fs)

    def test(self):
        if False:
            i = 10
            return i + 15
        result = self.trash_dir_volume.volume_of_trash_dir('/disk1/trash_dir_path')
        assert result == '/disk1'