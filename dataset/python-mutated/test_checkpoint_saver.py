import unittest
from paddle.base.incubate.checkpoint.checkpoint_saver import CheckpointSaver
from paddle.distributed.fleet.utils.fs import HDFSClient

class CheckpointerSaverTest(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        fs = HDFSClient('/usr/local/hadoop-2.7.7', None)
        dir_path = './checkpointsaver_test'
        fs.delete(dir_path)
        s = CheckpointSaver(fs)
        fs.mkdirs(f'{dir_path}/exe.exe')
        fs.mkdirs(f'{dir_path}/exe.1')
        fs.mkdirs(f'{dir_path}/exe')
        a = s.get_checkpoint_no(dir_path)
        self.assertEqual(len(a), 0)
        fs.mkdirs(f'{dir_path}/__paddle_checkpoint__.0')
        fs.mkdirs(f'{dir_path}/__paddle_checkpoint__.exe')
        a = s.get_checkpoint_no(dir_path)
        self.assertEqual(len(a), 1)
        s.clean_redundant_checkpoints(dir_path)
        s.clean_redundant_checkpoints(dir_path)
        fs.delete(dir_path)
if __name__ == '__main__':
    unittest.main()