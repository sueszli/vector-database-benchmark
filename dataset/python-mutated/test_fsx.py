import unittest
from troposphere.fsx import FileSystem

class TestFSx(unittest.TestCase):

    def test_FileSystem(self):
        if False:
            return 10
        FileSystem('filesystem', FileSystemType='type', StorageType='HDD', SubnetIds=['subnet']).to_dict()

    def test_invalid_storagetype(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            FileSystem('filesystem', FileSystemType='type', StorageType='floppy', SubnetIds=['subnet']).to_dict()
if __name__ == '__main__':
    unittest.main()