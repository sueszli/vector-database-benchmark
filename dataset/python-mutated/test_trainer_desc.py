"""
TestCases for TrainerDesc,
including config, etc.
"""
import unittest
from paddle import base

class TestTrainerDesc(unittest.TestCase):
    """TestCases for TrainerDesc."""

    def test_config(self):
        if False:
            print('Hello World!')
        '\n        Testcase for python config.\n        '
        trainer_desc = base.trainer_desc.TrainerDesc()
        trainer_desc._set_dump_fields(['a', 'b'])
        trainer_desc._set_mpi_rank(1)
        trainer_desc._set_dump_fields_path('path')
        dump_fields = trainer_desc.proto_desc.dump_fields
        mpi_rank = trainer_desc.proto_desc.mpi_rank
        dump_fields_path = trainer_desc.proto_desc.dump_fields_path
        self.assertEqual(len(dump_fields), 2)
        self.assertEqual(dump_fields[0], 'a')
        self.assertEqual(dump_fields[1], 'b')
        self.assertEqual(mpi_rank, 1)
        self.assertEqual(dump_fields_path, 'path')

    def test_config_dump_simple(self):
        if False:
            print('Hello World!')
        '\n        Testcase for dump_in_simple_mode\n        '
        trainer_desc = base.trainer_desc.TrainerDesc()
        trainer_desc._set_dump_fields(['a', 'b'])
        trainer_desc._set_is_dump_in_simple_mode(True)
        is_dump_in_simple_mode = trainer_desc.proto_desc.is_dump_in_simple_mode
        self.assertEqual(is_dump_in_simple_mode, 1)
if __name__ == '__main__':
    unittest.main()