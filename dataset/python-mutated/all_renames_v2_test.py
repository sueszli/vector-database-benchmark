"""Tests for all_renames_v2."""
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import all_renames_v2

class AllRenamesV2Test(test_util.TensorFlowTestCase):

    def test_no_identity_renames(self):
        if False:
            return 10
        identity_renames = [old_name for (old_name, new_name) in all_renames_v2.symbol_renames.items() if old_name == new_name]
        self.assertEmpty(identity_renames)
if __name__ == '__main__':
    test_lib.main()