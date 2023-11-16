import unittest

class TestDaliTfPluginLoadOk(unittest.TestCase):

    def test_import_dali_tf_ok(self):
        if False:
            for i in range(10):
                print('nop')
        import nvidia.dali.plugin.tf as dali_tf
        assert True

class TestDaliTfPluginLoadFail(unittest.TestCase):

    def test_import_dali_tf_load_fail(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception):
            import nvidia.dali.plugin.tf as dali_tf
if __name__ == '__main__':
    unittest.main()