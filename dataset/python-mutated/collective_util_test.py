"""Test for utilities for collectives."""
from tensorflow.python.distribute import collective_util
from tensorflow.python.eager import test

class OptionsTest(test.TestCase):

    def testCreateOptionsViaExportedAPI(self):
        if False:
            while True:
                i = 10
        options = collective_util._OptionsExported(bytes_per_pack=1)
        self.assertIsInstance(options, collective_util.Options)
        self.assertEqual(options.bytes_per_pack, 1)
        with self.assertRaises(ValueError):
            collective_util._OptionsExported(bytes_per_pack=-1)

    def testCreateOptionsViaHints(self):
        if False:
            i = 10
            return i + 15
        with self.assertLogs() as cm:
            options = collective_util.Hints(50, 1)
        self.assertTrue(any(('is deprecated' in msg for msg in cm.output)))
        self.assertIsInstance(options, collective_util.Options)
        self.assertEqual(options.bytes_per_pack, 50)
        self.assertEqual(options.timeout_seconds, 1)
if __name__ == '__main__':
    test.main()