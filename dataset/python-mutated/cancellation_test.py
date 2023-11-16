from tensorflow.python.eager import cancellation
from tensorflow.python.platform import test

class CancellationTest(test.TestCase):

    def testStartCancel(self):
        if False:
            print('Hello World!')
        manager = cancellation.CancellationManager()
        self.assertFalse(manager.is_cancelled)
        manager.start_cancel()
        self.assertTrue(manager.is_cancelled)
if __name__ == '__main__':
    test.main()