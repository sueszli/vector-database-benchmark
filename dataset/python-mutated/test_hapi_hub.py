import os
import unittest
import numpy as np
import paddle
from paddle.hapi import hub

class TestHub(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.local_repo = os.path.dirname(os.path.abspath(__file__))
        self.github_repo = 'lyuwenyu/paddlehub_demo:main'

    def testLoad(self):
        if False:
            i = 10
            return i + 15
        model = hub.load(self.local_repo, model='MM', source='local', out_channels=8)
        data = paddle.rand((1, 3, 100, 100))
        out = model(data)
        np.testing.assert_equal(out.shape, [1, 8, 50, 50])
        model = hub.load(self.github_repo, model='MM', source='github', force_reload=True)
        model = hub.load(self.github_repo, model='MM', source='github', force_reload=False, pretrained=False)
        model = hub.load(self.github_repo.split(':')[0], model='MM', source='github', force_reload=False, pretrained=False)
        model = hub.load(self.github_repo, model='MM', source='github', force_reload=False, pretrained=True, out_channels=8)
        data = paddle.ones((1, 3, 2, 2))
        out = model(data)
        gt = np.array([1.53965068, 0.0, 0.0, 1.39455748, 0.720662, 0.1977303, 2.09201908, 0.37345418])
        np.testing.assert_equal(out.shape, [1, 8, 1, 1])
        np.testing.assert_almost_equal(out.numpy(), gt.reshape(1, 8, 1, 1), decimal=5)

    def testHelp(self):
        if False:
            return 10
        docs1 = hub.help(self.local_repo, model='MM', source='local')
        docs2 = hub.help(self.github_repo, model='MM', source='github', force_reload=False)
        assert docs1 == docs2 == 'This is a test demo for paddle hub\n    ', ''

    def testList(self):
        if False:
            while True:
                i = 10
        models1 = hub.list(self.local_repo, source='local', force_reload=False)
        models2 = hub.list(self.github_repo, source='github', force_reload=False)
        assert models1 == models2 == ['MM'], ''

    def testExcept(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            _ = hub.help(self.github_repo, model='MM', source='github-test', force_reload=False)
        with self.assertRaises(ValueError):
            _ = hub.load(self.github_repo, model='MM', source='github-test', force_reload=False)
        with self.assertRaises(ValueError):
            _ = hub.list(self.github_repo, source='github-test', force_reload=False)
        with self.assertRaises(ValueError):
            _ = hub.load(self.local_repo, model=123, source='local', force_reload=False)
        with self.assertRaises(RuntimeError):
            _ = hub.load(self.local_repo, model='123', source='local', force_reload=False)
if __name__ == '__main__':
    unittest.main()