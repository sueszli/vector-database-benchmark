import github
from . import Framework

class Issue142(Framework.TestCase):

    def testDecodeJson(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(github.Github().get_rate_limit().core.limit, 60)