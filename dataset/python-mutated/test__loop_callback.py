from gevent import get_hub
from gevent import testing as greentest

class Test(greentest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        count = [0]

        def incr():
            if False:
                i = 10
                return i + 15
            count[0] += 1
        loop = get_hub().loop
        loop.run_callback(incr)
        loop.run()
        self.assertEqual(count, [1])
if __name__ == '__main__':
    greentest.main()