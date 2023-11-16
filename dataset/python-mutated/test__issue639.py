import gevent
from gevent import testing as greentest

class Test(greentest.TestCase):

    def test(self):
        if False:
            return 10
        gevent.sleep()
        gevent.idle()
if __name__ == '__main__':
    greentest.main()