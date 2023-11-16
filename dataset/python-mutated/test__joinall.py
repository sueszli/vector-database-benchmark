import gevent
from gevent import testing as greentest

class Test(greentest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                while True:
                    i = 10
            pass
        a = gevent.spawn(func)
        b = gevent.spawn(func)
        gevent.joinall([a, b, a])
if __name__ == '__main__':
    greentest.main()