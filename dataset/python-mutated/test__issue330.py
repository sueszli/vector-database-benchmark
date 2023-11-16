import gevent
import gevent.testing as greentest

class MyException(Exception):
    pass

class TestSwitch(greentest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestSwitch, self).setUp()
        self.switched_to = [False, False]
        self.caught = None

    def should_never_run(self, i):
        if False:
            i = 10
            return i + 15
        self.switched_to[i] = True

    def check(self, g, g2):
        if False:
            i = 10
            return i + 15
        gevent.joinall((g, g2))
        self.assertEqual([False, False], self.switched_to)
        self.assertIsInstance(g.value, gevent.GreenletExit)
        self.assertIsInstance(g2.value, gevent.GreenletExit)
        self.assertIsNone(g.exc_info)
        self.assertIsNone(g2.exc_info)
        self.assertIsNone(g.exception)
        self.assertIsNone(g2.exception)

    def test_gevent_kill(self):
        if False:
            print('Hello World!')
        g = gevent.spawn(self.should_never_run, 0)
        g2 = gevent.spawn(self.should_never_run, 1)
        gevent.kill(g)
        gevent.kill(g2)
        self.check(g, g2)

    def test_greenlet_kill(self):
        if False:
            i = 10
            return i + 15
        g = gevent.spawn(self.should_never_run, 0)
        g2 = gevent.spawn(self.should_never_run, 1)
        g.kill()
        g2.kill()
        self.check(g, g2)

    def test_throw(self):
        if False:
            return 10
        g = gevent.spawn(self.should_never_run, 0)
        g2 = gevent.spawn(self.should_never_run, 1)
        g.throw(gevent.GreenletExit)
        g2.throw(gevent.GreenletExit)
        self.check(g, g2)

    def catcher(self):
        if False:
            i = 10
            return i + 15
        try:
            while True:
                gevent.sleep(0)
        except MyException as e:
            self.caught = e

    def test_kill_exception(self):
        if False:
            for i in range(10):
                print('nop')
        g = gevent.spawn(self.catcher)
        g.start()
        gevent.sleep()
        gevent.kill(g, MyException())
        gevent.sleep()
        self.assertIsInstance(self.caught, MyException)
        self.assertIsNone(g.exception, MyException)
if __name__ == '__main__':
    greentest.main()