from gevent import monkey
monkey.patch_all()
import select
import gevent.testing as greentest

class TestSelect(greentest.TestCase):

    def _make_test(name, ns):
        if False:
            return 10

        def test(self):
            if False:
                while True:
                    i = 10
            self.assertIs(getattr(select, name, self), self)
            self.assertFalse(hasattr(select, name))
        test.__name__ = 'test_' + name + '_removed'
        ns[test.__name__] = test
    for name in ('epoll', 'kqueue', 'kevent', 'devpoll'):
        _make_test(name, locals())
    del name
    del _make_test
if __name__ == '__main__':
    greentest.main()