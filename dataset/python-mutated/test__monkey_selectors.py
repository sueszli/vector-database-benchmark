try:
    import selectors
except ImportError:
    import selectors2 as selectors
from gevent.monkey import patch_all
import gevent.testing as greentest
patch_all()
from gevent.selectors import DefaultSelector
from gevent.selectors import GeventSelector
from gevent.tests.test__selectors import SelectorTestMixin

class TestSelectors(SelectorTestMixin, greentest.TestCase):

    @greentest.skipOnWindows('SelectSelector._select is a normal function on Windows')
    def test_selectors_select_is_patched(self):
        if False:
            return 10
        _select = selectors.SelectSelector._select
        self.assertIn('_gevent_monkey', dir(_select))

    def test_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(DefaultSelector, GeventSelector)
        self.assertIs(selectors.DefaultSelector, GeventSelector)

    def test_import_selectors(self):
        if False:
            return 10
        __import__('selectors')

    def _make_test(name, kind):
        if False:
            while True:
                i = 10
        if kind is None:

            def m(self):
                if False:
                    i = 10
                    return i + 15
                self.skipTest(name + ' is not defined')
        else:

            def m(self, k=kind):
                if False:
                    while True:
                        i = 10
                with k() as sel:
                    self._check_selector(sel)
        m.__name__ = 'test_selector_' + name
        return m
    SelKind = SelKindName = None
    for SelKindName in ('KqueueSelector', 'EpollSelector', 'DevpollSelector', 'PollSelector', 'SelectSelector', GeventSelector):
        if not isinstance(SelKindName, type):
            SelKind = getattr(selectors, SelKindName, None)
        else:
            SelKind = SelKindName
            SelKindName = SelKind.__name__
        m = _make_test(SelKindName, SelKind)
        locals()[m.__name__] = m
    del SelKind
    del SelKindName
    del _make_test
if __name__ == '__main__':
    greentest.main()