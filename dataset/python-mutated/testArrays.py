from win32com.client import gencache
from win32com.test import util
ZeroD = 0
OneDEmpty = []
OneD = [1, 2, 3]
TwoD = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
TwoD1 = [[[1, 2, 3, 5], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
OneD1 = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]
OneD2 = [[1, 2, 3], [1, 2, 3, 4, 5], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]
ThreeD = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
FourD = [[[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]], [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]]
LargeD = [[[list(range(10))] * 10]] * 512

def _normalize_array(a):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(a, tuple):
        return a
    ret = []
    for i in a:
        ret.append(_normalize_array(i))
    return ret

class ArrayTest(util.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.arr = gencache.EnsureDispatch('PyCOMTest.ArrayTest')

    def tearDown(self):
        if False:
            return 10
        self.arr = None

    def _doTest(self, array):
        if False:
            i = 10
            return i + 15
        self.arr.Array = array
        self.assertEqual(_normalize_array(self.arr.Array), array)

    def testZeroD(self):
        if False:
            for i in range(10):
                print('nop')
        self._doTest(ZeroD)

    def testOneDEmpty(self):
        if False:
            while True:
                i = 10
        self._doTest(OneDEmpty)

    def testOneD(self):
        if False:
            while True:
                i = 10
        self._doTest(OneD)

    def testTwoD(self):
        if False:
            for i in range(10):
                print('nop')
        self._doTest(TwoD)

    def testThreeD(self):
        if False:
            i = 10
            return i + 15
        self._doTest(ThreeD)

    def testFourD(self):
        if False:
            i = 10
            return i + 15
        self._doTest(FourD)

    def testTwoD1(self):
        if False:
            while True:
                i = 10
        self._doTest(TwoD1)

    def testOneD1(self):
        if False:
            print('Hello World!')
        self._doTest(OneD1)

    def testOneD2(self):
        if False:
            while True:
                i = 10
        self._doTest(OneD2)

    def testLargeD(self):
        if False:
            while True:
                i = 10
        self._doTest(LargeD)
if __name__ == '__main__':
    try:
        util.testmain()
    except SystemExit as rc:
        if not rc:
            raise