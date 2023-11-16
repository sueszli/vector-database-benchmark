import time
from hypothesis.strategies._internal import SearchStrategy

class _Slow(SearchStrategy):

    def do_draw(self, data):
        if False:
            while True:
                i = 10
        time.sleep(1.0)
        data.draw_bytes(2)
        return None
SLOW = _Slow()

class HardToShrink(SearchStrategy):

    def __init__(self):
        if False:
            return 10
        self.__last = None
        self.accepted = set()

    def do_draw(self, data):
        if False:
            return 10
        x = bytes((data.draw_bits(8) for _ in range(100)))
        if x in self.accepted:
            return True
        ls = self.__last
        if ls is None:
            if all(x):
                self.__last = x
                self.accepted.add(x)
                return True
            else:
                return False
        diffs = [i for i in range(len(x)) if x[i] != ls[i]]
        if len(diffs) == 1:
            i = diffs[0]
            if x[i] + 1 == ls[i]:
                self.__last = x
                self.accepted.add(x)
                return True
        return False