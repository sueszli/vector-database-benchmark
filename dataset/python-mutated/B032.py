"""
Should emit:
B032 - on lines 9, 10, 12, 13, 16-19
"""
dct = {'a': 1}
dct['b']: 2
dct.b: 2
dct['b']: 'test'
dct.b: 'test'
test = 'test'
dct['b']: test
dct['b']: test.lower()
dct.b: test
dct.b: test.lower()
typed_dct: dict[str, int] = {'a': 1}
typed_dct['b'] = 2
typed_dct.b = 2

class TestClass:

    def test_self(self):
        if False:
            i = 10
            return i + 15
        self.test: int