import typing
import unittest
from decimal import Decimal
from borb.io.read.postfix.postfix_eval import PostScriptEval

class TestPostscriptEval(unittest.TestCase):
    """
    This test checks the PostscriptEval object, which is used to
    evaluate Function objects.
    """

    def test_postscripteval_evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        s: str = '\n        {\n            360 mul sin\n            2 div\n            exch 360 mul sin\n            2 div\n            add\n        }\n        '
        out: typing.List[Decimal] = PostScriptEval.evaluate(s, [Decimal(0.5), Decimal(0.4)])
        assert len(out) == 1
        assert abs(float(out[0]) - 0.6338117228265896) < 1 * 10 ** (-5)