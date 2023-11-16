import unittest
import step7_quote

class TestStep7(unittest.TestCase):

    def test_step7_cons(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('(1)', step7_quote.rep('(cons 1 (list))'))

    def test_step7_concat(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('()', step7_quote.rep('(concat)'))

    def test_step7_quote(self):
        if False:
            print('Hello World!')
        self.assertEqual('(+ 1 2)', step7_quote.rep('(quote (+ 1 2))'))

    def test_step7_quasiquote(self):
        if False:
            while True:
                i = 10
        self.assertEqual('(+ 1 3)', step7_quote.rep('(quasiquote (+ 1 (unquote (+ 1 2))))'))

    def test_step7_quasiquote_advanced(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('(2)', step7_quote.rep("(def! c '(2))"))
        self.assertEqual('(1 2 3)', step7_quote.rep('`[1 ~@c 3]'))
if __name__ == '__main__':
    unittest.main()