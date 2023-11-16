import unittest
import step2_eval

class TestStep3(unittest.TestCase):

    def test_step3_let_multiple(self):
        if False:
            return 10
        self.assertEqual('{"a" 15}', step2_eval.rep('{"a" (+ 7 8)} '))
if __name__ == '__main__':
    unittest.main()