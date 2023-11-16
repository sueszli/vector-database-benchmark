import unittest
from paddle.base.framework import Program

class TestDebugStringFramework(unittest.TestCase):

    def test_debug_str(self):
        if False:
            return 10
        p = Program()
        p.current_block().create_var(name='t', shape=[0, 1])
        self.assertRaises(ValueError, p.to_string, True)
if __name__ == '__main__':
    unittest.main()