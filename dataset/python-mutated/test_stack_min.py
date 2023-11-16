import unittest

class TestStackMin(unittest.TestCase):

    def test_stack_min(self):
        if False:
            while True:
                i = 10
        print('Test: Push on empty stack, non-empty stack')
        stack = StackMin()
        stack.push(5)
        self.assertEqual(stack.peek(), 5)
        self.assertEqual(stack.minimum(), 5)
        stack.push(1)
        self.assertEqual(stack.peek(), 1)
        self.assertEqual(stack.minimum(), 1)
        stack.push(3)
        self.assertEqual(stack.peek(), 3)
        self.assertEqual(stack.minimum(), 1)
        stack.push(0)
        self.assertEqual(stack.peek(), 0)
        self.assertEqual(stack.minimum(), 0)
        print('Test: Pop on non-empty stack')
        self.assertEqual(stack.pop(), 0)
        self.assertEqual(stack.minimum(), 1)
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.minimum(), 1)
        self.assertEqual(stack.pop(), 1)
        self.assertEqual(stack.minimum(), 5)
        self.assertEqual(stack.pop(), 5)
        self.assertEqual(stack.minimum(), sys.maxsize)
        print('Test: Pop empty stack')
        self.assertEqual(stack.pop(), None)
        print('Success: test_stack_min')

def main():
    if False:
        i = 10
        return i + 15
    test = TestStackMin()
    test.test_stack_min()
if __name__ == '__main__':
    main()