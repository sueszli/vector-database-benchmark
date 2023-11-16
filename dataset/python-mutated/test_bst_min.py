import unittest

def height(node):
    if False:
        while True:
            i = 10
    if node is None:
        return 0
    return 1 + max(height(node.left), height(node.right))

class TestBstMin(unittest.TestCase):

    def test_bst_min(self):
        if False:
            for i in range(10):
                print('nop')
        min_bst = MinBst()
        array = [0, 1, 2, 3, 4, 5, 6]
        root = min_bst.create_min_bst(array)
        self.assertEqual(height(root), 3)
        min_bst = MinBst()
        array = [0, 1, 2, 3, 4, 5, 6, 7]
        root = min_bst.create_min_bst(array)
        self.assertEqual(height(root), 4)
        print('Success: test_bst_min')

def main():
    if False:
        print('Hello World!')
    test = TestBstMin()
    test.test_bst_min()
if __name__ == '__main__':
    main()