import unittest

class TestHeight(unittest.TestCase):

    def test_height(self):
        if False:
            return 10
        bst = BstHeight(Node(5))
        self.assertEqual(bst.height(bst.root), 1)
        bst.insert(2)
        bst.insert(8)
        bst.insert(1)
        bst.insert(3)
        self.assertEqual(bst.height(bst.root), 3)
        print('Success: test_height')

def main():
    if False:
        i = 10
        return i + 15
    test = TestHeight()
    test.test_height()
if __name__ == '__main__':
    main()