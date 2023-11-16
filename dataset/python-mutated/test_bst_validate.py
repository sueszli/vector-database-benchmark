import unittest

class TestBstValidate(unittest.TestCase):

    def test_bst_validate_empty(self):
        if False:
            i = 10
            return i + 15
        bst = BstValidate(None)
        bst.validate()

    def test_bst_validate(self):
        if False:
            for i in range(10):
                print('nop')
        bst = BstValidate(Node(5))
        bst.insert(8)
        bst.insert(5)
        bst.insert(6)
        bst.insert(4)
        bst.insert(7)
        self.assertEqual(bst.validate(), True)
        bst = BstValidate(Node(5))
        left = Node(5)
        right = Node(8)
        invalid = Node(20)
        bst.root.left = left
        bst.root.right = right
        bst.root.left.right = invalid
        self.assertEqual(bst.validate(), False)
        print('Success: test_bst_validate')

def main():
    if False:
        for i in range(10):
            print('nop')
    test = TestBstValidate()
    test.assertRaises(TypeError, test.test_bst_validate_empty)
    test.test_bst_validate()
if __name__ == '__main__':
    main()