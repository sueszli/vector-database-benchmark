import unittest

class TestAddReverse(unittest.TestCase):

    def test_add_reverse(self):
        if False:
            i = 10
            return i + 15
        print('Test: Empty list(s)')
        self.assertEqual(MyLinkedList().add_reverse(None, None), None)
        self.assertEqual(MyLinkedList().add_reverse(Node(5), None), None)
        self.assertEqual(MyLinkedList().add_reverse(None, Node(10)), None)
        print('Test: Add values of different lengths')
        first_list = MyLinkedList(Node(6))
        first_list.append(5)
        second_list = MyLinkedList(Node(9))
        second_list.append(8)
        second_list.append(7)
        result = MyLinkedList().add_reverse(first_list, second_list)
        self.assertEqual(result.get_all_data(), [5, 4, 8])
        print('Test: Add values of same lengths')
        first_head = Node(6)
        first_list = MyLinkedList(first_head)
        first_list.append(5)
        first_list.append(4)
        second_head = Node(9)
        second_list = MyLinkedList(second_head)
        second_list.append(8)
        second_list.append(7)
        result = MyLinkedList().add_reverse(first_list, second_list)
        self.assertEqual(result.get_all_data(), [5, 4, 2, 1])
        print('Success: test_add_reverse')

def main():
    if False:
        print('Hello World!')
    test = TestAddReverse()
    test.test_add_reverse()
if __name__ == '__main__':
    main()