import unittest
from algorithms.linkedlist import reverse_list, reverse_list_recursive, is_sorted, remove_range, swap_pairs, rotate_right, is_cyclic, merge_two_list, merge_two_list_recur, is_palindrome, is_palindrome_stack, is_palindrome_dict, RandomListNode, copy_random_pointer_v1, copy_random_pointer_v2

class Node(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.next = None

def convert(head):
    if False:
        while True:
            i = 10
    ret = []
    if head:
        current = head
        while current:
            ret.append(current.val)
            current = current.next
    return ret

class TestSuite(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.l = Node('A')
        self.l.next = Node('B')
        self.l.next.next = Node('C')
        self.l.next.next.next = Node('B')
        self.l.next.next.next.next = Node('A')
        self.l1 = Node('A')
        self.l1.next = Node('B')
        self.l1.next.next = Node('C')
        self.l1.next.next.next = Node('B')

    def test_reverse_list(self):
        if False:
            for i in range(10):
                print('nop')
        head = Node(1)
        head.next = Node(2)
        head.next.next = Node(3)
        head.next.next.next = Node(4)
        self.assertEqual([4, 3, 2, 1], convert(reverse_list(head)))
        head = Node(1)
        head.next = Node(2)
        head.next.next = Node(3)
        head.next.next.next = Node(4)
        self.assertEqual([4, 3, 2, 1], convert(reverse_list_recursive(head)))

    def test_is_sorted(self):
        if False:
            i = 10
            return i + 15
        head = Node(-2)
        head.next = Node(2)
        head.next.next = Node(2)
        head.next.next.next = Node(4)
        head.next.next.next.next = Node(9)
        self.assertTrue(is_sorted(head))
        head = Node(1)
        head.next = Node(2)
        head.next.next = Node(8)
        head.next.next.next = Node(4)
        head.next.next.next.next = Node(6)
        self.assertFalse(is_sorted(head))

    def test_remove_range(self):
        if False:
            return 10
        head = Node(0)
        head.next = Node(1)
        head.next.next = Node(2)
        head.next.next.next = Node(3)
        head.next.next.next.next = Node(4)
        self.assertEqual([0, 4], convert(remove_range(head, 1, 3)))
        head = Node(0)
        head.next = Node(1)
        head.next.next = Node(2)
        head.next.next.next = Node(3)
        head.next.next.next.next = Node(4)
        self.assertEqual([2, 3, 4], convert(remove_range(head, 0, 1)))
        head = Node(0)
        head.next = Node(1)
        head.next.next = Node(2)
        head.next.next.next = Node(3)
        head.next.next.next.next = Node(4)
        self.assertEqual([], convert(remove_range(head, 0, 7)))

    def test_swap_in_pairs(self):
        if False:
            i = 10
            return i + 15
        head = Node(1)
        head.next = Node(2)
        head.next.next = Node(3)
        head.next.next.next = Node(4)
        self.assertEqual([2, 1, 4, 3], convert(swap_pairs(head)))

    def test_rotate_right(self):
        if False:
            for i in range(10):
                print('nop')
        head = Node(1)
        head.next = Node(2)
        head.next.next = Node(3)
        head.next.next.next = Node(4)
        head.next.next.next.next = Node(5)
        self.assertEqual([4, 5, 1, 2, 3], convert(rotate_right(head, 2)))

    def test_is_cyclic(self):
        if False:
            while True:
                i = 10
        head = Node('A')
        head.next = Node('B')
        curr = head.next
        cyclic_node = Node('C')
        curr.next = cyclic_node
        curr = curr.next
        curr.next = Node('D')
        curr = curr.next
        curr.next = Node('E')
        curr = curr.next
        curr.next = cyclic_node
        self.assertTrue(is_cyclic(head))
        head = Node(1)
        curr = head
        for i in range(2, 6):
            curr.next = Node(i)
            curr = curr.next
        self.assertFalse(is_cyclic(head))

    def test_merge_two_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Input: head1:1->2->4, head2: 1->3->4\n        Output: 1->1->2->3->4->4\n        '
        head1 = Node(1)
        head1.next = Node(2)
        head1.next.next = Node(4)
        head2 = Node(1)
        head2.next = Node(3)
        head2.next.next = Node(4)
        self.assertEqual([1, 1, 2, 3, 4, 4], convert(merge_two_list(head1, head2)))
        head1 = Node(1)
        head1.next = Node(2)
        head1.next.next = Node(4)
        head2 = Node(1)
        head2.next = Node(3)
        head2.next.next = Node(4)
        self.assertEqual([1, 1, 2, 3, 4, 4], convert(merge_two_list_recur(head1, head2)))

    def test_is_palindrome(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(is_palindrome(self.l))
        self.assertFalse(is_palindrome(self.l1))

    def test_is_palindrome_stack(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(is_palindrome_stack(self.l))
        self.assertFalse(is_palindrome_stack(self.l1))

    def test_is_palindrome_dict(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(is_palindrome_dict(self.l))
        self.assertFalse(is_palindrome_dict(self.l1))

    def test_solution_0(self):
        if False:
            i = 10
            return i + 15
        self._init_random_list_nodes()
        result = copy_random_pointer_v1(self.random_list_node1)
        self._assert_is_a_copy(result)

    def test_solution_1(self):
        if False:
            for i in range(10):
                print('nop')
        self._init_random_list_nodes()
        result = copy_random_pointer_v2(self.random_list_node1)
        self._assert_is_a_copy(result)

    def _assert_is_a_copy(self, result):
        if False:
            return 10
        self.assertEqual(5, result.next.next.next.next.label)
        self.assertEqual(4, result.next.next.next.label)
        self.assertEqual(3, result.next.next.label)
        self.assertEqual(2, result.next.label)
        self.assertEqual(1, result.label)
        self.assertEqual(3, result.next.next.next.next.random.label)
        self.assertIsNone(result.next.next.next.random)
        self.assertEqual(2, result.next.next.random.label)
        self.assertEqual(5, result.next.random.label)
        self.assertEqual(4, result.random.label)

    def _init_random_list_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        self.random_list_node1 = RandomListNode(1)
        random_list_node2 = RandomListNode(2)
        random_list_node3 = RandomListNode(3)
        random_list_node4 = RandomListNode(4)
        random_list_node5 = RandomListNode(5)
        (self.random_list_node1.next, self.random_list_node1.random) = (random_list_node2, random_list_node4)
        (random_list_node2.next, random_list_node2.random) = (random_list_node3, random_list_node5)
        (random_list_node3.next, random_list_node3.random) = (random_list_node4, random_list_node2)
        random_list_node4.next = random_list_node5
        random_list_node5.random = random_list_node3
if __name__ == '__main__':
    unittest.main()