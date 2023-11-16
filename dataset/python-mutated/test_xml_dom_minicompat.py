import copy
import pickle
import unittest
import xml.dom
from xml.dom.minicompat import *

class EmptyNodeListTestCase(unittest.TestCase):
    """Tests for the EmptyNodeList class."""

    def test_emptynodelist_item(self):
        if False:
            i = 10
            return i + 15
        node_list = EmptyNodeList()
        self.assertIsNone(node_list.item(0))
        self.assertIsNone(node_list.item(-1))
        with self.assertRaises(IndexError):
            node_list[0]
        with self.assertRaises(IndexError):
            node_list[-1]

    def test_emptynodelist_length(self):
        if False:
            while True:
                i = 10
        node_list = EmptyNodeList()
        self.assertEqual(node_list.length, 0)
        with self.assertRaises(xml.dom.NoModificationAllowedErr):
            node_list.length = 111

    def test_emptynodelist___add__(self):
        if False:
            for i in range(10):
                print('nop')
        node_list = EmptyNodeList() + NodeList()
        self.assertEqual(node_list, NodeList())

    def test_emptynodelist___radd__(self):
        if False:
            while True:
                i = 10
        node_list = [1, 2] + EmptyNodeList()
        self.assertEqual(node_list, [1, 2])

class NodeListTestCase(unittest.TestCase):
    """Tests for the NodeList class."""

    def test_nodelist_item(self):
        if False:
            return 10
        node_list = NodeList()
        self.assertIsNone(node_list.item(0))
        self.assertIsNone(node_list.item(-1))
        with self.assertRaises(IndexError):
            node_list[0]
        with self.assertRaises(IndexError):
            node_list[-1]
        node_list.append(111)
        node_list.append(999)
        self.assertEqual(node_list.item(0), 111)
        self.assertIsNone(node_list.item(-1))
        self.assertEqual(node_list[0], 111)
        self.assertEqual(node_list[-1], 999)

    def test_nodelist_length(self):
        if False:
            for i in range(10):
                print('nop')
        node_list = NodeList([1, 2])
        self.assertEqual(node_list.length, 2)
        with self.assertRaises(xml.dom.NoModificationAllowedErr):
            node_list.length = 111

    def test_nodelist___add__(self):
        if False:
            for i in range(10):
                print('nop')
        node_list = NodeList([3, 4]) + [1, 2]
        self.assertEqual(node_list, NodeList([3, 4, 1, 2]))

    def test_nodelist___radd__(self):
        if False:
            print('Hello World!')
        node_list = [1, 2] + NodeList([3, 4])
        self.assertEqual(node_list, NodeList([1, 2, 3, 4]))

    def test_nodelist_pickle_roundtrip(self):
        if False:
            i = 10
            return i + 15
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            node_list = NodeList()
            pickled = pickle.dumps(node_list, proto)
            unpickled = pickle.loads(pickled)
            self.assertIsNot(unpickled, node_list)
            self.assertEqual(unpickled, node_list)
            node_list.append(1)
            node_list.append(2)
            pickled = pickle.dumps(node_list, proto)
            unpickled = pickle.loads(pickled)
            self.assertIsNot(unpickled, node_list)
            self.assertEqual(unpickled, node_list)

    def test_nodelist_copy(self):
        if False:
            i = 10
            return i + 15
        node_list = NodeList()
        copied = copy.copy(node_list)
        self.assertIsNot(copied, node_list)
        self.assertEqual(copied, node_list)
        node_list.append([1])
        node_list.append([2])
        copied = copy.copy(node_list)
        self.assertIsNot(copied, node_list)
        self.assertEqual(copied, node_list)
        for (x, y) in zip(copied, node_list):
            self.assertIs(x, y)

    def test_nodelist_deepcopy(self):
        if False:
            while True:
                i = 10
        node_list = NodeList()
        copied = copy.deepcopy(node_list)
        self.assertIsNot(copied, node_list)
        self.assertEqual(copied, node_list)
        node_list.append([1])
        node_list.append([2])
        copied = copy.deepcopy(node_list)
        self.assertIsNot(copied, node_list)
        self.assertEqual(copied, node_list)
        for (x, y) in zip(copied, node_list):
            self.assertIsNot(x, y)
            self.assertEqual(x, y)
if __name__ == '__main__':
    unittest.main()