"""A circular doubly linked list implementation.
"""
import threading
from typing import Generic, Optional, Type, TypeVar
P = TypeVar('P')
LN = TypeVar('LN', bound='ListNode')

class ListNode(Generic[P]):
    """A node in a circular doubly linked list, with an (optional) reference to
    a cache entry.

    The reference should only be `None` for the root node or if the node has
    been removed from the list.
    """
    _LOCK = threading.Lock()
    __slots__ = ['cache_entry', 'prev_node', 'next_node']

    def __init__(self, cache_entry: Optional[P]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.cache_entry = cache_entry
        self.prev_node: Optional[ListNode[P]] = None
        self.next_node: Optional[ListNode[P]] = None

    @classmethod
    def create_root_node(cls: Type['ListNode[P]']) -> 'ListNode[P]':
        if False:
            print('Hello World!')
        'Create a new linked list by creating a "root" node, which is a node\n        that has prev_node/next_node pointing to itself and no associated cache\n        entry.\n        '
        root = cls()
        root.prev_node = root
        root.next_node = root
        return root

    @classmethod
    def insert_after(cls: Type[LN], cache_entry: P, node: 'ListNode[P]') -> LN:
        if False:
            return 10
        'Create a new list node that is placed after the given node.\n\n        Args:\n            cache_entry: The associated cache entry.\n            node: The existing node in the list to insert the new entry after.\n        '
        new_node = cls(cache_entry)
        with cls._LOCK:
            new_node._refs_insert_after(node)
        return new_node

    def remove_from_list(self) -> None:
        if False:
            i = 10
            return i + 15
        'Remove this node from the list.'
        with self._LOCK:
            self._refs_remove_node_from_list()
        self.cache_entry = None

    def move_after(self, node: 'ListNode[P]') -> None:
        if False:
            i = 10
            return i + 15
        'Move this node from its current location in the list to after the\n        given node.\n        '
        with self._LOCK:
            assert self.prev_node
            assert self.next_node
            assert node.prev_node
            assert node.next_node
            assert self is not node
            self._refs_remove_node_from_list()
            self._refs_insert_after(node)

    def _refs_remove_node_from_list(self) -> None:
        if False:
            i = 10
            return i + 15
        'Internal method to *just* remove the node from the list, without\n        e.g. clearing out the cache entry.\n        '
        if self.prev_node is None or self.next_node is None:
            return
        prev_node = self.prev_node
        next_node = self.next_node
        prev_node.next_node = next_node
        next_node.prev_node = prev_node
        self.prev_node = None
        self.next_node = None

    def _refs_insert_after(self, node: 'ListNode[P]') -> None:
        if False:
            while True:
                i = 10
        'Internal method to insert the node after the given node.'
        assert self.prev_node is None
        assert self.next_node is None
        assert node.next_node
        assert node.prev_node
        prev_node = node
        next_node = node.next_node
        self.prev_node = prev_node
        self.next_node = next_node
        prev_node.next_node = self
        next_node.prev_node = self

    def get_cache_entry(self) -> Optional[P]:
        if False:
            return 10
        'Get the cache entry, returns None if this is the root node (i.e.\n        cache_entry is None) or if the entry has been dropped.\n        '
        return self.cache_entry