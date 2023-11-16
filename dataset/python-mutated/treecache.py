SENTINEL = object()

class TreeCacheNode(dict):
    """The type of nodes in our tree.

    Has its own type so we can distinguish it from real dicts that are stored at the
    leaves.
    """

class TreeCache:
    """
    Tree-based backing store for LruCache. Allows subtrees of data to be deleted
    efficiently.
    Keys must be tuples.

    The data structure is a chain of TreeCacheNodes:
        root = {key_1: {key_2: _value}}
    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.size: int = 0
        self.root = TreeCacheNode()

    def __setitem__(self, key, value) -> None:
        if False:
            return 10
        self.set(key, value)

    def __contains__(self, key) -> bool:
        if False:
            return 10
        return self.get(key, SENTINEL) is not SENTINEL

    def set(self, key, value) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, TreeCacheNode):
            raise ValueError('Cannot store TreeCacheNodes in a TreeCache')
        node = self.root
        for k in key[:-1]:
            next_node = node.get(k, SENTINEL)
            if next_node is SENTINEL:
                next_node = node[k] = TreeCacheNode()
            elif not isinstance(next_node, TreeCacheNode):
                raise ValueError('value conflicts with an existing subtree')
            node = next_node
        node[key[-1]] = value
        self.size += 1

    def get(self, key, default=None):
        if False:
            i = 10
            return i + 15
        'When `key` is a full key, fetches the value for the given key (if\n        any).\n\n        If `key` is only a partial key (i.e. a truncated tuple) then returns a\n        `TreeCacheNode`, which can be passed to the `iterate_tree_cache_*`\n        functions to iterate over all entries in the cache with keys that start\n        with the given partial key.\n        '
        node = self.root
        for k in key[:-1]:
            node = node.get(k, None)
            if node is None:
                return default
        return node.get(key[-1], default)

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        self.size = 0
        self.root = TreeCacheNode()

    def pop(self, key, default=None):
        if False:
            print('Hello World!')
        "Remove the given key, or subkey, from the cache\n\n        Args:\n            key: key or subkey to remove.\n            default: value to return if key is not found\n\n        Returns:\n            If the key is not found, 'default'. If the key is complete, the removed\n            value. If the key is partial, the TreeCacheNode corresponding to the part\n            of the tree that was removed.\n        "
        if not isinstance(key, tuple):
            raise TypeError('The cache key must be a tuple not %r' % (type(key),))
        nodes = []
        node = self.root
        for k in key[:-1]:
            node = node.get(k, None)
            if node is None:
                return default
            if not isinstance(node, TreeCacheNode):
                raise ValueError('pop() key too long')
            nodes.append(node)
        popped = node.pop(key[-1], SENTINEL)
        if popped is SENTINEL:
            return default
        node_and_keys = list(zip(nodes, key))
        node_and_keys.reverse()
        node_and_keys.append((self.root, None))
        for i in range(len(node_and_keys) - 1):
            (n, k) = node_and_keys[i]
            if n:
                break
            node_and_keys[i + 1][0].pop(k)
        cnt = sum((1 for _ in iterate_tree_cache_entry(popped)))
        self.size -= cnt
        return popped

    def values(self):
        if False:
            i = 10
            return i + 15
        return iterate_tree_cache_entry(self.root)

    def items(self):
        if False:
            i = 10
            return i + 15
        return iterate_tree_cache_items((), self.root)

    def __len__(self) -> int:
        if False:
            return 10
        return self.size

def iterate_tree_cache_entry(d):
    if False:
        return 10
    'Helper function to iterate over the leaves of a tree, i.e. a dict of that\n    can contain dicts.\n    '
    if isinstance(d, TreeCacheNode):
        for value_d in d.values():
            yield from iterate_tree_cache_entry(value_d)
    else:
        yield d

def iterate_tree_cache_items(key, value):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to iterate over the leaves of a tree, i.e. a dict of that\n    can contain dicts.\n\n    The provided key is a tuple that will get prepended to the returned keys.\n\n    Example:\n\n        cache = TreeCache()\n        cache[(1, 1)] = "a"\n        cache[(1, 2)] = "b"\n        cache[(2, 1)] = "c"\n\n        tree_node = cache.get((1,))\n\n        items = iterate_tree_cache_items((1,), tree_node)\n        assert list(items) == [((1, 1), "a"), ((1, 2), "b")]\n\n    Returns:\n        A generator yielding key/value pairs.\n    '
    if isinstance(value, TreeCacheNode):
        for (sub_key, sub_value) in value.items():
            yield from iterate_tree_cache_items((*key, sub_key), sub_value)
    else:
        yield (key, value)