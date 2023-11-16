from aim.storage.container import Container, ContainerItemsIterator, ContainerKey, ContainerValue
from aim.storage.containertreeview import ContainerTreeView
from typing import Iterator, Tuple

class PrefixView(Container):
    """
    A mutable view to a :obj:`Container` given by a key prefix.

    Args:
        prefix (:obj:`bytes`): the prefix that defines the key range of the
            view-container. The resulting container will share an access to
            only records in the `prefix` key range, but with `prefix`-es
            stripped from them.

            For example, if the Container contents are:
            `{
                b'e.y': b'012',
                b'meta.x': b'123',
                b'meta.z': b'x',
                b'zzz': b'oOo'
            }`, then `container.view(prefix=b'meta.')` will behave (almost)
            exactly as an Container:
            `{
                b'x': b'123',
                b'z': b'x',
            }`
        container (:obj:`Container`): the parent container to build the view on
    """

    def __init__(self, *, prefix: bytes=b'', container: Container, read_only: bool=None) -> None:
        if False:
            return 10
        self.prefix = prefix
        self.parent = container
        self.read_only = read_only

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close all the resources.'
        pass

    def preload(self):
        if False:
            print('Hello World!')
        'Preload the Container.\n\n        The interface of Container is designed in such a way that (almost) all\n        the operations are supported to be done lazily.\n        Sometimes there is need to preload the storage without performing an\n        operation that will cause an actual read / write access.\n        '
        self.parent.preload()

    def finalize(self, index: Container):
        if False:
            return 10
        'Finalize the Container.\n\n        Perform operations of compactions, indexing, optimization, etc.\n        '
        prefix = self.absolute_path()
        index.delete_range(prefix, prefix + b'\xff')
        self.parent.finalize(index=index)

    def absolute_path(self, path: bytes=None) -> bytes:
        if False:
            while True:
                i = 10
        "Returns the absolute path for the given relative `path`.\n\n        Path separators / sentinels should be handled in higher level so that\n        `join(a, b) == a + b` property holds. This can be easily achieved by\n        having all the paths end with the sentinel:\n        `join('a/b/c/', 'e/f/') == 'a/b/c/' + 'e/f/' = 'a/b/c/e/f/'`\n        "
        if path is None:
            return self.prefix
        return self.prefix + path

    def get(self, key: ContainerKey, default=None) -> ContainerValue:
        if False:
            while True:
                i = 10
        'Returns the value by the given `key` if it exists else `default`.\n\n        The `default` is :obj:`None` by default.\n        '
        path = self.absolute_path(key)
        return self.parent.get(path, default)

    def __getitem__(self, key: ContainerKey) -> ContainerValue:
        if False:
            print('Hello World!')
        'Returns the value by the given `key`.'
        path = self.absolute_path(key)
        return self.parent[path]

    def set(self, key: ContainerKey, value: ContainerValue, store_batch=None) -> None:
        if False:
            while True:
                i = 10
        'Set a value for given key, optionally store in a batch.\n\n        If `store_batch` is provided, instead of the `(key, value)` being added\n        to the collection immediately, the operation is stored in a batch in\n        order to be executed in a whole with other write operations. Depending\n        on the :obj:`Conainer` implementation, this may feature transactions,\n        atomic writes, etc.\n        '
        path = self.absolute_path(key)
        self.parent.set(path, value, store_batch=store_batch)

    def __setitem__(self, key: ContainerKey, value: ContainerValue) -> None:
        if False:
            while True:
                i = 10
        'Set a value for given key.'
        path = self.absolute_path(key)
        self.parent[path] = value

    def delete(self, key: ContainerKey, store_batch=None):
        if False:
            i = 10
            return i + 15
        'Delete a key-value record by the given key,\n        optionally store in a batch.\n\n        If `store_batch` is provided, instead of the `(key, value)` being added\n        to the collection immediately, the operation is stored in a batch in\n        order to be executed in a whole with other write operations. Depending\n        on the :obj:`Conainer` implementation, this may feature transactions,\n        atomic writes, etc.\n        '
        path = self.absolute_path(key)
        return self.parent.delete(path, store_batch=store_batch)

    def __delitem__(self, key: ContainerKey) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete a key-value record by the given key.'
        path = self.absolute_path(key)
        del self.parent[path]

    def delete_range(self, begin: ContainerKey, end: ContainerKey, store_batch=None):
        if False:
            i = 10
            return i + 15
        'Delete all the records in the given `[begin, end)` key range.'
        begin_path = self.absolute_path(begin)
        end_path = self.absolute_path(end)
        self.parent.delete_range(begin_path, end_path, store_batch=store_batch)

    def next_item(self, key: ContainerKey=b'') -> Tuple[ContainerKey, ContainerValue]:
        if False:
            return 10
        'Returns `(key, value)` for the key that comes (lexicographically)\n        right after the provided `key`.\n        '
        path = self.absolute_path(key)
        (keys, value) = self.parent.next_item(path)
        if path:
            (_prefix, _path, keys) = keys.partition(path)
            if _prefix or _path != path:
                raise KeyError
        return (keys, value)

    def prev_item(self, key: ContainerKey=b'') -> Tuple[ContainerKey, ContainerValue]:
        if False:
            print('Hello World!')
        'Returns `(key, value)` for the key that comes (lexicographically)\n        right before the provided `key`.\n        '
        path = self.absolute_path(key)
        (keys, value) = self.parent.prev_item(path)
        if path:
            (_prefix, _path, keys) = keys.partition(path)
            if _prefix or _path != path:
                raise KeyError
        return (keys, value)

    def walk(self, key: ContainerKey=b''):
        if False:
            print('Hello World!')
        "A bi-directional generator to walk over the collection of records on\n        any arbitrary order. The `prefix` sent to the generator (lets call it\n        a `walker`) seeks for lower-bound key in the collection.\n\n        In other words, if the Container contents are:\n        `{\n            b'e.y': b'012',\n            b'meta.x': b'123',\n            b'meta.z': b'x',\n            b'zzz': b'oOo'\n        }` and `walker = container.walk()` then:\n        `walker.send(b'meta') == b'meta.x'`, `walker.send(b'e.y') == b'e.y'`\n        "
        path = self.absolute_path(key)
        walker = self.parent.walk(path)
        p = None
        while True:
            if p is None:
                next_key = next(walker)
            else:
                next_key = walker.send(p)
            if next_key is None:
                return
            if path:
                (_prefix, _path, next_key) = next_key.partition(path)
                if _prefix or _path != path:
                    return
            key = (yield next_key)
            p = self.absolute_path(key)

    def items(self, key: ContainerKey=b'') -> Iterator[Tuple[ContainerKey, ContainerValue]]:
        if False:
            i = 10
            return i + 15
        "Iterate over all the key-value records in the prefix key range.\n\n        The iteration is always performed in lexiographic order w.r.t keys.\n        If `prefix` is provided, iterate only over those records that have key\n        starting with the `prefix`.\n\n        For example, if `prefix == b'meta.'`, and the Container consists of:\n        `{\n            b'e.y': b'012',\n            b'meta.x': b'123',\n            b'meta.z': b'x',\n            b'zzz': b'oOo'\n        }`, the method will yield `(b'meta.x', b'123')` and `(b'meta.z', b'x')`\n\n        Args:\n            prefix (:obj:`bytes`): the prefix that defines the key range\n        "
        return PrefixViewItemsIterator(self, key)

    def view(self, prefix: bytes=b'') -> Container:
        if False:
            return 10
        "Return a view (even mutable ones) that enable access to the container\n        but with modifications.\n\n        Args:\n            prefix (:obj:`bytes`): the prefix that defines the key range of the\n                view-container. The resulting container will share an access to\n                only records in the `prefix` key range, but with `prefix`-es\n                stripped from them.\n\n                For example, if the Container contents are:\n                `{\n                    b'e.y': b'012',\n                    b'meta.x': b'123',\n                    b'meta.z': b'x',\n                    b'zzz': b'oOo'\n                }`, then `container.view(prefix=b'meta.')` will behave (almost)\n                exactly as an Container:\n                `{\n                    b'x': b'123',\n                    b'z': b'x',\n                }`\n        "
        return self.parent.view(self.prefix + prefix)

    def tree(self) -> ContainerTreeView:
        if False:
            return 10
        "Return a :obj:`ContainerTreeView` which enables hierarchical view and access\n        to the container records.\n\n        This is achieved by prefixing groups and using `PATH_SENTINEL` as a\n        separator for keys.\n\n        For example, if the Container contents are:\n            `{\n                b'e.y': b'012',\n                b'meta.x': b'123',\n                b'meta.z': b'x',\n                b'zzz': b'oOo'\n            }`, and the path sentinel is `b'.'` then `tree = container.tree()`\n            will behave as a (possibly deep) dict-like object:\n            `tree[b'meta'][b'x'] == b'123'`\n        "
        return ContainerTreeView(self)

    def batch(self):
        if False:
            print('Hello World!')
        'Creates a new batch object to store operations in before executing\n        using :obj:`Container.commit`.\n\n        The operations :obj:`Container.set`, :obj:`Container.delete`,\n        :obj:`Container.delete_range` are supported.\n\n        See more at :obj:`Container.commit`\n        '
        return self.parent.batch()

    def commit(self, batch):
        if False:
            print('Hello World!')
        'Execute the accumulated write operations in the given `batch`.\n\n        Depending on the :obj:`Container` implementation, this may feature\n        transactions, atomic writes, etc.\n        '
        return self.parent.commit(batch)

class PrefixViewItemsIterator(ContainerItemsIterator):

    def __init__(self, prefix_view, key):
        if False:
            for i in range(10):
                print('nop')
        self.prefix_view = prefix_view
        self.path = prefix_view.absolute_path(key)
        self.it = prefix_view.parent.items(self.path)
        self.prefix_len = len(self.path)

    def next(self):
        if False:
            i = 10
            return i + 15
        item = self.it.next()
        if item is None:
            return None
        keys = item[0]
        value = item[1]
        return (keys[self.prefix_len:], value)