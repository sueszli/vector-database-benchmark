from aim.storage import encoding as E
from aim.storage.encoding.encoding import decode
from aim.storage.object import CustomObject
from aim.storage.types import AimObject, AimObjectKey, AimObjectPath
from aim.storage.utils import ArrayFlag, CustomObjectFlagType
from aim.storage.container import Container
from aim.storage import treeutils
from aim.storage.treearrayview import TreeArrayView
from typing import Any, Iterator, Tuple, Union
from aim.storage.treeview import TreeView

class ContainerTreeView(TreeView):

    def __init__(self, container: Container) -> None:
        if False:
            print('Hello World!')
        self.container = container

    def preload(self):
        if False:
            print('Hello World!')
        self.container.preload()

    def finalize(self, index: 'ContainerTreeView'):
        if False:
            return 10
        self.container.finalize(index=index.container)

    def view(self, path: Union[AimObjectKey, AimObjectPath], resolve: bool=False):
        if False:
            while True:
                i = 10
        prefix = E.encode_path(path)
        container_view = self.container.view(prefix)
        tree_view = ContainerTreeView(container_view)
        if not resolve:
            return tree_view
        flag = decode(container_view.get(b'', default=b'\x00'))
        if isinstance(flag, CustomObjectFlagType):
            return CustomObject._aim_decode(flag.aim_name, tree_view)
        return tree_view

    def make_array(self, path: Union[AimObjectKey, AimObjectPath]=()):
        if False:
            return 10
        prefix = E.encode_path(path)
        self.container[prefix] = E.encode(ArrayFlag)

    def collect(self, path: Union[AimObjectKey, AimObjectPath]=(), strict: bool=True, resolve_objects: bool=False) -> AimObject:
        if False:
            i = 10
            return i + 15
        if path == Ellipsis:
            path = ()
        if isinstance(path, (int, str)):
            path = (path,)
        prefix = E.encode_path(path)
        it = self.container.view(prefix).items()
        try:
            return treeutils.decode_tree(it, strict=strict, resolve_objects=resolve_objects)
        except KeyError:
            raise KeyError('No key {} is present.'.format(path))

    def __delitem__(self, path: Union[AimObjectKey, AimObjectPath]):
        if False:
            print('Hello World!')
        if path == Ellipsis:
            path = ()
        if not isinstance(path, (tuple, list)):
            path = (path,)
        encoded_path = E.encode_path(path)
        self.container.delete_range(encoded_path, encoded_path + b'\xff')

    def set(self, path: Union[AimObjectKey, AimObjectPath], value: AimObject, strict: bool=True):
        if False:
            return 10
        if path == Ellipsis:
            path = ()
        if not isinstance(path, (tuple, list)):
            path = (path,)
        batch = self.container.batch()
        encoded_path = E.encode_path(path)
        self.container.delete_range(encoded_path, encoded_path + b'\xff', store_batch=batch)
        for (key, val) in treeutils.encode_tree(value, strict=strict):
            self.container.set(encoded_path + key, val, store_batch=batch)
        self.container.commit(batch)

    def keys_eager(self, path: Union[AimObjectKey, AimObjectPath]=()):
        if False:
            return 10
        return list(self.subtree(path).keys())

    def keys(self, path: Union[AimObjectKey, AimObjectPath]=(), level: int=0) -> Iterator[Union[AimObjectPath, AimObjectKey]]:
        if False:
            i = 10
            return i + 15
        encoded_path = E.encode_path(path)
        walker = self.container.walk(encoded_path)
        path = None
        while True:
            try:
                if path is None:
                    path = next(walker)
                else:
                    path = walker.send(path)
            except StopIteration:
                return
            path = E.decode_path(path)
            path = path[:max(level, 1)]
            if level <= 0:
                yield path[0]
            else:
                yield path
            p = E.encode_path(path)
            assert p.endswith(b'\xfe')
            path = p[:-1] + b'\xff'

    def items_eager(self, path: Union[AimObjectKey, AimObjectPath]=()):
        if False:
            i = 10
            return i + 15
        return list(self.subtree(path).items())

    def items(self, path: Union[AimObjectKey, AimObjectPath]=()) -> Iterator[Tuple[AimObjectKey, AimObject]]:
        if False:
            while True:
                i = 10
        prefix = E.encode_path(path)
        it = self.container.view(prefix).items()
        for (path, value) in treeutils.iter_decode_tree(it, level=1, skip_top_level=True):
            (key,) = path
            yield (key, value)

    def iterlevel(self, path: Union[AimObjectKey, AimObjectPath]=(), level: int=1) -> Iterator[Tuple[AimObjectPath, AimObject]]:
        if False:
            for i in range(10):
                print('nop')
        prefix = E.encode_path(path)
        it = self.container.items(prefix)
        for (path, value) in treeutils.iter_decode_tree(it, level=level, skip_top_level=True):
            yield (path, value)

    def array(self, path: Union[AimObjectKey, AimObjectPath]=(), dtype: Any=None) -> TreeArrayView:
        if False:
            while True:
                i = 10
        return TreeArrayView(self.subtree(path), dtype=dtype)

    def first_key(self, path: Union[AimObjectKey, AimObjectPath]=()) -> AimObjectKey:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(path, (int, str)):
            path = (path,)
        prefix = E.encode_path(path)
        p = E.decode_path(self.container.view(prefix).next_key())
        return p[0]

    def last_key(self, path: Union[AimObjectKey, AimObjectPath]=()) -> AimObjectKey:
        if False:
            print('Hello World!')
        if isinstance(path, (int, str)):
            path = (path,)
        prefix = E.encode_path(path)
        p = E.decode_path(self.container.view(prefix).prev_key())
        if not p:
            raise KeyError
        return p[0]