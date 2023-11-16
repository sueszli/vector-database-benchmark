def _pyi_rthook():
    if False:
        while True:
            i = 10
    import os
    import pathlib
    import sys
    import pkg_resources
    from pyimod02_importers import PyiFrozenImporter
    SYS_PREFIX = pathlib.PurePath(sys._MEIPASS)

    class _TocFilesystem:
        """
        A prefix tree implementation for embedded filesystem reconstruction.

        NOTE: as of PyInstaller 6.0, the embedded PYZ archive cannot contain data files anymore. Instead, it contains
        only .pyc modules - which are by design not returned by `PyiFrozenProvider`. So this implementation has been
        reduced to supporting only directories implied by collected packages.
        """

        def __init__(self, tree_node):
            if False:
                while True:
                    i = 10
            self._tree = tree_node

        def _get_tree_node(self, path):
            if False:
                for i in range(10):
                    print('nop')
            path = pathlib.PurePath(path)
            current = self._tree
            for component in path.parts:
                if component not in current:
                    return None
                current = current[component]
            return current

        def path_exists(self, path):
            if False:
                i = 10
                return i + 15
            node = self._get_tree_node(path)
            return isinstance(node, dict)

        def path_isdir(self, path):
            if False:
                return 10
            node = self._get_tree_node(path)
            return isinstance(node, dict)

        def path_listdir(self, path):
            if False:
                for i in range(10):
                    print('nop')
            node = self._get_tree_node(path)
            if not isinstance(node, dict):
                return []
            return [entry_name for (entry_name, entry_data) in node.items() if isinstance(entry_data, dict)]

    class PyiFrozenProvider(pkg_resources.NullProvider):
        """
        Custom pkg_resources provider for PyiFrozenImporter.
        """

        def __init__(self, module):
            if False:
                i = 10
                return i + 15
            super().__init__(module)
            self._pkg_path = pathlib.PurePath(module.__file__).parent
            self.embedded_tree = _TocFilesystem(self.loader.toc_tree)

        def _normalize_path(self, path):
            if False:
                return 10
            return pathlib.Path(os.path.normpath(path))

        def _is_relative_to_package(self, path):
            if False:
                while True:
                    i = 10
            return path == self._pkg_path or self._pkg_path in path.parents

        def _has(self, path):
            if False:
                while True:
                    i = 10
            path = self._normalize_path(path)
            if not self._is_relative_to_package(path):
                return False
            if path.exists():
                return True
            rel_path = path.relative_to(SYS_PREFIX)
            return self.embedded_tree.path_exists(rel_path)

        def _isdir(self, path):
            if False:
                for i in range(10):
                    print('nop')
            path = self._normalize_path(path)
            if not self._is_relative_to_package(path):
                return False
            rel_path = path.relative_to(SYS_PREFIX)
            node = self.embedded_tree._get_tree_node(rel_path)
            if node is None:
                return path.is_dir()
            else:
                return not isinstance(node, str)

        def _listdir(self, path):
            if False:
                return 10
            path = self._normalize_path(path)
            if not self._is_relative_to_package(path):
                return []
            rel_path = path.relative_to(SYS_PREFIX)
            content = self.embedded_tree.path_listdir(rel_path)
            if path.is_dir():
                path = str(path)
                content = list(set(content + os.listdir(path)))
            return content
    pkg_resources.register_loader_type(PyiFrozenImporter, PyiFrozenProvider)
_pyi_rthook()
del _pyi_rthook