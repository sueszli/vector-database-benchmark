def _pyi_rthook():
    if False:
        for i in range(10):
            print('nop')
    import pathlib
    import pkgutil
    import sys
    from pyimod02_importers import PyiFrozenImporter
    from _pyi_rth_utils import is_macos_app_bundle
    _orig_pkgutil_iter_modules = pkgutil.iter_modules

    def _pyi_pkgutil_iter_modules(path=None, prefix=''):
        if False:
            while True:
                i = 10
        yield from _orig_pkgutil_iter_modules(path, prefix)
        for importer in pkgutil.iter_importers():
            if isinstance(importer, PyiFrozenImporter):
                break
        else:
            return
        if path is None:
            for (entry_name, entry_data) in importer.toc_tree.items():
                is_pkg = isinstance(entry_data, dict)
                yield pkgutil.ModuleInfo(importer, prefix + entry_name, is_pkg)
        else:
            MEIPASS = pathlib.Path(sys._MEIPASS).resolve()
            if is_macos_app_bundle:
                ALT_MEIPASS = (pathlib.Path(sys._MEIPASS).parent / 'Resources').resolve()
            seen_pkg_prefices = set()
            for pkg_path in path:
                pkg_path = pathlib.Path(pkg_path).resolve()
                pkg_prefix = None
                try:
                    pkg_prefix = pkg_path.relative_to(MEIPASS)
                except ValueError:
                    pass
                if pkg_prefix is None and is_macos_app_bundle:
                    try:
                        pkg_prefix = pkg_path.relative_to(ALT_MEIPASS)
                    except ValueError:
                        pass
                if pkg_prefix is None:
                    continue
                if pkg_prefix in seen_pkg_prefices:
                    continue
                seen_pkg_prefices.add(pkg_prefix)
                tree_node = importer.toc_tree
                for pkg_name_part in pkg_prefix.parts:
                    tree_node = tree_node.get(pkg_name_part)
                    if tree_node is None:
                        tree_node = {}
                        break
                for (entry_name, entry_data) in tree_node.items():
                    is_pkg = isinstance(entry_data, dict)
                    yield pkgutil.ModuleInfo(importer, prefix + entry_name, is_pkg)
    pkgutil.iter_modules = _pyi_pkgutil_iter_modules
_pyi_rthook()
del _pyi_rthook