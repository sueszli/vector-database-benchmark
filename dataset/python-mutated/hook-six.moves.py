from PyInstaller import isolated

def pre_safe_import_module(api):
    if False:
        return 10
    '\n    Add the `six.moves` module as a dynamically defined runtime module node and all modules mapped by\n    `six._SixMetaPathImporter` as aliased module nodes to the passed graph.\n\n    The `six.moves` module is dynamically defined at runtime by the `six` module and hence cannot be imported in the\n    standard way. Instead, this hook adds a placeholder node for the `six.moves` module to the graph,\n    which implicitly adds an edge from that node to the node for its parent `six` module. This ensures that the `six`\n    module will be frozen into the executable. (Phew!)\n\n    `six._SixMetaPathImporter` is a PEP 302-compliant module importer converting imports independent of the current\n    Python version into imports specific to that version (e.g., under Python 3, from `from six.moves import\n    tkinter_tix` to `import tkinter.tix`). For each such mapping, this hook adds a corresponding module alias to the\n    graph allowing PyInstaller to translate the former to the latter.\n    '

    @isolated.call
    def real_to_six_module_name():
        if False:
            while True:
                i = 10
        '\n        Generate a dictionary from conventional module names to "six.moves" attribute names (e.g., from `tkinter.tix` to\n        `six.moves.tkinter_tix`).\n        '
        import six
        return {moved.mod: 'six.moves.' + moved.name for moved in six._moved_attributes if isinstance(moved, (six.MovedModule, six.MovedAttribute))}
    if isinstance(real_to_six_module_name, str):
        raise SystemExit('pre-safe-import-module hook failed, needs fixing.')
    api.add_runtime_package(api.module_name)
    for (real_module_name, six_module_name) in real_to_six_module_name.items():
        api.add_alias_module(real_module_name, six_module_name)