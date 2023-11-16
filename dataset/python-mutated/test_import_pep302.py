def test_pep302_loader_builtin(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source("\n        mod = 'sys'\n        import pkgutil\n        ldr = pkgutil.get_loader(mod)\n        assert ldr\n        assert ldr.is_package(mod) == False\n        assert ldr.get_code(mod) is None\n        assert ldr.get_source(mod) is None\n        ")

def test_pep302_loader_frozen_module(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source("\n        mod = 'compileall'\n        import pkgutil\n        ldr = pkgutil.get_loader(mod)\n        assert ldr\n        assert ldr.is_package(mod) == False\n        assert ldr.get_code(mod) is not None\n        assert ldr.get_source(mod) is None\n        # Import at the very end, just to get the module frozen.\n        import compileall\n        ")

def test_pep302_loader_frozen_package(pyi_builder):
    if False:
        return 10
    pyi_builder.test_source("\n        mod = 'json'\n        import pkgutil\n        ldr = pkgutil.get_loader(mod)\n        assert ldr\n        assert ldr.is_package(mod) == True\n        assert ldr.get_code(mod) is not None\n        assert ldr.get_source(mod) is None\n        # Import at the very end, just to get the module frozen.\n        import json\n        ")

def test_pep302_loader_frozen_submodule(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source("\n        mod = 'json.encoder'\n        import pkgutil\n        ldr = pkgutil.get_loader(mod)\n        assert ldr\n        assert ldr.is_package(mod) == False\n        assert ldr.get_code(mod) is not None\n        assert ldr.get_source(mod) is None\n        # Import at the very end, just to get the module frozen.\n        import json.encoder\n        ")

def test_pep302_loader_cextension(pyi_builder):
    if False:
        return 10
    pyi_builder.test_source("\n        mod = '_sqlite3'\n        import pkgutil\n        ldr = pkgutil.get_loader(mod)\n        assert ldr\n        assert ldr.is_package(mod) == False\n        assert ldr.get_code(mod) is None\n        assert ldr.get_source(mod) is None\n        # Import at the very end, just to get the module frozen.\n        import sqlite3\n        ")