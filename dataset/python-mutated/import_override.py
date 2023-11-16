def custom_import(name, globals, locals, fromlist, level):
    if False:
        while True:
            i = 10
    print('import', name, fromlist, level)

    class M:
        var = 456
    return M
orig_import = __import__
try:
    __import__('builtins').__import__ = custom_import
except AttributeError:
    print('SKIP')
    raise SystemExit
orig_import('import1a')