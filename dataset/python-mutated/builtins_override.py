import self
used_overriden_import = False

def replaced_import(*args):
    if False:
        i = 10
        return i + 15
    global used_overriden_import
    used_overriden_import = True
exec('import os; os', {'__builtins__': {'__import__': replaced_import}})
self.assertTrue(used_overriden_import)