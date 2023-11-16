__doc__ = "\ndbm compatibility module for Python versions that don't have dbm.\n\nThis does not not NOT (repeat, *NOT*) provide complete dbm functionality.\nIt's just a stub on which to hang just enough pieces of dbm functionality\nthat the whichdb.whichdb() implementstation in the various 2.X versions of\nPython won't blow up even if dbm wasn't compiled in.\n"
__revision__ = 'src/engine/SCons/compat/_scons_dbm.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'

class error(Exception):
    pass

def open(*args, **kw):
    if False:
        for i in range(10):
            print('nop')
    raise error()