"""Monkey patch to make epydoc work with bzrlib's lazy imports."""
import epydoc.uid
import bzrlib.lazy_import
_ObjectUID = epydoc.uid.ObjectUID
_ScopeReplacer = bzrlib.lazy_import.ScopeReplacer

class ObjectUID(_ObjectUID):

    def __init__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(obj, _ScopeReplacer):
            obj = object.__getattribute__(obj, '_real_obj')
        _ObjectUID.__init__(self, obj)
epydoc.uid.ObjectUID = ObjectUID
_ScopeReplacer._should_proxy = True