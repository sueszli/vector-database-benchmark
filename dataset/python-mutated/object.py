from __future__ import print_function
"\n>>> from object_ext import *\n\n>>> type(ref_to_noncopyable())\n<class 'object_ext.NotCopyable'>\n\n>>> def print1(x):\n...     print(x)\n>>> call_object_3(print1)\n3\n>>> message()\n'hello, world!'\n>>> number()\n42\n\n>>> test('hi')\n1\n>>> test(None)\n0\n>>> test_not('hi')\n0\n>>> test_not(0)\n1\n\n        Attributes\n\n>>> class X: pass\n...\n>>> x = X()\n\n>>> try: obj_getattr(x, 'foo')\n... except AttributeError: pass\n... else: print('expected an exception')\n>>> try: obj_objgetattr(x, 'objfoo')\n... except AttributeError: pass\n... else: print('expected an exception')\n\n>>> obj_setattr(x, 'foo', 1)\n>>> x.foo\n1\n>>> obj_objsetattr(x, 'objfoo', 1)\n>>> try:obj_objsetattr(x, 1)\n... except TypeError: pass\n... else: print('expected an exception')\n>>> x.objfoo\n1\n>>> obj_getattr(x, 'foo')\n1\n>>> obj_objgetattr(x, 'objfoo')\n1\n>>> try:obj_objgetattr(x, 1)\n... except TypeError: pass\n... else: print('expected an exception')\n>>> obj_const_getattr(x, 'foo')\n1\n>>> obj_const_objgetattr(x, 'objfoo')\n1\n>>> obj_setattr42(x, 'foo')\n>>> x.foo\n42\n>>> obj_objsetattr42(x, 'objfoo')\n>>> x.objfoo\n42\n>>> obj_moveattr(x, 'foo', 'bar')\n>>> x.bar\n42\n>>> obj_objmoveattr(x, 'objfoo', 'objbar')\n>>> x.objbar\n42\n>>> test_attr(x, 'foo')\n1\n>>> test_objattr(x, 'objfoo')\n1\n>>> test_not_attr(x, 'foo')\n0\n>>> test_not_objattr(x, 'objfoo')\n0\n>>> x.foo = None\n>>> test_attr(x, 'foo')\n0\n>>> x.objfoo = None\n>>> test_objattr(x, 'objfoo')\n0\n>>> test_not_attr(x, 'foo')\n1\n>>> test_not_objattr(x, 'objfoo')\n1\n>>> obj_delattr(x, 'foo')\n>>> obj_objdelattr(x, 'objfoo')\n>>> try:obj_delattr(x, 'foo')\n... except AttributeError: pass\n... else: print('expected an exception')\n>>> try:obj_objdelattr(x, 'objfoo')\n... except AttributeError: pass\n... else: print('expected an exception')\n\n        Items\n\n>>> d = {}\n>>> obj_setitem(d, 'foo', 1)\n>>> d['foo']\n1\n>>> obj_getitem(d, 'foo')\n1\n>>> obj_const_getitem(d, 'foo')\n1\n>>> obj_setitem42(d, 'foo')\n>>> obj_getitem(d, 'foo')\n42\n>>> d['foo']\n42\n>>> obj_moveitem(d, 'foo', 'bar')\n>>> d['bar']\n42\n>>> obj_moveitem2(d, 'bar', d, 'baz')\n>>> d['baz']\n42\n>>> test_item(d, 'foo')\n1\n>>> test_not_item(d, 'foo')\n0\n>>> d['foo'] = None\n>>> test_item(d, 'foo')\n0\n>>> test_not_item(d, 'foo')\n1\n\n        Slices\n\n>>> assert check_string_slice()\n\n        Operators\n\n>>> def print_args(*args, **kwds):\n...     print(args, kwds)\n>>> test_call(print_args, (0, 1, 2, 3), {'a':'A'})\n(0, 1, 2, 3) {'a': 'A'}\n\n\n>>> assert check_binary_operators()\n\n>>> class X: pass\n...\n>>> assert check_inplace(list(range(3)), X())\n\n\n       Now make sure that object is actually managing reference counts\n\n>>> import weakref\n>>> class Z: pass\n...\n>>> z = Z()\n>>> def death(r): print('death')\n...\n>>> r = weakref.ref(z, death)\n>>> z.foo = 1\n>>> obj_getattr(z, 'foo')\n1\n>>> del z\ndeath\n"

def run(args=None):
    if False:
        i = 10
        return i + 15
    import sys
    import doctest
    if args is not None:
        sys.argv = args
    return doctest.testmod(sys.modules.get(__name__))
if __name__ == '__main__':
    print('running...')
    import sys
    status = run()[0]
    if status == 0:
        print('Done.')
    sys.exit(status)