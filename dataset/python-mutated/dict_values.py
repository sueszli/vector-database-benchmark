"""
Test the lazy imports objects are not exposed when checking the values of dictionaries
"""
import self
import importlib
from test.lazyimports.data.metasyntactic import names
from test.lazyimports.data.metasyntactic.names import *
g = globals().copy()
g_copy1 = g.copy()
g_copy2 = g.copy()
g_copy3 = g.copy()
g_copy4 = g.copy()

def notExposeLazyPrefix(obj_repr):
    if False:
        while True:
            i = 10
    return not obj_repr.startswith('<lazy_import ')
for value in g_copy1.values():
    self.assertNotRegex(repr(value), '^<lazy_import ')
for (key, value) in g_copy2.items():
    self.assertNotRegex(repr(value), '^<lazy_import ')
it = iter(g_copy3.values())
for value in it:
    self.assertNotRegex(repr(value), '^<lazy_import ')
it = iter(g_copy4.items())
for (key, value) in it:
    self.assertNotRegex(repr(value), '^<lazy_import ')
self.assertNotRegex(repr(g['names']), '^<lazy_import ')
self.assertNotRegex(repr(g['Foo']), '^<lazy_import ')
self.assertNotRegex(repr(g['Ack']), '^<lazy_import ')
self.assertNotRegex(repr(g['Bar']), '^<lazy_import ')
self.assertNotRegex(repr(g['Baz']), '^<lazy_import ')
self.assertNotRegex(repr(g['Thud']), '^<lazy_import ')
self.assertNotRegex(repr(g['Waldo']), '^<lazy_import ')
self.assertNotRegex(repr(g['Fred']), '^<lazy_import ')
self.assertNotRegex(repr(g['Plugh']), '^<lazy_import ')
self.assertNotRegex(repr(g['Metasyntactic']), '^<lazy_import ')