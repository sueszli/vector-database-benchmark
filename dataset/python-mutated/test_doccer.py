""" Some tests for the documenting decorator and support functions """
import sys
import pytest
from numpy.testing import assert_equal, suppress_warnings
from scipy._lib import doccer
DOCSTRINGS_STRIPPED = sys.flags.optimize > 1
docstring = 'Docstring\n    %(strtest1)s\n        %(strtest2)s\n     %(strtest3)s\n'
param_doc1 = 'Another test\n   with some indent'
param_doc2 = 'Another test, one line'
param_doc3 = '    Another test\n       with some indent'
doc_dict = {'strtest1': param_doc1, 'strtest2': param_doc2, 'strtest3': param_doc3}
filled_docstring = 'Docstring\n    Another test\n       with some indent\n        Another test, one line\n     Another test\n       with some indent\n'

def test_unindent():
    if False:
        i = 10
        return i + 15
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        assert_equal(doccer.unindent_string(param_doc1), param_doc1)
        assert_equal(doccer.unindent_string(param_doc2), param_doc2)
        assert_equal(doccer.unindent_string(param_doc3), param_doc1)

def test_unindent_dict():
    if False:
        print('Hello World!')
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        d2 = doccer.unindent_dict(doc_dict)
    assert_equal(d2['strtest1'], doc_dict['strtest1'])
    assert_equal(d2['strtest2'], doc_dict['strtest2'])
    assert_equal(d2['strtest3'], doc_dict['strtest1'])

def test_docformat():
    if False:
        while True:
            i = 10
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        udd = doccer.unindent_dict(doc_dict)
        formatted = doccer.docformat(docstring, udd)
        assert_equal(formatted, filled_docstring)
        single_doc = 'Single line doc %(strtest1)s'
        formatted = doccer.docformat(single_doc, doc_dict)
        assert_equal(formatted, 'Single line doc Another test\n   with some indent')

@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason='docstrings stripped')
def test_decorator():
    if False:
        for i in range(10):
            print('nop')
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        decorator = doccer.filldoc(doc_dict, True)

        @decorator
        def func():
            if False:
                i = 10
                return i + 15
            ' Docstring\n            %(strtest3)s\n            '
        assert_equal(func.__doc__, ' Docstring\n            Another test\n               with some indent\n            ')
        decorator = doccer.filldoc(doc_dict, False)

        @decorator
        def func():
            if False:
                return 10
            ' Docstring\n            %(strtest3)s\n            '
        assert_equal(func.__doc__, ' Docstring\n                Another test\n                   with some indent\n            ')

@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason='docstrings stripped')
def test_inherit_docstring_from():
    if False:
        return 10
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)

        class Foo:

            def func(self):
                if False:
                    return 10
                'Do something useful.'
                return

            def func2(self):
                if False:
                    i = 10
                    return i + 15
                'Something else.'

        class Bar(Foo):

            @doccer.inherit_docstring_from(Foo)
            def func(self):
                if False:
                    print('Hello World!')
                '%(super)sABC'
                return

            @doccer.inherit_docstring_from(Foo)
            def func2(self):
                if False:
                    while True:
                        i = 10
                return
    assert_equal(Bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    assert_equal(Bar.func2.__doc__, Foo.func2.__doc__)
    bar = Bar()
    assert_equal(bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    assert_equal(bar.func2.__doc__, Foo.func2.__doc__)