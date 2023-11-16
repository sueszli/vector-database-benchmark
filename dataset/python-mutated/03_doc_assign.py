"""This program is self-checking!"""

def test_doc_descriptor():
    if False:
        while True:
            i = 10

    class DocDescr(object):

        def __get__(self, object, otype):
            if False:
                while True:
                    i = 10
            if object:
                object = object.__class__.__name__ + ' instance'
            if otype:
                otype = otype.__name__
            return 'object=%s; type=%s' % (object, otype)

    class OldClass:
        __doc__ = DocDescr()

    class NewClass(object):
        __doc__ = DocDescr()
    assert OldClass.__doc__ == 'object=None; type=OldClass'
    assert OldClass().__doc__ == 'object=OldClass instance; type=OldClass'
    assert NewClass.__doc__ == 'object=None; type=NewClass'
    assert NewClass().__doc__ == 'object=NewClass instance; type=NewClass'
test_doc_descriptor()