from ..trim_docstring import trim_docstring

def test_trim_docstring():
    if False:
        while True:
            i = 10

    class WellDocumentedObject:
        """
        This object is very well-documented. It has multiple lines in its
        description.

        Multiple paragraphs too
        """
    assert trim_docstring(WellDocumentedObject.__doc__) == 'This object is very well-documented. It has multiple lines in its\ndescription.\n\nMultiple paragraphs too'

    class UndocumentedObject:
        pass
    assert trim_docstring(UndocumentedObject.__doc__) is None