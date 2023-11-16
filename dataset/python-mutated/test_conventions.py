from hypothesis.utils.conventions import UniqueIdentifier

def test_unique_identifier_repr():
    if False:
        i = 10
        return i + 15
    assert repr(UniqueIdentifier('hello_world')) == 'hello_world'