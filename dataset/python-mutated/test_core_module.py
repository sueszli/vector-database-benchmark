import chainerx

def test_core():
    if False:
        i = 10
        return i + 15
    assert chainerx.__name__ == 'chainerx'

def test_is_available():
    if False:
        print('Hello World!')
    assert chainerx.is_available()