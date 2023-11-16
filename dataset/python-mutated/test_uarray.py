from cupy import testing
import cupyx.scipy.linalg._uarray

@testing.with_requires('scipy>=1.5')
def test_implements_names():
    if False:
        print('Hello World!')
    assert not cupyx.scipy.linalg._uarray._notfound