import pytest
from PIL import __version__
pyroma = pytest.importorskip('pyroma', reason='Pyroma not installed')

def test_pyroma():
    if False:
        print('Hello World!')
    data = pyroma.projectdata.get_data('.')
    rating = pyroma.ratings.rate(data)
    if 'rc' in __version__:
        assert rating == (9, ["The package's version number does not comply with PEP-386."])
    else:
        assert rating == (10, [])