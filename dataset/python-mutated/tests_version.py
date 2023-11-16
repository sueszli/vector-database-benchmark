"""Test `tqdm.__version__`."""
import re
from ast import literal_eval

def test_version():
    if False:
        return 10
    'Test version string'
    from tqdm import __version__
    version_parts = re.split('[.-]', __version__)
    if __version__ != 'UNKNOWN':
        assert 3 <= len(version_parts), 'must have at least Major.minor.patch'
        assert all((isinstance(literal_eval(i), int) for i in version_parts[:3])), 'Version Major.minor.patch must be 3 integers'