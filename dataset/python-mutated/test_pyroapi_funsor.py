import pytest
try:
    import funsor
    import pyro.contrib.funsor
    funsor.set_backend('torch')
except ImportError:
    pytestmark = pytest.mark.skip()
from pyroapi import pyro_backend
from pyroapi.tests import *

@pytest.fixture(params=['contrib.funsor'])
def backend(request):
    if False:
        while True:
            i = 10
    with pyro_backend(request.param):
        yield