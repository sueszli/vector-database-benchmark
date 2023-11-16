import pytest
from pyroapi import pyro_backend
from pyroapi.tests import *
pytestmark = pytest.mark.stage('unit')

@pytest.fixture(params=['pyro', 'minipyro'])
def backend(request):
    if False:
        while True:
            i = 10
    with pyro_backend(request.param):
        yield