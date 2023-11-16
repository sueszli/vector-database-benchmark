import logging
import pytest
import salt.renderers.pyobjects as pyobjects
from salt.utils.odict import OrderedDict
from tests.support.mock import MagicMock
log = logging.getLogger(__name__)

@pytest.fixture()
def configure_loader_modules(minion_opts):
    if False:
        return 10
    minion_opts['file_client'] = 'local'
    minion_opts['id'] = 'testminion'
    pillar = MagicMock(return_value={})
    return {pyobjects: {'__opts__': minion_opts, '__pillar__': pillar, '__salt__': {'config.get': MagicMock(), 'grains.get': MagicMock(), 'mine.get': MagicMock(), 'pillar.get': MagicMock()}}}

@pytest.fixture
def pyobjects_template():
    if False:
        while True:
            i = 10

    class Template:

        def readlines():
            if False:
                i = 10
                return i + 15
            return ['#!pyobjects', "state_id = __sls__ + '_' + __opts__['id']", "File.directory(state_id, name='/tmp', mode='1777', owner='root', group='root')"]
    return Template

@pytest.mark.slow_test
def test_opts_and_sls_access(pyobjects_template):
    if False:
        i = 10
        return i + 15
    ret = pyobjects.render(pyobjects_template, sls='pyobj.runtest')
    assert ret == OrderedDict([('pyobj.runtest_testminion', {'file.directory': [{'group': 'root'}, {'mode': '1777'}, {'name': '/tmp'}, {'owner': 'root'}]})])