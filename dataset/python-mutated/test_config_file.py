import os
import pytest
from jrnl.exception import JrnlException
from jrnl.install import find_alt_config

def test_find_alt_config(request):
    if False:
        return 10
    work_config_path = os.path.join(request.fspath.dirname, '..', 'data', 'configs', 'basic_onefile.yaml')
    found_alt_config = find_alt_config(work_config_path)
    assert found_alt_config == work_config_path

def test_find_alt_config_not_exist(request):
    if False:
        while True:
            i = 10
    bad_config_path = os.path.join(request.fspath.dirname, '..', 'data', 'configs', 'does-not-exist.yaml')
    with pytest.raises(JrnlException) as ex:
        found_alt_config = find_alt_config(bad_config_path)
        assert found_alt_config is not None
    assert isinstance(ex.value, JrnlException)