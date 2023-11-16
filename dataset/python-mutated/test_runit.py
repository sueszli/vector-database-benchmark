import pathlib
import pytest
import salt.modules.runit as runit
from tests.support.mock import patch
pytestmark = [pytest.mark.skip_on_windows]

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {runit: {}}

@pytest.fixture
def service_dir(tmp_path):
    if False:
        return 10
    dirname = tmp_path / 'services'
    dirname.mkdir(exist_ok=True, parents=True)
    return str(dirname)

def test__get_svc_path_on_non_symlinked_service(service_dir):
    if False:
        return 10
    service = pathlib.Path(service_dir, 'service')
    service.mkdir(exist_ok=True, parents=True)
    service_runfile = service / 'run'
    service_runfile.touch()
    with patch.object(runit, 'SERVICE_DIR', service_dir):
        with patch('os.access', return_value=True):
            path_list = runit._get_svc_path(str(service), 'ENABLED')
            assert path_list
            assert len(path_list) == 1
            assert path_list[0] == str(service)

def test__get_svc_path_on_symlinked_service(service_dir, tmp_path):
    if False:
        i = 10
        return i + 15
    sym_dir = tmp_path / 'sym_dir'
    sym_dir.mkdir(exist_ok=True, parents=True)
    service_runfile = sym_dir / 'run'
    service_runfile.touch()
    service = pathlib.Path(service_dir, 'service')
    service.symlink_to(sym_dir)
    with patch.object(runit, 'SERVICE_DIR', service_dir):
        with patch('os.access', return_value=True):
            path_list = runit._get_svc_path(str(service), 'ENABLED')
            assert path_list
            assert len(path_list) == 1
            assert path_list[0] == str(sym_dir)