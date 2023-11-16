import json
from pathlib import Path
import pytest
from redbot.pytest.data_manager import *
from redbot.core import data_manager

def test_no_basic(cog_instance):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):
        data_manager.core_data_path()
    with pytest.raises(RuntimeError):
        data_manager.cog_data_path(cog_instance)

@pytest.mark.skip
def test_core_path(data_mgr_config, tmpdir):
    if False:
        print('Hello World!')
    conf_path = tmpdir.join('config.json')
    conf_path.write(json.dumps(data_mgr_config))
    data_manager.load_basic_configuration(Path(str(conf_path)))
    assert data_manager.core_data_path().parent == Path(data_mgr_config['BASE_DIR'])