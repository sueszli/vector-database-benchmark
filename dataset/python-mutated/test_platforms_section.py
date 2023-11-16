import pytest
from molecule.model import schema_v3

@pytest.mark.parametrize('_config', ['_model_platforms_delegated_section_data'], indirect=True)
def test_platforms_delegated(_config):
    if False:
        return 10
    assert not schema_v3.validate(_config)