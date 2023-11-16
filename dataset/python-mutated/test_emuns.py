""""Test enums."""
from custom_components.hacs.enums import RepositoryFile

def test_enum_value():
    if False:
        return 10
    'Test enum value.'
    assert RepositoryFile.HACS_JSON == 'hacs.json'
    assert RepositoryFile.HACS_JSON.value == 'hacs.json'
    assert str(RepositoryFile.HACS_JSON) == 'hacs.json'