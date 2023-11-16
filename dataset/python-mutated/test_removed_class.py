from custom_components.hacs.base import RemovedRepository

def test_removed():
    if False:
        i = 10
        return i + 15
    removed = RemovedRepository()
    assert isinstance(removed.to_json(), dict)