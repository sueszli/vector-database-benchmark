import pytest
import salt.executors.splay as splay_exec

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {splay_exec: {'__grains__': {'id': 'foo'}}}

def test__get_hash():
    if False:
        while True:
            i = 10
    assert splay_exec._get_hash()