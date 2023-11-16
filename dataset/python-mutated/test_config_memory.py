from apprise.config.ConfigMemory import ConfigMemory
import logging
logging.disable(logging.CRITICAL)

def test_config_memory():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: ConfigMemory() object\n\n    '
    assert ConfigMemory.parse_url('garbage://') is None
    cm = ConfigMemory(content='json://localhost', format='text')
    assert len(cm) == 1
    assert isinstance(cm.url(), str) is True
    assert isinstance(cm.read(), str) is True
    cm = ConfigMemory(content='json://localhost')
    assert len(cm) == 1
    assert isinstance(cm.url(), str) is True
    assert isinstance(cm.read(), str) is True
    assert len(ConfigMemory(content='garbage')) == 0