import eventlet
import pytest
from nameko.extensions import ProviderCollector

def test_provider_collector():
    if False:
        print('Hello World!')
    collector = ProviderCollector()
    provider1 = object()
    provider2 = object()
    collector.register_provider(provider1)
    collector.register_provider(provider2)
    assert provider1 in collector._providers
    assert provider2 in collector._providers
    collector.unregister_provider(provider1)
    assert provider1 not in collector._providers
    collector.unregister_provider(provider1)
    with pytest.raises(eventlet.Timeout):
        with eventlet.Timeout(0):
            collector.stop()
    with eventlet.Timeout(0):
        collector.unregister_provider(provider2)
        collector.stop()