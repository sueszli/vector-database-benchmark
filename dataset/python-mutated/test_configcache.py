"""Tests for qutebrowser.config.configcache."""
import pytest
from qutebrowser.config import config

def test_configcache_except_pattern(config_stub):
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError):
        assert config.cache['content.javascript.enabled']

def test_configcache_error_set(config_stub):
    if False:
        return 10
    with pytest.raises(TypeError):
        config.cache['content.javascript.enabled'] = True

def test_configcache_get(config_stub):
    if False:
        print('Hello World!')
    assert len(config.cache._cache) == 0
    assert not config.cache['auto_save.session']
    assert len(config.cache._cache) == 1
    assert not config.cache['auto_save.session']

def test_configcache_get_after_set(config_stub):
    if False:
        print('Hello World!')
    assert not config.cache['auto_save.session']
    config_stub.val.auto_save.session = True
    assert config.cache['auto_save.session']

def test_configcache_naive_benchmark(config_stub, benchmark):
    if False:
        return 10

    def _run_bench():
        if False:
            i = 10
            return i + 15
        for _i in range(10000):
            config.cache['tabs.padding']
            config.cache['tabs.indicator.width']
            config.cache['tabs.indicator.padding']
            config.cache['tabs.min_width']
            config.cache['tabs.max_width']
            config.cache['tabs.pinned.shrink']
    benchmark(_run_bench)