import pytest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pedalboard
TESTABLE_PLUGINS = []
for plugin_class in pedalboard.Plugin.__subclasses__():
    try:
        plugin_class()
        TESTABLE_PLUGINS.append(plugin_class)
    except Exception:
        pass

@pytest.mark.parametrize('plugin_class', TESTABLE_PLUGINS)
def test_concurrent_processing_produces_identical_audio(plugin_class):
    if False:
        return 10
    num_concurrent_plugins = 10
    sr = 48000
    plugins = [plugin_class() for _ in range(num_concurrent_plugins)]
    noise = np.random.rand(1, sr * 10)
    expected_output = plugins[0].process(noise, sr)
    if not np.allclose(expected_output, plugins[0].process(noise, sr)):
        return
    futures = []
    with ThreadPoolExecutor(max_workers=num_concurrent_plugins) as e:
        for plugin in plugins:
            futures.append(e.submit(plugin.process, noise, sample_rate=sr))
        processed = [future.result(timeout=10 * num_concurrent_plugins) for future in futures]
    for result in processed:
        np.testing.assert_allclose(expected_output, result)