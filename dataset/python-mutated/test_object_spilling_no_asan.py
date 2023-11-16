import os
import platform
import sys
import numpy as np
import pytest
import ray

@pytest.mark.skipif(platform.system() == 'Windows', reason='Hangs on Windows.')
def test_spill_fusion(fs_only_object_spilling_config, shutdown_only):
    if False:
        return 10
    (object_spilling_config, temp_folder) = fs_only_object_spilling_config
    min_spilling_size = 10 * 1024 * 1024
    ray.init(num_cpus=1, object_store_memory=75 * 1024 * 1024, _system_config={'max_io_workers': 1, 'object_spilling_config': object_spilling_config, 'min_spilling_size': min_spilling_size, 'object_spilling_threshold': 0.8, 'object_store_full_delay_ms': 1000})
    object_size = 1024 * 1024
    xs = [ray.put(np.zeros(object_size // 8)) for _ in range(300)]
    spill_dir = os.path.join(temp_folder, ray._private.ray_constants.DEFAULT_OBJECT_PREFIX)
    (under_min, over_min) = (0, 0)
    for filename in os.listdir(spill_dir):
        size = os.stat(os.path.join(spill_dir, filename)).st_size
        if size < 2 * object_size // 8:
            under_min += 1
        else:
            over_min += 1
    assert over_min > under_min
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))