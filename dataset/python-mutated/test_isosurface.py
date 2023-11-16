import numpy as np
from vispy import scene
from vispy.testing import run_tests_if_main, requires_pyopengl

@requires_pyopengl()
def test_isosurface():
    if False:
        return 10
    vol = np.arange(1000).reshape((10, 10, 10)).astype(np.float32)
    iso = scene.visuals.Isosurface(vol, level=200)
    iso.color = (1.0, 0.8, 0.9, 1.0)
run_tests_if_main()