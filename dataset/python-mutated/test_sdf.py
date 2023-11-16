import gc
import numpy as np
from numpy.testing import assert_allclose
from vispy.app import Canvas
from vispy.visuals.text.text import SDFRendererCPU
from vispy.visuals.text._sdf_gpu import SDFRendererGPU
from vispy import gloo
from vispy.testing import requires_application, run_tests_if_main

@requires_application()
def test_sdf():
    if False:
        while True:
            i = 10
    'Test basic text support - sdf'
    data = (np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0]]) * 255).astype(np.uint8)
    gpu = np.array([[105, 110, 112, 112, 112, 112, 112, 110, 105], [110, 117, 120, 120, 120, 120, 120, 117, 110], [112, 120, 128, 128, 128, 128, 128, 120, 112], [112, 120, 128, 136, 144, 136, 128, 120, 112], [112, 120, 128, 136, 144, 136, 128, 120, 112], [112, 120, 128, 136, 144, 136, 128, 120, 112]])
    cpu = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 115, 118, 118, 118, 118, 118, 115, 0], [0, 118, 137, 137, 137, 137, 137, 118, 0], [0, 118, 137, 143, 143, 143, 137, 118, 0], [0, 118, 137, 143, 149, 143, 137, 118, 0], [0, 0, 255, 255, 255, 255, 255, 0, 0]])
    for (Rend, expd) in zip((SDFRendererGPU, SDFRendererCPU), (gpu, cpu)):
        with Canvas(size=(100, 100)) as c:
            tex = gloo.Texture2D(data.shape + (3,), format='rgb')
            Rend().render_to_texture(data, tex, (0, 0), data.shape[::-1])
            gloo.set_viewport(0, 0, *data.shape[::-1])
            gloo.util.draw_texture(tex)
            result = gloo.util._screenshot()[:, :, 0].astype(np.int64)
            assert_allclose(result, expd, atol=1, err_msg=Rend.__name__)
            del tex, result
        del c
        gc.collect()
run_tests_if_main()