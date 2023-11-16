"""Tests to ensure that base vispy namespace functions correctly,
including configuration options.
"""
import vispy.app
from vispy.testing import requires_application, run_tests_if_main, assert_raises, assert_equal, assert_not_equal

@requires_application('pyside')
def test_use():
    if False:
        print('Hello World!')
    vispy.app.use_app()
    default_app = vispy.app._default_app.default_app
    vispy.app._default_app.default_app = None
    app_name = default_app.backend_name.split(' ')[0]
    try:
        assert_raises(TypeError, vispy.use)
        assert_equal(vispy.app._default_app.default_app, None)
        vispy.use(gl='gl2')
        assert_equal(vispy.app._default_app.default_app, None)
        vispy.use(app_name)
        assert_not_equal(vispy.app._default_app.default_app, None)
        wrong_name = 'glfw' if app_name.lower() != 'glfw' else 'pyqt4'
        assert_raises(RuntimeError, vispy.use, wrong_name)
        vispy.use(app_name, 'gl2')
    finally:
        vispy.app._default_app.default_app = default_app
run_tests_if_main()