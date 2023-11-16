from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['test', 'distutils'], pytest_assert_rewrites=False)
def test_distutils(selenium):
    if False:
        while True:
            i = 10
    import sys
    import unittest
    import unittest.mock
    from test import libregrtest
    name = 'test_distutils'
    ignore_tests = ['test_check_environ_getpwuid', 'test_get_platform', 'test_simple_built', 'test_optional_extension', 'test_customize_compiler_before_get_config_vars', 'test_spawn', 'test_debug_mode', 'test_record', 'test_get_config_h_filename', 'test_srcdir', 'test_mkpath_with_custom_mode', 'test_finalize_options']
    sys.modules['_osx_support'] = unittest.mock.Mock()
    try:
        libregrtest.main([name], ignore_tests=ignore_tests, verbose=True, verbose3=True)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f'Failed with code: {e.code}') from None