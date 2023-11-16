def _pyi_rthook():
    if False:
        for i in range(10):
            print('nop')

    def _install_setuptools_distutils_hack():
        if False:
            print('Hello World!')
        import os
        import setuptools
        setuptools_major = int(setuptools.__version__.split('.')[0])
        default_value = 'stdlib' if setuptools_major < 60 else 'local'
        if os.environ.get('SETUPTOOLS_USE_DISTUTILS', default_value) == 'local':
            import _distutils_hack
            _distutils_hack.add_shim()
    try:
        _install_setuptools_distutils_hack()
    except Exception:
        pass
_pyi_rthook()
del _pyi_rthook