from PyInstaller.utils.tests import importorskip

def test_importlib_lazy_loader(pyi_builder):
    if False:
        return 10
    pyi_builder.test_script('pyi_lazy_import.py', app_args=['json'], pyi_args=['--hiddenimport', 'json'])

@importorskip('pkg_resources._vendor.jaraco.text')
def test_importlib_lazy_loader_alias1(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_script('pyi_lazy_import.py', app_args=['pkg_resources._vendor.jaraco.text'], pyi_args=['--hiddenimport', 'pkg_resources'])

@importorskip('pkg_resources.extern.jaraco.text')
def test_importlib_lazy_loader_alias2(pyi_builder):
    if False:
        return 10
    pyi_builder.test_script('pyi_lazy_import.py', app_args=['pkg_resources.extern.jaraco.text'], pyi_args=['--hiddenimport', 'pkg_resources'])