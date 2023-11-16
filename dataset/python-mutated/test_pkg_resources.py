from PyInstaller.utils.tests import importorskip

@importorskip('pkg_resources')
def test_pkg_resources_importable(pyi_builder):
    if False:
        i = 10
        return i + 15
    '\n    Check that a trivial example using pkg_resources does build.\n    '
    pyi_builder.test_source('\n        import pkg_resources\n        pkg_resources.working_set.require()\n        ')