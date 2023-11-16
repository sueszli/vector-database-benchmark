from direct.dist.FreezeTool import Freezer, PandaModuleFinder
import pytest
import os
import sys
import subprocess
import platform

def test_Freezer_moduleSuffixes():
    if False:
        print('Hello World!')
    freezer = Freezer()
    for (suffix, mode, type) in freezer.mf.suffixes:
        if type == 2:
            assert mode == 'rb'

def test_Freezer_getModulePath_getModuleStar(tmp_path):
    if False:
        while True:
            i = 10
    package1 = tmp_path / 'package1'
    package1.mkdir()
    (package1 / 'submodule1.py').write_text('')
    (package1 / '__init__.py').write_text('')
    package2 = tmp_path / 'package2'
    package2.mkdir()
    (package2 / 'submodule2.py').write_text('')
    (package2 / '__init__.py').write_text('raise ImportError\n')
    (tmp_path / 'module1.py').write_text('')
    (tmp_path / 'module2.py').write_text('raise ImportError\n')
    backup = sys.path
    try:
        sys.path = [str(tmp_path / 'nonexistent'), str(tmp_path)]
        freezer = Freezer()
        assert freezer.getModulePath('nonexist') == None
        assert freezer.getModulePath('package1') == [str(package1)]
        assert freezer.getModulePath('package2') == [str(package2)]
        assert freezer.getModulePath('package1.submodule1') == None
        assert freezer.getModulePath('package1.nonexist') == None
        assert freezer.getModulePath('package2.submodule2') == None
        assert freezer.getModulePath('package2.nonexist') == None
        assert freezer.getModulePath('module1') == None
        assert freezer.getModulePath('module2') == None
        assert freezer.getModuleStar('nonexist') == None
        assert freezer.getModuleStar('package1') == ['submodule1']
        assert freezer.getModuleStar('package2') == ['submodule2']
        assert freezer.getModuleStar('package1.submodule1') == None
        assert freezer.getModuleStar('package1.nonexist') == None
        assert freezer.getModuleStar('package2.submodule2') == None
        assert freezer.getModuleStar('package2.nonexist') == None
        assert freezer.getModuleStar('module1') == None
        assert freezer.getModuleStar('module2') == None
    finally:
        sys.path = backup

@pytest.mark.parametrize('use_console', (False, True))
def test_Freezer_generateRuntimeFromStub(tmp_path, use_console):
    if False:
        for i in range(10):
            print('nop')
    try:
        import panda3d_tools
        bin_dir = os.path.dirname(panda3d_tools.__file__)
    except:
        import panda3d
        bin_dir = os.path.join(os.path.dirname(os.path.dirname(panda3d.__file__)), 'bin')
    if sys.platform == 'win32':
        suffix = '.exe'
    else:
        suffix = ''
    if not use_console:
        stub_file = os.path.join(bin_dir, 'deploy-stubw' + suffix)
    if use_console or not os.path.isfile(stub_file):
        stub_file = os.path.join(bin_dir, 'deploy-stub' + suffix)
    if not os.path.isfile(stub_file):
        pytest.skip('Unable to find deploy-stub executable')
    target = str(tmp_path / ('stubtest' + suffix))
    freezer = Freezer()
    freezer.addModule('site', filename='site.py', text='import sys\nsys.frozen=True')
    freezer.addModule('module2', filename='module2.py', text='print("Module imported")')
    freezer.addModule('__main__', filename='main.py', text='import module2\nprint("Hello world")')
    assert '__main__' in freezer.modules
    freezer.done(addStartupModules=True)
    assert '__main__' in dict(freezer.getModuleDefs())
    freezer.generateRuntimeFromStub(target, open(stub_file, 'rb'), use_console)
    if sys.platform == 'darwin' and platform.machine().lower() == 'arm64':
        return
    output = subprocess.check_output(target)
    assert output.replace(b'\r\n', b'\n') == b'Module imported\nHello world\n'