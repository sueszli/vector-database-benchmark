import os
import plistlib
import pytest
from PyInstaller.utils.tests import importorskip
from PyInstaller.building.osx import DOT_REPLACEMENT

@pytest.mark.darwin
@importorskip('psutil')
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_signing_metadata(pyi_builder, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setenv('PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR', '1')
    monkeypatch.setenv('PYINSTALLER_VERIFY_BUNDLE_SIGNATURE', '1')
    pyi_builder.test_source('\n        import psutil\n        ', pyi_args=['--windowed', '--copy-metadata', 'psutil'])

@pytest.mark.darwin
@importorskip('psutil')
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_signing_py_files(pyi_builder, monkeypatch):
    if False:
        return 10
    monkeypatch.setenv('PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR', '1')
    monkeypatch.setenv('PYINSTALLER_VERIFY_BUNDLE_SIGNATURE', '1')

    def AnalysisOverride(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['module_collection_mode'] = {'psutil': 'py'}
        return Analysis(*args, **kwargs)
    import PyInstaller.building.build_main
    Analysis = PyInstaller.building.build_main.Analysis
    monkeypatch.setattr('PyInstaller.building.build_main.Analysis', AnalysisOverride)
    pyi_builder.test_source('\n        import psutil\n        ', pyi_args=['--windowed'])

@pytest.mark.darwin
@importorskip('psutil')
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_signing_pyc_files(pyi_builder, monkeypatch):
    if False:
        return 10
    monkeypatch.setenv('PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR', '1')
    monkeypatch.setenv('PYINSTALLER_VERIFY_BUNDLE_SIGNATURE', '1')

    def AnalysisOverride(*args, **kwargs):
        if False:
            return 10
        kwargs['module_collection_mode'] = {'psutil': 'pyc'}
        return Analysis(*args, **kwargs)
    import PyInstaller.building.build_main
    Analysis = PyInstaller.building.build_main.Analysis
    monkeypatch.setattr('PyInstaller.building.build_main.Analysis', AnalysisOverride)
    pyi_builder.test_source('\n        import psutil\n        ', pyi_args=['--windowed'])

def _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=None, binaries=None):
    if False:
        print('Hello World!')
    monkeypatch.setenv('PYINSTALLER_STRICT_BUNDLE_CODESIGN_ERROR', '1')
    monkeypatch.setenv('PYINSTALLER_VERIFY_BUNDLE_SIGNATURE', '1')
    extra_args = []
    for (src_name, dest_name) in datas or []:
        extra_args += ['--add-data', f'{src_name}{os.pathsep}{dest_name}']
    for (src_name, dest_name) in binaries or []:
        extra_args += ['--add-binary', f'{src_name}{os.pathsep}{dest_name}']
    pyi_builder.test_source('\n        print("Hello world!")\n        ', pyi_args=['--windowed', *extra_args])
    return os.path.join(tmpdir, 'dist/test_source.app')

def _create_test_data_file(filename):
    if False:
        for i in range(10):
            print('nop')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fp:
        fp.write('Test file')

def _create_test_binary(filename):
    if False:
        while True:
            i = 10
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    import _struct
    import shutil
    shutil.copy2(_struct.__file__, filename)

def _create_test_framework(bundle_path):
    if False:
        return 10
    assert bundle_path.endswith('.framework')
    binary_name = os.path.basename(bundle_path)[:-10]
    binary_path = os.path.join(bundle_path, 'Versions/A', binary_name)
    _create_test_binary(os.path.join(binary_path))
    resources_path = os.path.join(bundle_path, 'Versions/A/Resources')
    os.makedirs(resources_path, exist_ok=True)
    info_plist_file = os.path.join(bundle_path, 'Versions/A/Resources/Info.plist')
    info_plist = {'CFBundleExecutable': binary_name, 'CFBundleIdentifier': f'org.pyinstaller.{binary_name}', 'CFBundlePackageType': 'FMWK', 'CFBundleShortVersionString': '1.0', 'CFBundleSignature': '????', 'CFBundleVersion': '1.0.0'}
    with open(info_plist_file, 'wb') as fp:
        plistlib.dump(info_plist, fp)

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_data_file(pyi_builder, monkeypatch, tmpdir):
    if False:
        print('Hello World!')
    datas = []
    src_path = os.path.join(tmpdir, 'data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas)
    filename = os.path.join(bundle_path, 'Contents/Resources/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/data_file.txt')
    assert os.path.islink(filename)
    assert os.path.isfile(filename)
    assert os.readlink(filename) == '../Resources/data_file.txt'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_binary(pyi_builder, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    binaries = []
    src_path = os.path.join(tmpdir, 'binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary.dylib')
    assert os.path.islink(filename)
    assert os.path.isfile(filename)
    assert os.readlink(filename) == '../Frameworks/binary.dylib'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_data_only_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    datas = []
    src_path = os.path.join(tmpdir, 'data_dir/data_file1.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'data_dir'))
    src_path = os.path.join(tmpdir, 'data_dir/data_file2.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'data_dir'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas)
    filename = os.path.join(bundle_path, 'Contents/Resources/data_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/data_dir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../Resources/data_dir'
    filename = os.path.join(bundle_path, 'Contents/Resources/data_dir/data_file1.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/data_dir/data_file1.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/data_dir/data_file2.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/data_dir/data_file2.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_binary_only_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    binaries = []
    src_path = os.path.join(tmpdir, 'binary_dir/binary1.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'binary_dir'))
    src_path = os.path.join(tmpdir, 'binary_dir/binary2.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'binary_dir'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../Frameworks/binary_dir'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/binary1.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/binary1.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/binary2.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/binary2.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_mixed_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'mixed_dir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed_dir'))
    src_path = os.path.join(tmpdir, 'mixed_dir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'mixed_dir'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Resources/mixed_dir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Frameworks/mixed_dir/binary.dylib'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_mixed_dir_with_subdirs(pyi_builder, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'mixed_dir/data_subdir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed_dir/data_subdir'))
    src_path = os.path.join(tmpdir, 'mixed_dir/binary_subdir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'mixed_dir/binary_subdir'))
    src_path = os.path.join(tmpdir, 'mixed_dir/mixed_subdir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed_dir/mixed_subdir'))
    src_path = os.path.join(tmpdir, 'mixed_dir/mixed_subdir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'mixed_dir/mixed_subdir'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/data_subdir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/data_subdir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Resources/mixed_dir/data_subdir'
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/data_subdir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/data_subdir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/binary_subdir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/binary_subdir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Frameworks/mixed_dir/binary_subdir'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/binary_subdir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/binary_subdir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/mixed_subdir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/mixed_subdir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/mixed_subdir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/mixed_subdir/data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../../Resources/mixed_dir/mixed_subdir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/mixed_subdir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/mixed_subdir/binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../../Frameworks/mixed_dir/mixed_subdir/binary.dylib'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_mixed_dir_with_subdirs_and_dots(pyi_builder, monkeypatch, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'mixed.dir/.data_subdir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed.dir/.data_subdir'))
    src_path = os.path.join(tmpdir, 'mixed.dir/.binary_subdir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'mixed.dir/.binary_subdir'))
    src_path = os.path.join(tmpdir, 'mixed.dir/mixed_subdir./data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed.dir/mixed_subdir.'))
    src_path = os.path.join(tmpdir, 'mixed.dir/mixed_subdir./binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'mixed.dir/mixed_subdir.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, f'Contents/Frameworks/mixed{DOT_REPLACEMENT}dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed__dot__dir'
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, f'Contents/Resources/mixed{DOT_REPLACEMENT}dir')
    assert not os.path.exists(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/.data_subdir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, f'Contents/Resources/mixed.dir/{DOT_REPLACEMENT}data_subdir')
    assert not os.path.exists(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/.data_subdir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Resources/mixed.dir/.data_subdir'
    filename = os.path.join(bundle_path, f'Contents/Frameworks/mixed.dir/{DOT_REPLACEMENT}data_subdir')
    assert not os.path.exists(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/.data_subdir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/.data_subdir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, f'Contents/Frameworks/mixed.dir/{DOT_REPLACEMENT}binary_subdir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/.binary_subdir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == f'{DOT_REPLACEMENT}binary_subdir'
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/.binary_subdir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Frameworks/mixed.dir/.binary_subdir'
    filename = os.path.join(bundle_path, f'Contents/Resources/mixed.dir/{DOT_REPLACEMENT}binary_subdir')
    assert not os.path.exists(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/.binary_subdir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/.binary_subdir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, f'Contents/Frameworks/mixed.dir/mixed_subdir{DOT_REPLACEMENT}')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/mixed_subdir.')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == f'mixed_subdir{DOT_REPLACEMENT}'
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/mixed_subdir.')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, f'Contents/Resources/mixed.dir/mixed_subdir{DOT_REPLACEMENT}')
    assert not os.path.exists(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/mixed_subdir./data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/mixed_subdir./data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../../Resources/mixed.dir/mixed_subdir./data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed.dir/mixed_subdir./binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed.dir/mixed_subdir./binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../../Frameworks/mixed.dir/mixed_subdir./binary.dylib'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_symlink_into_data_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        i = 10
        return i + 15
    datas = []
    src_path = os.path.join(tmpdir, 'data_dir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'data_dir'))
    src_path = os.path.join(tmpdir, 'link_to_data_file.txt')
    os.symlink('data_dir/data_file.txt', src_path)
    datas.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas)
    filename = os.path.join(bundle_path, 'Contents/Resources/data_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/data_dir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../Resources/data_dir'
    filename = os.path.join(bundle_path, 'Contents/Resources/data_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/data_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/link_to_data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'data_dir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Resources/link_to_data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'data_dir/data_file.txt'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_symlink_into_binary_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        print('Hello World!')
    binaries = []
    src_path = os.path.join(tmpdir, 'binary_dir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'binary_dir'))
    src_path = os.path.join(tmpdir, 'link_to_binary.dylib')
    os.symlink('binary_dir/binary.dylib', src_path)
    binaries.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../Frameworks/binary_dir'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/link_to_binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'binary_dir/binary.dylib'
    filename = os.path.join(bundle_path, 'Contents/Resources/link_to_binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'binary_dir/binary.dylib'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_symlink_into_mixed_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        return 10
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'mixed_dir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed_dir'))
    src_path = os.path.join(tmpdir, 'mixed_dir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'mixed_dir'))
    src_path = os.path.join(tmpdir, 'link_to_data_file.txt')
    os.symlink('mixed_dir/data_file.txt', src_path)
    datas.append((src_path, '.'))
    src_path = os.path.join(tmpdir, 'link_to_binary.dylib')
    os.symlink('mixed_dir/binary.dylib', src_path)
    binaries.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Resources/mixed_dir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Frameworks/mixed_dir/binary.dylib'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/link_to_data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed_dir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Resources/link_to_data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed_dir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/link_to_binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed_dir/binary.dylib'
    filename = os.path.join(bundle_path, 'Contents/Resources/link_to_binary.dylib')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed_dir/binary.dylib'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_framework_in_top_level(pyi_builder, monkeypatch, tmpdir):
    if False:
        i = 10
        return i + 15
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'Dummy.framework')
    _create_test_framework(src_path)
    binaries.append((os.path.join(src_path, 'Versions/A/Dummy'), 'Dummy.framework/Versions/A'))
    src_path = os.path.join(tmpdir, 'Dummy')
    os.symlink('Dummy.framework/Versions/A/Dummy', src_path)
    binaries.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../Frameworks/Dummy.framework'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework/Versions')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework/Versions')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework/Versions/A')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework/Versions/A')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework/Versions/A/Dummy')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework/Versions/A/Dummy')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework/Versions/A/Resources')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework/Versions/A/Resources')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework/Versions/A/Resources/Info.plist')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework/Versions/A/Resources/Info.plist')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy.framework/Versions/Current')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'A'
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy.framework/Versions/Current')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'A'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'Dummy.framework/Versions/A/Dummy'
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'Dummy.framework/Versions/A/Dummy'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_framework_in_binary_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        print('Hello World!')
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'binary_dir/binary.dylib')
    _create_test_binary(src_path)
    binaries.append((src_path, 'binary_dir'))
    src_path = os.path.join(tmpdir, 'binary_dir/Dummy.framework')
    _create_test_framework(src_path)
    binaries.append((os.path.join(src_path, 'Versions/A/Dummy'), 'binary_dir/Dummy.framework/Versions/A'))
    src_path = os.path.join(tmpdir, 'Dummy')
    os.symlink('binary_dir/Dummy.framework/Versions/A/Dummy', src_path)
    binaries.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../Frameworks/binary_dir'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/binary.dylib')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework/Versions')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework/Versions')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework/Versions/A')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework/Versions/A')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework/Versions/A/Dummy')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework/Versions/A/Dummy')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework/Versions/A/Resources')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework/Versions/A/Resources')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework/Versions/A/Resources/Info.plist')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework/Versions/A/Resources/Info.plist')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/binary_dir/Dummy.framework/Versions/Current')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'A'
    filename = os.path.join(bundle_path, 'Contents/Resources/binary_dir/Dummy.framework/Versions/Current')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'A'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'binary_dir/Dummy.framework/Versions/A/Dummy'
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'binary_dir/Dummy.framework/Versions/A/Dummy'

@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_macos_bundle_layout_framework_in_mixed_dir(pyi_builder, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    datas = []
    binaries = []
    src_path = os.path.join(tmpdir, 'mixed_dir/data_file.txt')
    _create_test_data_file(src_path)
    datas.append((src_path, 'mixed_dir'))
    src_path = os.path.join(tmpdir, 'mixed_dir/Dummy.framework')
    _create_test_framework(src_path)
    binaries.append((os.path.join(src_path, 'Versions/A/Dummy'), 'mixed_dir/Dummy.framework/Versions/A'))
    src_path = os.path.join(tmpdir, 'Dummy')
    os.symlink('mixed_dir/Dummy.framework/Versions/A/Dummy', src_path)
    binaries.append((src_path, '.'))
    bundle_path = _create_app_bundle(pyi_builder, monkeypatch, tmpdir, datas=datas, binaries=binaries)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/data_file.txt')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Resources/mixed_dir/data_file.txt'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == '../../Frameworks/mixed_dir/Dummy.framework'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework/Versions')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework/Versions')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework/Versions/A')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework/Versions/A')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework/Versions/A/Dummy')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework/Versions/A/Dummy')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework/Versions/A/Resources')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework/Versions/A/Resources')
    assert os.path.isdir(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework/Versions/A/Resources/Info.plist')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework/Versions/A/Resources/Info.plist')
    assert os.path.isfile(filename)
    assert not os.path.islink(filename)
    filename = os.path.join(bundle_path, 'Contents/Frameworks/mixed_dir/Dummy.framework/Versions/Current')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'A'
    filename = os.path.join(bundle_path, 'Contents/Resources/mixed_dir/Dummy.framework/Versions/Current')
    assert os.path.isdir(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'A'
    filename = os.path.join(bundle_path, 'Contents/Frameworks/Dummy')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed_dir/Dummy.framework/Versions/A/Dummy'
    filename = os.path.join(bundle_path, 'Contents/Resources/Dummy')
    assert os.path.isfile(filename)
    assert os.path.islink(filename)
    assert os.readlink(filename) == 'mixed_dir/Dummy.framework/Versions/A/Dummy'