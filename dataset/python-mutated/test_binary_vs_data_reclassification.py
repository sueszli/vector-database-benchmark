import os
import pathlib
import pytest
import PyInstaller.utils.misc as miscutils

def _create_test_data_file(filename):
    if False:
        print('Hello World!')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fp:
        fp.write('Test file')

def _create_test_binary(filename):
    if False:
        print('Hello World!')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    import _ctypes
    import shutil
    shutil.copy2(_ctypes.__file__, filename)

def _create_test_build(pyi_builder, tmpdir, datas=None, binaries=None):
    if False:
        i = 10
        return i + 15
    extra_args = []
    for (src_name, dest_name) in datas or []:
        extra_args += ['--add-data', f'{src_name}{os.pathsep}{dest_name}']
    for (src_name, dest_name) in binaries or []:
        extra_args += ['--add-binary', f'{src_name}{os.pathsep}{dest_name}']
    pyi_builder.test_source('\n        print("Hello world!")\n        ', pyi_args=extra_args)
    analysis_toc_file = list((pathlib.Path(tmpdir) / 'build/test_source').glob('Analysis-??.toc'))
    assert len(analysis_toc_file) == 1
    analysis_toc_file = analysis_toc_file[0]
    analysis_data = miscutils.load_py_data_struct(analysis_toc_file)
    return (analysis_data[14], analysis_data[17])

@pytest.mark.linux
@pytest.mark.win32
@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_automatic_reclassification_data_file(pyi_builder, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    binaries = []
    src_path = os.path.join(tmpdir, 'test_file')
    _create_test_data_file(src_path)
    binaries.append((src_path, '.'))
    (binaries_toc, datas_toc) = _create_test_build(pyi_builder, tmpdir, binaries=binaries)
    test_file_entries = [typecode for (dest_name, src_name, typecode) in binaries_toc if dest_name == 'test_file']
    assert len(test_file_entries) == 0
    test_file_entries = [typecode for (dest_name, src_name, typecode) in datas_toc if dest_name == 'test_file']
    assert len(test_file_entries) == 1
    assert test_file_entries[0] == 'DATA'

@pytest.mark.linux
@pytest.mark.win32
@pytest.mark.darwin
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_automatic_reclassification_binary(pyi_builder, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    datas = []
    src_path = os.path.join(tmpdir, 'test_file')
    _create_test_binary(src_path)
    datas.append((src_path, '.'))
    (binaries_toc, datas_toc) = _create_test_build(pyi_builder, tmpdir, datas=datas)
    test_file_entries = [typecode for (dest_name, src_name, typecode) in datas_toc if dest_name == 'test_file']
    assert len(test_file_entries) == 0
    test_file_entries = [typecode for (dest_name, src_name, typecode) in binaries_toc if dest_name == 'test_file']
    assert len(test_file_entries) == 1
    assert test_file_entries[0] == 'BINARY'