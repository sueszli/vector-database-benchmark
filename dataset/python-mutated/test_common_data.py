import os
import pytest
import shutil
from pathlib import Path
from typing import Union
from utils_cv.classification.data import Urls
from utils_cv.common.data import data_path, get_files_in_directory, unzip_url, unzip_urls, root_path

def test_root_path():
    if False:
        for i in range(10):
            print('nop')
    s = root_path()
    assert isinstance(s, Path) and s != '' and os.path.isdir(str(s / 'utils_cv'))

def test_data_path():
    if False:
        i = 10
        return i + 15
    s = data_path()
    assert isinstance(s, Path) and s != '' and os.path.isdir(str(s)) and (s.name == 'data')

def test_get_files_in_directory(tiny_ic_data_path):
    if False:
        print('Hello World!')
    im_dir = os.path.join(tiny_ic_data_path, 'can')
    Path(os.path.join(im_dir, 'image.not_jpg')).touch()
    os.makedirs(os.path.join(im_dir, 'test_get_files_in_directory'), exist_ok=True)
    assert len(get_files_in_directory(im_dir)) == 23
    assert len(get_files_in_directory(im_dir, suffixes=['.jpg'])) == 22
    assert len(get_files_in_directory(im_dir, suffixes=['.not_jpg'])) == 1
    assert len(get_files_in_directory(im_dir, suffixes=['.nonsense'])) == 0
    os.remove(os.path.join(im_dir, 'image.not_jpg'))
    os.rmdir(os.path.join(im_dir, 'test_get_files_in_directory'))

def _test_url_data(url: str, path: Union[Path, str], dir_name: str):
    if False:
        i = 10
        return i + 15
    dest_path = os.path.join(path, dir_name)
    assert not os.path.isdir(dest_path)
    unzipped_path = unzip_url(url, fpath=path, dest=path, exist_ok=True)
    assert os.path.exists(os.path.join(path, f'{dir_name}.zip'))
    assert os.path.exists(dest_path)
    assert os.path.realpath(dest_path) == os.path.realpath(unzipped_path)

def test_unzip_url_rel_path(tmp_path):
    if False:
        i = 10
        return i + 15
    ' Test unzip with relative path. '
    rel_path = tmp_path
    _test_url_data(Urls.fridge_objects_path, rel_path, 'fridgeObjects')

def test_unzip_url_abs_path(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    ' Test unzip with absolute path. '
    abs_path = Path(os.path.abspath(tmp_path))
    _test_url_data(Urls.fridge_objects_path, abs_path, 'fridgeObjects')

def test_unzip_url_exist_ok(tmp_path):
    if False:
        while True:
            i = 10
    '\n    Test if exist_ok is true and (file exists, file does not exist)\n    '
    os.makedirs(tmp_path / 'fridgeObjects')
    fridge_object_path = unzip_url(Urls.fridge_objects_path, fpath=tmp_path, dest=tmp_path, exist_ok=True)
    assert len(os.listdir(fridge_object_path)) == 0
    shutil.rmtree(tmp_path / 'fridgeObjects')
    fridge_object_path = unzip_url(Urls.fridge_objects_watermark_path, fpath=tmp_path, dest=tmp_path, exist_ok=True)
    assert len(os.listdir(fridge_object_path)) == 4

def test_unzip_url_not_exist_ok(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Test if exist_ok is false and (file exists, file does not exist)\n    '
    os.makedirs(tmp_path / 'fridgeObjects')
    with pytest.raises(FileExistsError):
        unzip_url(Urls.fridge_objects_path, fpath=tmp_path, dest=tmp_path, exist_ok=False)
    shutil.rmtree(tmp_path / 'fridgeObjects')
    os.remove(tmp_path / 'fridgeObjects.zip')
    fridge_object_path = unzip_url(Urls.fridge_objects_path, fpath=tmp_path, dest=tmp_path, exist_ok=False)
    assert len(os.listdir(fridge_object_path)) == 4

def test_unzip_urls(tmp_path):
    if False:
        return 10
    result_paths = unzip_urls([Urls.fridge_objects_tiny_path, Urls.fridge_objects_watermark_path, Urls.fridge_objects_negatives_path], tmp_path)
    assert len(result_paths) == 3
    for p in result_paths:
        assert os.path.isdir(p)