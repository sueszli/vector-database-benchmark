import os
import requests
from utils_cv.action_recognition.data import _DatasetSpec, Urls
from utils_cv.common.data import data_path

def test__DatasetSpec_kinetics():
    if False:
        i = 10
        return i + 15
    ' Tests DatasetSpec initialize with kinetics classes '
    kinetics = _DatasetSpec(Urls.kinetics_label_map, 400)
    kinetics.class_names
    assert os.path.exists(str(data_path() / 'label_map.txt'))

def test__DatasetSpec_hmdb():
    if False:
        i = 10
        return i + 15
    ' Tests DatasetSpec initialize with hmdb51 classes '
    hmdb51 = _DatasetSpec(Urls.hmdb51_label_map, 51)
    hmdb51.class_names
    assert os.path.exists(str(data_path() / 'label_map.txt'))

def test_urls():
    if False:
        for i in range(10):
            print('nop')
    ' Test that urls work '
    for (attr, value) in Urls.__dict__.items():
        if not str.startswith(attr, '__') and 'base' not in attr:
            with requests.get(value):
                pass