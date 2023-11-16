from unittest.mock import patch
from hamcrest import assert_that, calling, instance_of, is_, raises
from torch.utils.data import DataLoader
from deepchecks import vision
from deepchecks.vision.datasets.detection.coco_torch import COCO_DIR, CocoDataset, load_dataset
from deepchecks.vision.datasets.detection.coco_utils import download_coco128

def patch_side_effect(*args, **kwargs):
    if False:
        while True:
            i = 10
    return download_coco128(*args, **kwargs)

def load_dataset_test(mock_download_and_extract_archive):
    if False:
        while True:
            i = 10

    def verify(loader):
        if False:
            for i in range(10):
                print('nop')
        assert_that(loader, instance_of(DataLoader))
        assert_that(loader.dataset, instance_of(CocoDataset))
        assert_that(loader.dataset.train is True)
        assert_that((COCO_DIR / 'coco128' / 'images').exists())
        assert_that((COCO_DIR / 'coco128' / 'labels').exists())
    if not (COCO_DIR / 'coco128').exists():
        loader = load_dataset(train=True, object_type='DataLoader')
        verify(loader)
        mock_download_and_extract_archive.reset_mock()
        load_dataset_test(mock_download_and_extract_archive)
    else:
        loader = load_dataset(train=True, object_type='DataLoader')
        assert_that(mock_download_and_extract_archive.called, is_(False))
        verify(loader)
        assert_that(loader, instance_of(DataLoader))

@patch('deepchecks.vision.datasets.detection.coco_utils.download_coco128')
def test_load_dataset(mock_download_and_extract_archive):
    if False:
        print('Hello World!')
    mock_download_and_extract_archive.side_effect = patch_side_effect
    load_dataset_test(mock_download_and_extract_archive)

def test_deepchecks_dataset_load():
    if False:
        while True:
            i = 10
    loader = load_dataset(train=True, object_type='VisionData')
    assert_that(loader, instance_of(vision.VisionData))

def test__load_dataset__func_with_unknow_object_type_parameter():
    if False:
        print('Hello World!')
    assert_that(calling(load_dataset).with_args(object_type='<unknonw>'), raises(TypeError))

def test_train_test_split():
    if False:
        while True:
            i = 10
    train = load_dataset(train=True, object_type='DataLoader')
    test = load_dataset(train=False, object_type='DataLoader')
    assert_that(len(train.dataset) + len(test.dataset) == 128)
    train_images = set((it.name for it in train.dataset.images))
    test_images = set((it.name for it in test.dataset.images))
    intersection = train_images.intersection(test_images)
    assert_that(len(intersection) == 0)