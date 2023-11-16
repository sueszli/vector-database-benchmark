import pathlib
from shutil import rmtree
import numpy as np
import PIL.Image as pilimage
import pytest
from hamcrest import all_of, assert_that, calling, equal_to, has_length, has_property, instance_of, raises
from torch.utils.data import DataLoader
from deepchecks.utils.strings import get_random_string
from deepchecks.vision.vision_data import simple_classification_data as simple
PARENT_FOLDER = pathlib.Path(__file__).absolute().parent

@pytest.fixture
def correct_images_folder():
    if False:
        for i in range(10):
            print('nop')
    image = pilimage.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 2)
    images_folder = PARENT_FOLDER / get_random_string()
    images_folder.mkdir()
    (images_folder / 'train').mkdir()
    (images_folder / 'train' / 'class1').mkdir()
    (images_folder / 'test').mkdir()
    (images_folder / 'test' / 'class1').mkdir()
    image.save(str(images_folder / 'train' / 'class1' / 'image.png'))
    image.save(str(images_folder / 'test' / 'class1' / 'image.png'))
    yield images_folder
    rmtree(str(images_folder))

@pytest.fixture
def incorrect_images_folder():
    if False:
        print('Hello World!')
    image = pilimage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    images_folder = PARENT_FOLDER / get_random_string()
    images_folder.mkdir()
    (images_folder / 'train').mkdir()
    (images_folder / 'test').mkdir()
    image.save(str(images_folder / 'train' / 'image.png'))
    image.save(str(images_folder / 'test' / 'image.png'))
    yield images_folder
    rmtree(str(images_folder))

def test_load_simple_classification_dataset(correct_images_folder):
    if False:
        return 10
    train_test_dataloaders = simple.classification_dataset_from_directory(root=str(correct_images_folder), object_type='DataLoader', image_extension='PNG')
    for data_loader in train_test_dataloaders:
        assert_that(data_loader, all_of(instance_of(DataLoader), has_length(1), has_property('dataset', instance_of(simple.SimpleClassificationDataset))))
        (img, label) = data_loader.dataset[0]
        assert_that(img, all_of(instance_of(np.ndarray), has_property('shape', equal_to((10, 10, 3)))))
        assert_that(label, instance_of(int))
        assert_that(data_loader.dataset.reverse_classes_map[label] == 'class1')

def test_load_simple_classification_dataset_only_train(correct_images_folder):
    if False:
        print('Hello World!')
    data_loader = simple.classification_dataset_from_directory(root=str(correct_images_folder.joinpath('train')), object_type='DataLoader', image_extension='PNG')
    assert_that(data_loader, all_of(instance_of(DataLoader), has_length(1), has_property('dataset', instance_of(simple.SimpleClassificationDataset))))
    (img, label) = data_loader.dataset[0]
    assert_that(img, all_of(instance_of(np.ndarray), has_property('shape', equal_to((10, 10, 3)))))
    assert_that(label, instance_of(int))
    assert_that(data_loader.dataset.reverse_classes_map[label] == 'class1')

def test_load_simple_classification_vision_data(correct_images_folder):
    if False:
        return 10
    (train_vision_data, test_vision_data) = simple.classification_dataset_from_directory(root=str(correct_images_folder), object_type='VisionData', image_extension='PNG')
    for vision_data in [train_vision_data, test_vision_data]:
        batches = list(vision_data)
        assert_that(len(batches) == 1)
        images = batches[0].get('images')
        labels = batches[0].get('labels')
        assert_that(len(images) == 1 and len(labels) == 1)
        assert_that(images[0], all_of(instance_of(np.ndarray), has_property('shape', equal_to((10, 10, 3)))))
        assert_that(vision_data.batch_loader.dataset.reverse_classes_map[labels[0]] == 'class1')

def test_load_simple_classification_dataset_from_broken_folder(incorrect_images_folder):
    if False:
        while True:
            i = 10
    assert_that(calling(simple.classification_dataset_from_directory).with_args(root=str(incorrect_images_folder), object_type='DataLoader', image_extension='PNG'), raises(ValueError))