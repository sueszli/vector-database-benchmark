import tensorflow as tf
from hamcrest import assert_that, calling, equal_to, instance_of, is_not, raises
from deepchecks.vision import VisionData
from deepchecks.vision.datasets.detection.coco_tensorflow import load_dataset
from deepchecks.vision.datasets.detection.coco_utils import COCO_DIR
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

def test_load_dataset_test():
    if False:
        while True:
            i = 10
    loader = load_dataset(train=True, object_type='Dataset')
    assert_that(loader, instance_of(tf.data.Dataset))
    assert_that((COCO_DIR / 'coco128' / 'images').exists())
    assert_that((COCO_DIR / 'coco128' / 'labels').exists())

def test_deepchecks_dataset_load():
    if False:
        i = 10
        return i + 15
    vision_data = load_dataset(train=True, object_type='VisionData')
    assert_that(vision_data, instance_of(VisionData))
    assert_that(vision_data.num_classes, equal_to(80))
    assert_that(vision_data.number_of_images_cached, equal_to(0))
    loader = vision_data._batch_loader
    assert_that(loader, instance_of(tf.data.Dataset))
    assert_that((COCO_DIR / 'coco128' / 'images').exists())
    assert_that((COCO_DIR / 'coco128' / 'labels').exists())

def test_load_dataset_func_with_unknown_object_type_parameter():
    if False:
        print('Hello World!')
    assert_that(calling(load_dataset).with_args(object_type='<unknonw>'), raises(TypeError))

def test_train_test_split():
    if False:
        i = 10
        return i + 15
    train = load_dataset(train=True, object_type='VisionData')
    test = load_dataset(train=False, object_type='VisionData')
    for batch in train:
        batch = BatchWrapper(batch, train.task_type, train.number_of_images_cached)
        train.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
    for batch in test:
        batch = BatchWrapper(batch, test.task_type, test.number_of_images_cached)
        test.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)
    assert_that(train.number_of_images_cached + test.number_of_images_cached, equal_to(128))
    assert_that(train.get_observed_classes(), is_not(equal_to(test.get_observed_classes())))