import numpy as np
from hamcrest import assert_that, equal_to, instance_of, is_not
from deepchecks.vision import VisionData
from deepchecks.vision.datasets.classification.mnist_tensorflow import load_dataset, load_model

def test_deepchecks_dataset_load():
    if False:
        return 10
    dataset = load_dataset(train=True)
    assert_that(dataset, instance_of(VisionData))
    dataset = load_dataset(train=False)
    assert_that(dataset, instance_of(VisionData))

def test_regular_visiondata_with_shuffle():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(42)
    vision_data = load_dataset(n_samples=100, shuffle=False)
    batch = next(iter(vision_data))
    vision_data_again = load_dataset(n_samples=100, shuffle=False)
    batch_again = next(iter(vision_data_again))
    vision_data_shuffled = load_dataset(n_samples=100, shuffle=True)
    batch_shuffled = next(iter(vision_data_shuffled))
    vision_data_shuffled_again = load_dataset(n_samples=100, shuffle=True)
    batch_shuffled_again = next(iter(vision_data_shuffled_again))
    assert_that(batch['labels'][0], is_not(equal_to(batch_shuffled['labels'][0])))
    assert_that(batch['labels'][0], equal_to(batch_again['labels'][0]))
    assert_that(batch_shuffled_again['labels'][0], is_not(equal_to(batch_shuffled['labels'][0])))
    assert_that(batch['predictions'][0][0], is_not(equal_to(batch_shuffled['predictions'][0][0])))
    assert_that(batch['predictions'][0][0], equal_to(batch_again['predictions'][0][0]))
    assert_that(batch_shuffled_again['predictions'][0][0], is_not(equal_to(batch_shuffled['predictions'][0][0])))