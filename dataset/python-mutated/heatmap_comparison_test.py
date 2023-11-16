"""Test functions of the heatmap comparison check."""
from hamcrest import assert_that, close_to, greater_than, has_length
from deepchecks.vision.checks import HeatmapComparison

def test_object_detection(coco_visiondata_train, coco_visiondata_test):
    if False:
        return 10
    check = HeatmapComparison()
    result = check.run(coco_visiondata_train, coco_visiondata_test)
    brightness_diff = result.value['diff']
    assert_that(brightness_diff.mean(), close_to(10.461, 0.001))
    assert_that(brightness_diff.max(), close_to(44, 0.001))
    bbox_diff = result.value['diff_bbox']
    assert_that(bbox_diff.mean(), close_to(5.593, 0.001))
    assert_that(bbox_diff.max(), close_to(23, 0.001))

def test_classification(mnist_visiondata_train, mnist_visiondata_test):
    if False:
        i = 10
        return i + 15
    check = HeatmapComparison(n_samples=None)
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    brightness_diff = result.value['diff']
    assert_that(brightness_diff.mean(), close_to(6.834, 0.001))
    assert_that(brightness_diff.max(), close_to(42, 0.001))
    assert_that(result.display, has_length(greater_than(0)))

def test_classification_without_display(mnist_visiondata_train, mnist_visiondata_test):
    if False:
        return 10
    check = HeatmapComparison()
    result = check.run(mnist_visiondata_train, mnist_visiondata_test, with_display=False)
    brightness_diff = result.value['diff']
    assert_that(brightness_diff.mean(), close_to(6.834, 0.001))
    assert_that(brightness_diff.max(), close_to(42, 0.001))
    assert_that(result.display, has_length(0))

def test_custom_task(mnist_train_custom_task, mnist_test_custom_task):
    if False:
        print('Hello World!')
    check = HeatmapComparison(n_samples=None)
    result = check.run(mnist_train_custom_task, mnist_test_custom_task)
    brightness_diff = result.value['diff']
    assert_that(brightness_diff.mean(), close_to(1.095, 0.001))
    assert_that(brightness_diff.max(), close_to(9, 0.001))

def test_dataset_name(mnist_visiondata_train, mnist_visiondata_test):
    if False:
        print('Hello World!')
    mnist_visiondata_train.name = 'Ref'
    mnist_visiondata_test.name = 'Win'
    result = HeatmapComparison().run(mnist_visiondata_train, mnist_visiondata_test)
    assert_that(result.display[0].layout.annotations[0].text, 'Ref')
    assert_that(result.display[0].layout.annotations[1].text, 'Win')