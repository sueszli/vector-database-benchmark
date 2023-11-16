from hamcrest import assert_that, calling, close_to, has_length, raises
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.vision.checks import MeanAverageRecallReport
from tests.base.utils import equal_condition_result

def test_mnist_error(mnist_visiondata_test):
    if False:
        while True:
            i = 10
    check = MeanAverageRecallReport()
    assert_that(calling(check.run).with_args(mnist_visiondata_test), raises(DeepchecksNotSupportedError, 'Check is irrelevant for task of type TaskType.CLASSIFICATION'))

def test_coco(coco_visiondata_test):
    if False:
        while True:
            i = 10
    check = MeanAverageRecallReport().add_condition_test_average_recall_greater_than(0.1).add_condition_test_average_recall_greater_than(0.4)
    result = check.run(coco_visiondata_test)
    df = result.value
    assert_that(df, has_length(4))
    assert_that(df.loc['All', 'AR@1 (%)'], close_to(0.33, 0.001))
    assert_that(df.loc['All', 'AR@10 (%)'], close_to(0.423, 0.001))
    assert_that(df.loc['All', 'AR@100 (%)'], close_to(0.429, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AR@1 (%)'], close_to(0.104, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AR@10 (%)'], close_to(0.22, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AR@100 (%)'], close_to(0.22, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AR@1 (%)'], close_to(0.325, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AR@10 (%)'], close_to(0.417, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AR@100 (%)'], close_to(0.423, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AR@1 (%)'], close_to(0.481, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AR@10 (%)'], close_to(0.544, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AR@100 (%)'], close_to(0.549, 0.001))
    assert_that(result.conditions_results[0], equal_condition_result(details='Found lowest score of 0.1 for area Small (area < 32^2) and IoU AR@1 (%)', is_pass=True, name='Scores are greater than 0.1'))
    assert_that(result.conditions_results[1], equal_condition_result(is_pass=False, name='Scores are greater than 0.4', details='Found lowest score of 0.1 for area Small (area < 32^2) and IoU AR@1 (%)'))

def test_coco_area_param(coco_visiondata_test):
    if False:
        print('Hello World!')
    check = MeanAverageRecallReport(area_range=(40 ** 2, 100 ** 2))
    result = check.run(coco_visiondata_test)
    df = result.value
    assert_that(df, has_length(4))
    assert_that(df.loc['All', 'AR@1 (%)'], close_to(0.33, 0.001))
    assert_that(df.loc['All', 'AR@10 (%)'], close_to(0.423, 0.001))
    assert_that(df.loc['All', 'AR@100 (%)'], close_to(0.429, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AR@1 (%)'], close_to(0.101, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AR@10 (%)'], close_to(0.197, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AR@100 (%)'], close_to(0.204, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AR@1 (%)'], close_to(0.314, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AR@10 (%)'], close_to(0.428, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AR@100 (%)'], close_to(0.446, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AR@1 (%)'], close_to(0.482, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AR@10 (%)'], close_to(0.547, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AR@100 (%)'], close_to(0.551, 0.001))