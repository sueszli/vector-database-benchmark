"""Contains unit tests for the new_label_train_validation check"""
import pandas as pd
from hamcrest import assert_that, calling, equal_to, greater_than, has_items, has_length, raises
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.train_test_validation import NewLabelTrainTest
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result

def test_dataset_wrong_input():
    if False:
        while True:
            i = 10
    x = 'wrong_input'
    assert_that(calling(NewLabelTrainTest().run).with_args(x, x), raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))

def test_no_new_label():
    if False:
        i = 10
        return i + 15
    train_data = {'col1': [1, 2, 3]}
    test_data = {'col1': [1, 1, 2, 3]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    assert_that(result.value)
    assert_that(result.value['n_new_labels_samples'], equal_to(0))
    assert_that(result.value['n_samples'], equal_to(4))
    assert_that(result.value['new_labels'], equal_to([]))

def test_new_label():
    if False:
        i = 10
        return i + 15
    train_data = {'col1': [1, 2, 3]}
    test_data = {'col1': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    assert_that(result.value)
    assert_that(result.value['n_new_labels_samples'], equal_to(1))
    assert_that(result.value['n_samples'], equal_to(4))
    assert_that(result.value['new_labels'], equal_to([4]))
    assert_that(result.display, has_length(greater_than(0)))

def test_new_label_without_display():
    if False:
        return 10
    train_data = {'col1': [1, 2, 3]}
    test_data = {'col1': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset, with_display=False)
    assert_that(result.value)
    assert_that(result.value['n_new_labels_samples'], equal_to(1))
    assert_that(result.value['n_samples'], equal_to(4))
    assert_that(result.value['new_labels'], equal_to([4]))
    assert_that(result.display, has_length(0))

def test_missing_label():
    if False:
        print('Hello World!')
    train_data = {'col1': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    assert_that(result.value)
    assert_that(result.value['n_new_labels_samples'], equal_to(0))
    assert_that(result.value['n_samples'], equal_to(3))
    assert_that(result.value['new_labels'], equal_to([]))

def test_missing_new_label():
    if False:
        for i in range(10):
            print('nop')
    train_data = {'col1': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    assert_that(result)
    assert_that(result['n_new_labels_samples'], equal_to(1))
    assert_that(result['n_samples'], equal_to(4))
    assert_that(result['new_labels'], equal_to([5]))

def test_multiple_categories():
    if False:
        while True:
            i = 10
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    assert_that(result)
    assert_that(result['n_new_labels_samples'], equal_to(1))
    assert_that(result['n_samples'], equal_to(4))
    assert_that(result['new_labels'], equal_to([5]))

def test_new_label_reduce():
    if False:
        i = 10
        return i + 15
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    reduce_value = check.reduce_output(result)
    assert_that(reduce_value['Samples with New Labels'], equal_to(1))

def test_new_label_reduce_no_new_labels():
    if False:
        print('Hello World!')
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest()
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    reduce_value = check.reduce_output(result)
    assert_that(reduce_value['Samples with New Labels'], equal_to(0))

def test_condition_number_of_new_labels_pass():
    if False:
        print('Hello World!')
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(3)
    result = check.conditions_decision(check.run(train_dataset, test_dataset))
    assert_that(result, has_items(equal_condition_result(is_pass=True, details='Found 1 new labels in test data: [5]', name='Number of new label values is less or equal to 3')))

def test_condition_number_of_new_labels_fail():
    if False:
        print('Hello World!')
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(0)
    result = check.conditions_decision(check.run(train_dataset, test_dataset))
    assert_that(result, has_items(equal_condition_result(is_pass=False, details='Found 1 new labels in test data: [5]', name='Number of new label values is less or equal to 0')))

def test_condition_ratio_of_new_label_samples_pass():
    if False:
        i = 10
        return i + 15
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest().add_condition_new_label_ratio_less_or_equal(0.3)
    result = check.conditions_decision(check.run(train_dataset, test_dataset))
    assert_that(result, has_items(equal_condition_result(is_pass=True, details='Found 25% of labels in test data are new labels: [5]', name='Ratio of samples with new label is less or equal to 30%')))

def test_condition_ratio_of_new_label_samples_fail():
    if False:
        print('Hello World!')
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']), label='col1', label_type='multiclass')
    check = NewLabelTrainTest().add_condition_new_label_ratio_less_or_equal(0.1)
    result = check.conditions_decision(check.run(train_dataset, test_dataset))
    assert_that(result, has_items(equal_condition_result(is_pass=False, details='Found 25% of labels in test data are new labels: [5]', name='Ratio of samples with new label is less or equal to 10%')))