"""

.. _configure_check_conditions:

Configure Check Conditions
**************************

The following guide includes different options for configuring a check's condition(s):

* `Add Condition <#add-condition>`__
* `Remove / Edit a Condition <#remove-edit-a-condition>`__
* `Add a Custom Condition <#add-a-custom-condition>`__
* `Set Custom Condition Category <#set-custom-condition-category>`__

Add Condition
=============
In order to add a condition to an existing check, we can use any of the pre-defined
conditions for that check. The naming convention for the methods that add the
condition is ``add_condition_...``.

If you want to create and add your custom condition logic for parsing the check's
result value, see `Add a Custom Condition <#add-a-custom-condition>`__.
"""
from deepchecks.tabular.checks import DatasetsSizeComparison
check = DatasetsSizeComparison().add_condition_test_size_greater_or_equal(1000)
check
import pandas as pd
from deepchecks.tabular import Dataset
train_dataset = Dataset(pd.DataFrame(data={'x': [1, 2, 3, 4, 5, 6, 7, 8, 9]}))
test_dataset = Dataset(pd.DataFrame(data={'x': [1, 2, 3]}))
condition_results = check.conditions_decision(check.run(train_dataset, test_dataset))
condition_results
from deepchecks.tabular.suites import train_test_validation
suite = train_test_validation()
suite
check = suite[8]
check.remove_condition(0)
suite
check.add_condition_feature_pps_difference_less_than(0.01)
suite
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DatasetsSizeComparison
train_dataset = Dataset(pd.DataFrame(data={'x': [1, 2, 3, 4, 5, 6, 7, 8, 9]}))
test_dataset = Dataset(pd.DataFrame(data={'x': [1, 2, 3]}))
result = DatasetsSizeComparison().run(train_dataset, test_dataset)
result.value
from deepchecks.core import ConditionResult
low_threshold = 0.4
high_threshold = 0.6

def custom_condition(value: dict, low=low_threshold, high=high_threshold):
    if False:
        i = 10
        return i + 15
    ratio = value['Test'] / value['Train']
    if low <= ratio <= high:
        return ConditionResult(ConditionCategory.PASS)
    else:
        return ConditionResult(ConditionCategory.FAIL, f'Test-Train ratio is {ratio:.2}')
condition_name = f'Test-Train ratio is between {low_threshold} to {high_threshold}'
check = DatasetsSizeComparison().add_condition(condition_name, custom_condition)
from deepchecks.tabular import Suite
suite = Suite('Suite for Condition', check)
suite.run(train_dataset, test_dataset)
from deepchecks.core import ConditionCategory, ConditionResult
low_threshold = 0.3
high_threshold = 0.7

def custom_condition(value: dict):
    if False:
        return 10
    ratio = value['Test'] / value['Train']
    if low_threshold <= ratio <= high_threshold:
        return ConditionResult(ConditionCategory.PASS)
    elif ratio < low_threshold:
        return ConditionResult(ConditionCategory.FAIL, f'Test-Train ratio is {ratio:.2}', ConditionCategory.FAIL)
    else:
        return ConditionResult(ConditionCategory.FAIL, f'Test-Train ratio is {ratio:.2}', ConditionCategory.WARN)