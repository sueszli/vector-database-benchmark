"""

.. _tabular__create_custom_check:

=======================
Creating a Custom Check
=======================

It is possible to extend deepchecks by implementing custom checks. This
enables you to have your own logic of metrics or validation, or even just
to display your own graph using deepchecks' suite.

* `Check Structure <#check-structure>`__
* `Write a Basic Check <#write-a-basic-check>`__
* `Check Display <#check-display>`__
* :ref:`tabular__custom_check_templates`


Check Structure
===============
Each check consists of 3 main parts:

* Return Value
* Display
* Conditions

This guide will demonstrate how to implement a Check with a return value and
display, for adding a condition see :ref:`configure_check_conditions`,
or have a look at the examples in :ref:`tabular__custom_check_templates` guide..

Write a Basic Check
===================
Let's implement a check for comparing the sizes of the test and the train datasets.

The first step is to create check class, which inherits from a base check class.
Each base check is differed by its run method signature, read more about all
`types <#base-checks-types>`__. In this case we will use ``TrainTestBaseCheck``,
which is used to compare between the test and the train datasets. After
creating the basic class with the run_logic function we will write our check
logic inside it.

*Good to know: the return value of a check can be any object, a number,
dictionary, string, etc...*

The Context Object
------------------
The logic of all tabular checks is executed inside the run_logic() function. The sole argument of the function is
the context object, which has the following optional members:

- **train**: the train dataset
- **test**: the test dataset
- **model**: the model

When writing your run_logic() function, you can access the train and test datasets using the context object.
For more examples of using the Context object for different types of base checks, see the 
:ref:`tabular__custom_check_templates` guide.

Check Example
-------------
"""
from deepchecks.core import CheckResult
from deepchecks.tabular import Context, Dataset, TrainTestCheck

class DatasetSizeComparison(TrainTestCheck):
    """Check which compares the sizes of train and test datasets."""

    def run_logic(self, context: Context) -> CheckResult:
        if False:
            i = 10
            return i + 15
        train_size = context.train.n_samples
        test_size = context.test.n_samples
        return_value = {'train_size': train_size, 'test_size': test_size}
        return CheckResult(return_value)
import pandas as pd
train_dataset = Dataset(pd.DataFrame(data={'x': [1, 2, 3, 4, 5, 6, 7, 8, 9]}), label=None)
test_dataset = Dataset(pd.DataFrame(data={'x': [1, 2, 3]}), label=None)
result = DatasetSizeComparison().run(train_dataset, test_dataset)
result
result.value
import matplotlib.pyplot as plt
from deepchecks.core import CheckResult
from deepchecks.tabular import Context, Dataset, TrainTestCheck

class DatasetSizeComparison(TrainTestCheck):
    """Check which compares the sizes of train and test datasets."""

    def run_logic(self, context: Context) -> CheckResult:
        if False:
            i = 10
            return i + 15
        train_size = context.train.n_samples
        test_size = context.test.n_samples
        sizes = {'Train': train_size, 'Test': test_size}
        sizes_df_for_display = pd.DataFrame(sizes, index=['Size'])

        def graph_display():
            if False:
                i = 10
                return i + 15
            plt.bar(sizes.keys(), sizes.values(), color='green')
            plt.xlabel('Dataset')
            plt.ylabel('Size')
            plt.title('Datasets Size Comparison')
        return CheckResult(sizes, display=[sizes_df_for_display, graph_display])
DatasetSizeComparison().run(train_dataset, test_dataset)