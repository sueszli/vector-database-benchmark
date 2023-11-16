"""
.. _quick_data_integrity:

Data Integrity Suite Quickstart
*********************************

The deepchecks integrity suite is relevant any time you have data that you wish to validate:
whether it's on a fresh batch of data, or right before splitting it or using it for training. 
Here we'll use the avocado prices dataset (:mod:`deepchecks.tabular.datasets.regression.avocado`),
to demonstrate how you can run the suite with only a few simple lines of code,
and see which kind of insights it can find.

.. code-block:: bash

    # Before we start, if you don't have deepchecks installed yet, run:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet

    # or install using pip from your python environment
"""
from deepchecks.tabular import datasets
data = datasets.regression.avocado.load_data(data_format='DataFrame', as_train_test=False)
import pandas as pd

def add_dirty_data(df):
    if False:
        return 10
    df.loc[df[df['type'] == 'organic'].sample(frac=0.18).index, 'type'] = 'Organic'
    df.loc[df[df['type'] == 'organic'].sample(frac=0.01).index, 'type'] = 'ORGANIC'
    df = pd.concat([df, df.sample(frac=0.156)], axis=0, ignore_index=True)
    df['Is Ripe'] = True
    return df
dirty_df = add_dirty_data(data)
from deepchecks.tabular import Dataset
ds = Dataset(dirty_df, cat_features=['type'], datetime_name='Date', label='AveragePrice')
from deepchecks.tabular.suites import data_integrity
integ_suite = data_integrity()
suite_result = integ_suite.run(ds)
suite_result.show()
from deepchecks.tabular.checks import IsSingleValue, DataDuplicates
IsSingleValue().run(ds)
single_value_with_condition = IsSingleValue().add_condition_not_single_value()
result = single_value_with_condition.run(ds)
result.show()
result.value
ds.data.drop('Is Ripe', axis=1, inplace=True)
result = single_value_with_condition.run(ds)
result.show()
dirty_df.drop_duplicates(inplace=True)
dirty_df.drop('Is Ripe', axis=1, inplace=True)
ds = Dataset(dirty_df, cat_features=['type'], datetime_name='Date', label='AveragePrice')
result = DataDuplicates().add_condition_ratio_less_or_equal(0).run(ds)
result.show()
integ_suite
integ_suite[3].clean_conditions()
res = integ_suite.run(ds)