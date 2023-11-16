"""
.. _tabular__feature_label_correlation_change:

Feature Label Correlation Change
********************************

This notebook provides an overview for using and understanding the "Feature Label Correlation Change" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Generate data <#generate-data>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check estimates for every feature its ability to predict the label by itself.
This check can help find:

* A potential leakage (between the label and a feature) in both datasets
  - e.g. due to incorrect sampling during data collection. This is a critical
  problem, that will likely stay hidden without this check (as it won't pop
  up when comparing model performance on train and test).
* A strong drift between the the feature-label relation in both datasets,
  possibly originating from a leakage in one of the datasets - e.g. a
  leakage that exists in the training data, but not necessarily in a
  "fresh" dataset, that may have been built differently.

The check is based on calculating the predictive power score (PPS) of each
feature. For more details you can read here `how the PPS is calculated 
<#how-is-the-predictive-power-score-pps-calculated>`__.

What is a problematic result?
-----------------------------
1. Features with a high predictive score - can indicate that there is a leakage
   between the label and the feature, meaning that the feature holds information
   that is somewhat based on the label to begin with.

   For example: a bank uses their loans database to create a model of whether
   a customer will be able to return a loan. One of the features they extract
   is "number of late payments". It is clear this feature will have a very
   strong prediction power on the customer's ability to return his loan,
   but this feature is based on data the bank knows only after the loan is
   given, so it won't be available during the time of the prediction, and is
   a type of leakage.
2. A high difference between the PPS scores of a certain feature in the train
   and in the test datasets - this is an indication for a drift between the
   relation of the feature and the label and a possible leakage in one of 
   the datasets.

   For example: a coffee shop chain trained a model to predict the number of
   coffee cups ordered in a store, and the model was trained on data from a
   specific state, and tested on data from all states. Running the Feature
   Label Correlation check on this split found that there was a high
   difference in the PPS score of the feature "time_in_day" - it had a
   much higher predictive power on the training data than on the test data.
   Investigating this topic led to detection of the problem - the time in
   day was saved in UTC time for all states, which made the feature much
   less indicative for the test data as it had data from several time
   zones (and much more coffee cups are ordered in during the morning/noon
   than during the evening/night time). This was fixed by changing the
   feature to be the time relative to the local time zone, thus fixing its
   predictive power and improving the model's overall performance.

.. _plot_tabular_feature_label_correlation_change__how_is_the_predictive_power_score_pps_calculated:

How is the Predictive Power Score (PPS) calculated?
===================================================
The features' predictive score results in a numeric score between 0 (feature
has no predictive power) and 1 (feature can fully predict the label alone).

The process of calculating the PPS is the following:
"""
from deepchecks.tabular.datasets.classification.phishing import load_data

def relate_column_to_label(dataset, column, label_power):
    if False:
        print('Hello World!')
    col_data = dataset.data[column]
    dataset.data[column] = col_data + dataset.data[dataset.label_name] * col_data.mean() * label_power
(train_dataset, test_dataset) = load_data()
relate_column_to_label(train_dataset, 'numDigits', 10)
relate_column_to_label(train_dataset, 'numLinks', 10)
relate_column_to_label(test_dataset, 'numDigits', 0.1)
from deepchecks.tabular.checks import FeatureLabelCorrelationChange
result = FeatureLabelCorrelationChange().run(train_dataset=train_dataset, test_dataset=test_dataset)
result
result.value
check = FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than().add_condition_feature_pps_in_train_less_than()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result.show(show_additional_outputs=False)