"""Test for the NLP TextEmbeddingsDrift check"""
from sys import platform
import numba
from hamcrest import assert_that, close_to
from deepchecks.nlp.checks import TextEmbeddingsDrift
from tests.base.utils import equal_condition_result

def test_tweet_emotion_no_drift(tweet_emotion_train_test_textdata_sampled):
    if False:
        i = 10
        return i + 15
    (train, _) = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift(with_display=False)
    result = check.run(train, train)
    assert_that(result.value['domain_classifier_drift_score'], close_to(0, 0.01))

def test_tweet_emotion(tweet_emotion_train_test_textdata_sampled):
    if False:
        while True:
            i = 10
    (train, test) = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift()
    result = check.run(train, test)
    assert_that(result.value['domain_classifier_drift_score'], close_to(0.2, 0.1))

def test_reduction_method(tweet_emotion_train_test_textdata_sampled):
    if False:
        for i in range(10):
            print('nop')
    (train, test) = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift(dimension_reduction_method='PCA')
    result = check.run(train, test)
    assert_that(result.value['domain_classifier_drift_score'], close_to(0.11, 0.01))
    check = TextEmbeddingsDrift(dimension_reduction_method='auto')
    result = check.run(train, test, with_display=False)
    assert_that(result.value['domain_classifier_drift_score'], close_to(0.11, 0.01))
    check = TextEmbeddingsDrift(dimension_reduction_method='none')
    result = check.run(train, test)
    assert_that(result.value['domain_classifier_drift_score'], close_to(0.18, 0.01))

def test_max_drift_score_condition_pass(tweet_emotion_train_test_textdata_sampled):
    if False:
        i = 10
        return i + 15
    (train, test) = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift().add_condition_overall_drift_value_less_than()
    result = check.run(train, test, with_display=False)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=True, details='Found drift value of: 0.12, corresponding to a domain classifier AUC of: 0.56', name='Drift value is less than 0.25'))

def test_max_drift_score_condition_fail(tweet_emotion_train_test_textdata_sampled):
    if False:
        while True:
            i = 10
    (train, test) = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift().add_condition_overall_drift_value_less_than(0.1)
    result = check.run(train, test, with_display=False)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=False, name='Drift value is less than 0.1', details='Found drift value of: 0.12, corresponding to a domain classifier AUC of: 0.56'))