"""Test for the NLP SpecialCharacters check."""
import pytest
from hamcrest import *
from deepchecks.nlp.checks.data_integrity.special_characters import SpecialCharacters
from deepchecks.nlp.text_data import TextData
from tests.base.utils import equal_condition_result

@pytest.fixture
def clean_dataset():
    if False:
        while True:
            i = 10
    return TextData(raw_text=['Hello world', 'Do not worry be happy', 'Weather is fine'])

@pytest.fixture
def dataset_with_special_characters():
    if False:
        return 10
    return TextData(raw_text=['Hello world¶¶', 'Do not worry¸ be happy·', 'Weather is fine', 'Readability counts·', 'Errors should never pass silently·'])

def test_check_on_clean_dataset(clean_dataset):
    if False:
        i = 10
        return i + 15
    check = SpecialCharacters().add_condition_samples_ratio_w_special_characters_less_or_equal(0)
    result = check.run(dataset=clean_dataset)
    conditions_decision = check.conditions_decision(result)
    assert_that(result.value, has_entries({'samples_per_special_char': has_length(0), 'percent_of_samples_with_special_chars': equal_to(0), 'percent_special_chars_per_sample': has_length(3)}))
    assert_that(result.display, has_length(0))
    assert_that(conditions_decision[0], equal_condition_result(is_pass=True, details='Found 0 samples with special char ratio above threshold', name='Ratio of samples containing more than 20% special characters is below 0%'))

def test_check_on_dataset_with_emptt_sample():
    if False:
        print('Hello World!')
    data = TextData(raw_text=['', 'aa'])
    check = SpecialCharacters().add_condition_samples_ratio_w_special_characters_less_or_equal(0)
    result = check.run(dataset=data)
    assert_that(result.value, has_entries({'samples_per_special_char': has_length(0), 'percent_of_samples_with_special_chars': equal_to(0), 'percent_special_chars_per_sample': has_length(2)}))

def test_check_on_samples_with_special_characters(dataset_with_special_characters):
    if False:
        i = 10
        return i + 15
    check = SpecialCharacters().add_condition_samples_ratio_w_special_characters_less_or_equal(threshold_ratio_per_sample=0.1, max_ratio=0.15)
    result = check.run(dataset=dataset_with_special_characters)
    conditions_decision = check.conditions_decision(result)
    assert_that(result.value, has_entries({'samples_per_special_char': has_entries({'¶': [0], '·': [1, 3, 4], '¸': [1]}), 'percent_of_samples_with_special_chars': equal_to(0.8), 'percent_special_chars_per_sample': has_length(5)}))
    assert_that(result.display, has_length(3))
    assert_that(conditions_decision[0], equal_condition_result(is_pass=False, details='Found 1 samples with special char ratio above threshold', name='Ratio of samples containing more than 10% special characters is below 15%'))

def test_tweet_dataset(tweet_emotion_train_test_textdata_sampled):
    if False:
        i = 10
        return i + 15
    (_, text_data) = tweet_emotion_train_test_textdata_sampled
    check = SpecialCharacters().add_condition_samples_ratio_w_special_characters_less_or_equal()
    result = check.run(dataset=text_data)
    conditions_decision = check.conditions_decision(result)
    assert_that(result.value, has_entries({'samples_per_special_char': has_entries({'😍': [71, 614, 1813, 1901]}), 'percent_of_samples_with_special_chars': equal_to(0.168), 'percent_special_chars_per_sample': has_length(500)}))
    assert_that(result.display, has_length(3))
    assert_that(conditions_decision[0], equal_condition_result(is_pass=True, details='Found 1 samples with special char ratio above threshold', name='Ratio of samples containing more than 20% special characters is below 5%'))