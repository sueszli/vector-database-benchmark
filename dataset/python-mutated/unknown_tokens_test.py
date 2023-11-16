"""Test for the NLP UnknownTokens check."""
import pytest
from hamcrest import *
from transformers import AutoTokenizer, BertTokenizer
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.checks import UnknownTokens
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result

@pytest.fixture
def dataset_without_unknown_tokens():
    if False:
        print('Hello World!')
    return TextData(label=[0, 1, 2], task_type='text_classification', raw_text=['Explicit is better than implicit.', 'Simple is better than complex.', 'Complex is better than complicated.'])

@pytest.fixture
def dataset_with_unknown_tokens():
    if False:
        print('Hello World!')
    return TextData(label=[0, 1, 2], task_type='text_classification', raw_text=['Explicit is better than ˚implicit.', 'Simple is better than ∫complex.', 'Complex is better than complicated.'])

@pytest.fixture
def dataset_with_reoccurring_unknown_words():
    if False:
        return 10
    return TextData(label=[0, 1, 2, 0, 1, 2], task_type='text_classification', raw_text=['Explicit is better than ˚implicit.', 'Simple is better than ∫complex.', 'Complex is better than complicated.', 'Explicit is better than ˚implicit.', 'Simple is better than ∫complex∫.', 'Complex is better than complicated.'])

def test_without_unknown_tokens(dataset_without_unknown_tokens):
    if False:
        print('Hello World!')
    check = UnknownTokens().add_condition_ratio_of_unknown_words_less_or_equal()
    result = check.run(dataset=dataset_without_unknown_tokens)
    conditions_decisions = check.conditions_decision(result)
    assert_that(result.value, has_entries({'unknown_word_ratio': equal_to(0), 'unknown_word_details': instance_of(dict)}))
    assert_that(conditions_decisions[0], equal_condition_result(is_pass=True, details=f'Ratio was 0%', name='Ratio of unknown words is less than 0%'))
    assert_that(len(result.value['unknown_word_details']), equal_to(0))
    assert_that(result.display, equal_to([]))

def test_with_unknown_tokens(dataset_with_unknown_tokens):
    if False:
        while True:
            i = 10
    check = UnknownTokens().add_condition_ratio_of_unknown_words_less_or_equal()
    result = check.run(dataset=dataset_with_unknown_tokens)
    conditions_decisions = check.conditions_decision(result)
    assert_that(result.value, has_entries({'unknown_word_ratio': close_to(0.1111, 0.0001), 'unknown_word_details': instance_of(dict)}))
    assert_that(conditions_decisions[0], equal_condition_result(is_pass=False, details=f"Ratio was {format_percent(result.value['unknown_word_ratio'])}", name='Ratio of unknown words is less than 0%'))
    assert_that(len(result.value['unknown_word_details']), equal_to(2))
    assert_that(result.value['unknown_word_details'], has_entries({'˚implicit': has_entries({'ratio': close_to(0.0555, 0.0001), 'indexes': contains_exactly(0)}), '∫complex': has_entries({'ratio': close_to(0.0555, 0.0001), 'indexes': contains_exactly(1)})}))

def test_compare_fast_to_slow_tokenizer(dataset_with_unknown_tokens):
    if False:
        while True:
            i = 10
    check = UnknownTokens()
    check_slow = UnknownTokens(tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
    result = check.run(dataset=dataset_with_unknown_tokens)
    result_slow = check_slow.run(dataset=dataset_with_unknown_tokens)
    assert_that(result.value, equal_to(result_slow.value))

def test_group_singleton_words_true(dataset_with_reoccurring_unknown_words):
    if False:
        i = 10
        return i + 15
    check = UnknownTokens(group_singleton_words=True).add_condition_ratio_of_unknown_words_less_or_equal()
    result = check.run(dataset=dataset_with_reoccurring_unknown_words)
    conditions_decisions = check.conditions_decision(result)
    assert_that(result.value, has_entries({'unknown_word_ratio': close_to(0.1111, 0.0001), 'unknown_word_details': instance_of(dict)}))
    assert_that(conditions_decisions[0], equal_condition_result(is_pass=False, details=f"Ratio was {format_percent(result.value['unknown_word_ratio'])}", name='Ratio of unknown words is less than 0%'))
    display = result.display
    assert_that(display, has_length(2))
    pie_chart = display[0]
    assert_that(pie_chart['data'][0]['labels'], has_item('Other Unknown Words'))

def test_with_more_robust_tokenizer(dataset_with_unknown_tokens):
    if False:
        return 10
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    check = UnknownTokens(tokenizer=tokenizer)
    result = check.run(dataset=dataset_with_unknown_tokens)
    conditions_decisions = check.conditions_decision(result)
    assert_that(result.value, has_entries({'unknown_word_ratio': equal_to(0), 'unknown_word_details': instance_of(dict)}))
    assert_that(len(result.value['unknown_word_details']), equal_to(0))

def test_for_illegal_tokenizer(dataset_with_unknown_tokens):
    if False:
        for i in range(10):
            print('nop')
    tokenizer = 'a'
    assert_that(calling(UnknownTokens).with_args(tokenizer=tokenizer), raises(DeepchecksValueError, 'tokenizer must have a "tokenize" method'))