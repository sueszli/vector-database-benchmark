import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from ludwig.constants import LOGITS, PREDICTIONS, PROBABILITIES
from ludwig.utils.llm_utils import add_left_padding, create_attention_mask, FALLBACK_CONTEXT_LEN, find_last_matching_index, generate_merged_ids, get_context_len, get_realigned_target_and_prediction_tensors_for_inference, has_padding_token, pad_target_tensor_for_fine_tuning, remove_left_padding, set_pad_token
pytestmark = [pytest.mark.llm]
TEST_MODEL_NAME = 'hf-internal-testing/tiny-random-OPTForCausalLM'

@pytest.fixture
def tokenizer():
    if False:
        for i in range(10):
            print('nop')
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    set_pad_token(tokenizer)
    return tokenizer

@pytest.fixture
def input_ids():
    if False:
        return 10
    return torch.tensor([[3, 4, 5], [6, 7, 8]])

@pytest.fixture
def target_ids():
    if False:
        i = 10
        return i + 15
    return torch.tensor([[9, 10, 11], [12, 13, 14]])

def test_set_pad_token_doesnt_exist():
    if False:
        i = 10
        return i + 15
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
    assert tokenizer.pad_token_id is None
    set_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 50256

def test_set_pad_token_already_exists():
    if False:
        while True:
            i = 10
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, use_fast=False)
    assert tokenizer.pad_token_id == 1
    set_pad_token(tokenizer)
    assert tokenizer.pad_token_id == 1

class TestSetContextLen:

    def test_max_sequence_length(self):
        if False:
            for i in range(10):
                print('nop')
        config = AutoConfig.from_pretrained('huggyllama/llama-7b')
        assert get_context_len(config) == config.max_sequence_length

    def test_max_position_embeddings(self):
        if False:
            return 10
        config = AutoConfig.from_pretrained('huggyllama/llama-7b')
        del config.max_sequence_length
        assert get_context_len(config) == config.max_position_embeddings

    def test_n_positions(self):
        if False:
            print('Hello World!')
        config = AutoConfig.from_pretrained('hf-internal-testing/tiny-random-GPTJForCausalLM')
        assert get_context_len(config) == config.n_positions

    def test_default_value(self):
        if False:
            for i in range(10):
                print('nop')
        config = AutoConfig.from_pretrained('hf-internal-testing/tiny-random-GPTJForCausalLM')
        del config.n_positions
        assert get_context_len(config) == FALLBACK_CONTEXT_LEN

def test_has_padding_token_with_padding_tokens(tokenizer):
    if False:
        print('Hello World!')
    input_sentence = 'This is an example sentence.'
    input_ids = tokenizer([input_sentence])
    input_ids['input_ids'] = torch.tensor(input_ids['input_ids'])
    padded_input_ids = torch.nn.functional.pad(input_ids['input_ids'], (10 - len(input_ids['input_ids']), 1), value=1)
    assert has_padding_token(padded_input_ids, tokenizer)

def test_has_padding_token_without_padding_tokens(tokenizer):
    if False:
        print('Hello World!')
    input_sentence = 'This is an example sentence.'
    input_ids = tokenizer([input_sentence])
    input_ids['input_ids'] = torch.tensor(input_ids['input_ids'])
    assert not has_padding_token(input_ids['input_ids'], tokenizer)

@pytest.mark.parametrize('input_ids, expected', [(torch.tensor([5]), torch.tensor([5])), (torch.tensor([5, 3]), torch.tensor([5, 3])), (torch.tensor([1, 5, 5, 3]), torch.tensor([5, 5, 3])), (torch.tensor([2, 5, 5, 3]), torch.tensor([2, 5, 5, 3])), (torch.tensor([1, 2, 5, 5, 3]), torch.tensor([2, 5, 5, 3]))])
def test_remove_left_padding(input_ids, expected, tokenizer):
    if False:
        while True:
            i = 10
    assert torch.equal(remove_left_padding(input_ids, tokenizer).squeeze(0), expected)

@pytest.mark.parametrize('input_ids, max_length, pad_value, expected', [(torch.tensor([1, 2, 3]), 3, 0, torch.tensor([1, 2, 3])), (torch.tensor([1, 2, 3]), 5, 0, torch.tensor([0, 0, 1, 2, 3])), (torch.tensor([4, 5, 6, 7]), 6, 2, torch.tensor([2, 2, 4, 5, 6, 7])), (torch.tensor([8, 9]), 3, 1, torch.tensor([1, 8, 9]))])
def test_add_left_padding(input_ids, max_length, pad_value, expected):
    if False:
        return 10
    padded = add_left_padding(input_ids, max_length, pad_value).squeeze(0)
    assert torch.equal(padded, expected)

def test_create_attention_mask_last_token_padding(tokenizer):
    if False:
        while True:
            i = 10
    input_ids = torch.tensor([3, 4, tokenizer.pad_token_id])
    attention_mask = create_attention_mask(input_ids, tokenizer)
    assert attention_mask[-1] == 1

@pytest.mark.parametrize('input_ids, expected_output', [(torch.tensor([3, 4, 5]), torch.tensor([1, 1, 1])), (torch.tensor([1, 1, 4, 6, 8]), torch.tensor([0, 0, 1, 1, 1])), (torch.tensor([1, 1, 1]), torch.tensor([0, 0, 1]))])
def test_create_attention_mask(input_ids, expected_output, tokenizer):
    if False:
        for i in range(10):
            print('nop')
    attention_mask = create_attention_mask(input_ids, tokenizer)
    assert torch.equal(attention_mask, expected_output)

@pytest.mark.parametrize('tensor_a, tensor_b, expected_index', [(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), torch.tensor([6, 7, 8]), 5), (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), torch.tensor([9, 10]), -1), (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), torch.tensor([4, 5, 6]), -1)])
def test_find_last_matching_index(tensor_a, tensor_b, expected_index):
    if False:
        return 10
    last_matching_index = find_last_matching_index(tensor_a, tensor_b)
    assert last_matching_index == expected_index

def test_generate_merged_ids_with_target(tokenizer, input_ids, target_ids):
    if False:
        i = 10
        return i + 15
    (merged_ids, attention_masks) = generate_merged_ids(input_ids, target_ids, tokenizer)
    assert torch.equal(merged_ids, torch.tensor([[3, 4, 5, 9, 10, 11, 1], [6, 7, 8, 12, 13, 14, 1]]))
    assert merged_ids.shape == (2, 7)
    assert attention_masks.shape == (2, 7)

def test_generate_merged_ids_with_max_sequence_length(tokenizer, input_ids, target_ids):
    if False:
        while True:
            i = 10
    max_sequence_length = 5
    (merged_ids, attention_masks) = generate_merged_ids(input_ids, target_ids, tokenizer, max_sequence_length)
    assert merged_ids.shape == (2, 5)
    assert attention_masks.shape == (2, 5)

def test_generate_merged_ids_padding_removal(tokenizer, input_ids, target_ids):
    if False:
        return 10
    padding_tokens = torch.tensor([tokenizer.pad_token_id, tokenizer.pad_token_id])
    input_ids_with_padding = torch.cat((padding_tokens.unsqueeze(0).expand(input_ids.size(0), -1), input_ids), dim=1)
    target_ids_with_padding = torch.cat((padding_tokens.unsqueeze(0).expand(target_ids.size(0), -1), target_ids), dim=1)
    (merged_ids, attention_masks) = generate_merged_ids(input_ids_with_padding, target_ids_with_padding, tokenizer)
    assert merged_ids.shape == (2, 7)
    assert attention_masks.shape == (2, 7)
    assert torch.equal(merged_ids[0][:3], input_ids[0])
    assert torch.equal(merged_ids[0][3:-1], target_ids[0])
    assert torch.equal(merged_ids[0][-1], torch.tensor(tokenizer.pad_token_id))
    assert torch.all(attention_masks == 1)

def test_generate_merged_ids_returns_tensor(tokenizer, input_ids, target_ids):
    if False:
        i = 10
        return i + 15
    (merged_ids, attention_masks) = generate_merged_ids(input_ids, target_ids, tokenizer)
    assert isinstance(merged_ids, torch.Tensor)
    assert isinstance(attention_masks, torch.Tensor)

def test_pad_target_tensor_for_fine_tuning():
    if False:
        for i in range(10):
            print('nop')
    of_name = 'out_1'
    prediction = {of_name: {PREDICTIONS: torch.tensor([[764, 764, 764, 764, 764, 764, 764, 578, 619, 841, 182, 905, 483, 764]])}}
    model_input = torch.tensor([[0, 0, 24, 52, 654, 529, 221, 78, 79, 504, 76, 397, 84, 0]])
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {of_name: torch.tensor([[-100, -100, -100, -100, -100, -100, -100, 78, 79, 504, 76, 397, 84, 0]])}
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])
    model_input = torch.tensor([[13, 24, 395, 13, 46, 57, 52, 41, 45, 37, 51, 14, 380, 435]])
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {of_name: torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]])}
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])
    model_input = torch.tensor([[0, 0, 24, 52, 654, 529, 221, 78, 79, 504, 76, 78, 79, 504]])
    target = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    expected_target = {of_name: torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 78, 79, 504]])}
    updated_targets = pad_target_tensor_for_fine_tuning(target, prediction, model_input, of_name)
    assert torch.equal(expected_target[of_name], updated_targets[of_name])

def test_get_realigned_target_and_prediction_tensors_for_inference(tokenizer):
    if False:
        print('Hello World!')
    of_name = 'out_1'
    vocab_size = 8
    targets = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    predictions = {of_name: {PREDICTIONS: torch.tensor([[78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64), PROBABILITIES: torch.randn(1, 7, vocab_size).to(torch.float32), LOGITS: torch.randn(1, 7, vocab_size).to(torch.float32)}}
    (updated_targets, updated_predictions) = get_realigned_target_and_prediction_tensors_for_inference(targets, predictions, of_name, tokenizer)
    assert targets == updated_targets
    assert predictions == updated_predictions
    assert predictions[of_name][PREDICTIONS].shape[1] == targets[of_name].shape[1]
    assert predictions[of_name][PROBABILITIES].shape[1] == targets[of_name].shape[1]
    assert predictions[of_name][LOGITS].shape[1] == targets[of_name].shape[1]
    targets = {of_name: torch.tensor([[78, 79, 504, 76, 397, 84, 0]])}
    predictions = {of_name: {PREDICTIONS: torch.tensor([[98, 47, 78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64), PROBABILITIES: torch.randn(1, 9, vocab_size).to(torch.float32), LOGITS: torch.randn(1, 9, vocab_size).to(torch.float32)}}
    (updated_targets, updated_predictions) = get_realigned_target_and_prediction_tensors_for_inference(targets, predictions, of_name, tokenizer)
    for key in updated_predictions.keys():
        assert torch.equal(updated_predictions[key][PREDICTIONS], predictions[key][PREDICTIONS])
        assert torch.equal(updated_predictions[key][PROBABILITIES], predictions[key][PROBABILITIES])
        assert torch.equal(updated_predictions[key][LOGITS], predictions[key][LOGITS])
    assert torch.equal(updated_targets[of_name], torch.tensor([[78, 79, 504, 76, 397, 84, 0, 1, 1]]))
    targets = {of_name: torch.tensor([[98, 47, 78, 79, 504, 76, 397, 84, 0]])}
    predictions = {of_name: {PREDICTIONS: torch.tensor([[78, 79, 504, 76, 397, 84, 0]], dtype=torch.int64), PROBABILITIES: torch.randn(1, 7, vocab_size).to(torch.float32), LOGITS: torch.randn(1, 7, vocab_size).to(torch.float32)}}
    (updated_targets, updated_predictions) = get_realigned_target_and_prediction_tensors_for_inference(targets, predictions, of_name, tokenizer)
    assert torch.equal(updated_targets[of_name], targets[of_name])
    assert torch.equal(updated_predictions[of_name][PREDICTIONS], torch.tensor([[78, 79, 504, 76, 397, 84, 0, 1, 1]]))
    assert updated_predictions[of_name][PROBABILITIES].shape[1] == targets[of_name].shape[1]
    assert updated_predictions[of_name][LOGITS].shape[1] == targets[of_name].shape[1]
    assert torch.equal(updated_predictions[of_name][PROBABILITIES][0][-1], torch.zeros(vocab_size))
    assert torch.equal(updated_predictions[of_name][PROBABILITIES][0][-2], torch.zeros(vocab_size))
    assert not torch.equal(updated_predictions[of_name][PROBABILITIES][0][-3], torch.zeros(vocab_size))
    assert torch.equal(updated_predictions[of_name][LOGITS][0][-1], torch.zeros(vocab_size))
    assert torch.equal(updated_predictions[of_name][LOGITS][0][-2], torch.zeros(vocab_size))
    assert not torch.equal(updated_predictions[of_name][LOGITS][0][-3], torch.zeros(vocab_size))