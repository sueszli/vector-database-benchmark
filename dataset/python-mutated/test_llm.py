import os
import pathlib
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import pytest
import torch
import ludwig.error as ludwig_error
from ludwig.api import LudwigModel
from ludwig.constants import ADAPTER, BACKEND, BASE_MODEL, BATCH_SIZE, EPOCHS, GENERATION, INPUT_FEATURES, MERGE_ADAPTER_INTO_BASE_MODEL, MODEL_LLM, MODEL_TYPE, OUTPUT_FEATURES, POSTPROCESSOR, PREPROCESSING, PRETRAINED_ADAPTER_WEIGHTS, PROGRESSBAR, PROMPT, QUANTIZATION, TRAINER, TYPE
from ludwig.models.llm import LLM
from ludwig.schema.model_types.base import ModelConfig
from ludwig.utils.fs_utils import list_file_names_in_directory
from ludwig.utils.types import DataFrame
from tests.integration_tests.utils import category_feature, generate_data, text_feature
pytestmark = pytest.mark.llm
LOCAL_BACKEND = {'type': 'local'}
TEST_MODEL_NAME = 'hf-internal-testing/tiny-random-GPTJForCausalLM'
MAX_NEW_TOKENS_TEST_DEFAULT = 5
RAY_BACKEND = {'type': 'ray', 'processor': {'parallelism': 1}, 'trainer': {'use_gpu': False, 'num_workers': 2, 'resources_per_worker': {'CPU': 1, 'GPU': 0}}}

def get_num_non_empty_tokens(iterable):
    if False:
        return 10
    'Returns the number of non-empty tokens.'
    return len(list(filter(bool, iterable)))

@pytest.fixture(scope='module')
def local_backend():
    if False:
        print('Hello World!')
    return LOCAL_BACKEND

@pytest.fixture(scope='module')
def ray_backend():
    if False:
        while True:
            i = 10
    return RAY_BACKEND

def get_dataset():
    if False:
        while True:
            i = 10
    data = [{'review': 'I loved this movie!', 'output': 'positive'}, {'review': 'The food was okay, but the service was terrible.', 'output': 'negative'}, {'review': "I can't believe how rude the staff was.", 'output': 'negative'}, {'review': 'This book was a real page-turner.', 'output': 'positive'}, {'review': 'The hotel room was dirty and smelled bad.', 'output': 'negative'}, {'review': 'I had a great experience at this restaurant.', 'output': 'positive'}, {'review': 'The concert was amazing!', 'output': 'positive'}, {'review': 'The traffic was terrible on my way to work this morning.', 'output': 'negative'}, {'review': 'The customer service was excellent.', 'output': 'positive'}, {'review': 'I was disappointed with the quality of the product.', 'output': 'negative'}]
    df = pd.DataFrame(data)
    return df

def get_generation_config():
    if False:
        while True:
            i = 10
    return {'temperature': 0.1, 'top_p': 0.75, 'top_k': 40, 'num_beams': 4, 'max_new_tokens': MAX_NEW_TOKENS_TEST_DEFAULT}

def convert_preds(preds: DataFrame):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(preds, pd.DataFrame):
        return preds.to_dict()
    return preds.compute().to_dict()

@pytest.mark.llm
@pytest.mark.parametrize('backend', [pytest.param(LOCAL_BACKEND, id='local'), pytest.param(RAY_BACKEND, id='ray')])
def test_llm_text_to_text(tmpdir, backend, ray_cluster_4cpu):
    if False:
        while True:
            i = 10
    'Test that the LLM model can train and predict with text inputs and text outputs.'
    input_features = [{'name': 'Question', 'type': 'text', 'encoder': {'type': 'passthrough'}}]
    output_features = [text_feature(output_feature=True, name='Answer', decoder={'type': 'text_extractor'})]
    csv_filename = os.path.join(tmpdir, 'training.csv')
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, GENERATION: get_generation_config(), INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, BACKEND: backend}
    model = LudwigModel(config)
    model.train(dataset=dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)
    (preds, _) = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split='test')
    preds = convert_preds(preds)
    assert 'Answer_predictions' in preds
    assert 'Answer_probabilities' in preds
    assert 'Answer_probability' in preds
    assert 'Answer_response' in preds
    assert preds['Answer_predictions']
    assert preds['Answer_probabilities']
    assert preds['Answer_probability']
    assert preds['Answer_response']
    assert get_num_non_empty_tokens(preds['Answer_predictions'][0]) <= MAX_NEW_TOKENS_TEST_DEFAULT
    original_max_new_tokens = model.model.generation.max_new_tokens
    (preds, _) = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split='test', generation_config={'min_new_tokens': 2, 'max_new_tokens': 3})
    preds = convert_preds(preds)
    print(preds['Answer_predictions'][0])
    num_non_empty_tokens = get_num_non_empty_tokens(preds['Answer_predictions'][0])
    assert 2 <= num_non_empty_tokens <= 3
    assert model.model.generation.max_new_tokens == original_max_new_tokens

@pytest.mark.llm
@pytest.mark.parametrize('backend', [pytest.param(LOCAL_BACKEND, id='local'), pytest.param(RAY_BACKEND, id='ray')])
def test_llm_zero_shot_classification(tmpdir, backend, ray_cluster_4cpu):
    if False:
        i = 10
        return i + 15
    input_features = [{'name': 'review', 'type': 'text'}]
    output_features = [category_feature(name='output', preprocessing={'fallback_label': 'neutral'}, decoder={'type': 'category_extractor', 'match': {'positive': {'type': 'contains', 'value': 'positive'}, 'neutral': {'type': 'regex', 'value': '\\bneutral\\b'}, 'negative': {'type': 'contains', 'value': 'negative'}}})]
    df = get_dataset()
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, GENERATION: get_generation_config(), PROMPT: {'task': 'This is a review of a restaurant. Classify the sentiment.'}, INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, BACKEND: backend}
    model = LudwigModel(config)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=True)
    prediction_df = pd.DataFrame([{'review': 'The food was amazing!', 'output': 'positive'}, {'review': 'The service was terrible.', 'output': 'negative'}, {'review': 'The food was okay.', 'output': 'neutral'}])
    (preds, _) = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)
    assert preds

@pytest.mark.llm
@pytest.mark.parametrize('backend', [pytest.param(LOCAL_BACKEND, id='local'), pytest.param(RAY_BACKEND, id='ray')])
def test_llm_few_shot_classification(tmpdir, backend, csv_filename, ray_cluster_4cpu):
    if False:
        while True:
            i = 10
    input_features = [text_feature(output_feature=False, name='body', encoder={'type': 'passthrough'})]
    output_features = [category_feature(output_feature=True, name='output', preprocessing={'fallback_label': '3'}, decoder={'type': 'category_extractor', 'match': {'1': {'type': 'contains', 'value': '1'}, '2': {'type': 'contains', 'value': '2'}, '3': {'type': 'contains', 'value': '3'}, '4': {'type': 'contains', 'value': '4'}, '5': {'type': 'contains', 'value': '5'}}})]
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, GENERATION: get_generation_config(), PROMPT: {'retrieval': {'type': 'random', 'k': 3}, 'task': 'Given the sample input, complete this sentence by replacing XXXX: The review rating is XXXX. Choose one value in this list: [1, 2, 3, 4, 5].'}, INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, PREPROCESSING: {'split': {TYPE: 'fixed'}}, BACKEND: {**backend, 'cache_dir': str(tmpdir)}}
    dataset_path = generate_data(input_features, output_features, filename=csv_filename, num_examples=25, nan_percent=0.1, with_split=True)
    df = pd.read_csv(dataset_path)
    df['output'] = np.random.choice([1, 2, 3, 4, 5], size=len(df)).astype(str)
    df.to_csv(dataset_path, index=False)
    model = LudwigModel(config)
    model.train(dataset=dataset_path, output_directory=str(tmpdir), skip_save_processed_input=True)
    (preds, _) = model.predict(dataset=dataset_path)
    preds = convert_preds(preds)
    assert preds

def _prepare_finetuning_test(csv_filename: str, finetune_strategy: str, backend: Dict, adapter_args: Dict) -> Tuple[Dict, str]:
    if False:
        print('Hello World!')
    input_features = [text_feature(name='input', encoder={'type': 'passthrough'})]
    output_features = [text_feature(name='output')]
    train_df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)
    prediction_df = pd.DataFrame([{'input': 'The food was amazing!', 'output': 'positive'}, {'input': 'The service was terrible.', 'output': 'negative'}, {'input': 'The food was okay.', 'output': 'neutral'}])
    model_name = TEST_MODEL_NAME
    if finetune_strategy == 'adalora':
        model_name = 'hf-internal-testing/tiny-random-BartModel'
    elif finetune_strategy == 'adaption_prompt':
        model_name = 'yujiepan/llama-2-tiny-random'
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: model_name, INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 2}, BACKEND: backend}
    if finetune_strategy is not None:
        config[ADAPTER] = {TYPE: finetune_strategy, **adapter_args}
    return (train_df, prediction_df, config)

def _finetune_strategy_requires_cuda(finetune_strategy_name: str, quantization_args: Union[dict, None]) -> bool:
    if False:
        i = 10
        return i + 15
    'This method returns whether a given finetine_strategy requires CUDA.\n\n    For all finetune strategies, except "qlora", the decision is based just on the name of the finetine_strategy; in the\n    case of qlora, if the quantization dictionary is non-empty (i.e., contains quantization specifications), then the\n    original finetine_strategy name of "lora" is interpreted as "qlora" and used in the lookup, based on the list of\n    finetine strategies requiring CUDA.\n    '
    cuda_only_finetune_strategy_names: List[str] = ['prompt_tuning', 'prefix_tuning', 'p_tuning', 'qlora']
    if finetune_strategy_name == 'lora' and quantization_args:
        finetune_strategy_name = 'qlora'
    return finetune_strategy_name in cuda_only_finetune_strategy_names

def _verify_lm_lora_finetuning_layers(attention_layer: torch.nn.Module, merge_adapter_into_base_model: bool, model_weights_directory: str, expected_lora_in_features: int, expected_lora_out_features: int, expected_file_names: List[str]) -> None:
    if False:
        return 10
    'This method verifies that LoRA finetuning layers have correct types and shapes, depending on whether the\n    optional "model.merge_and_unload()" method (based on the "merge_adapter_into_base_model" directive) was\n    executed.\n\n    If merge_adapter_into_base_model is True, then both LoRA projection layers, V and Q, in the attention layer must\n    contain square weight matrices (with the dimensions expected_lora_in_features by expected_lora_in_features).\n    However, if merge_adapter_into_base_model is False, then the LoRA part of the attention layer must include Lora_A\n    and Lora_B children layers for each of V and Q projections, such that the product of V and Q matrices is a square\n    matrix (with the dimensions expected_lora_in_features by expected_lora_in_features) for both V and Q projections.\n    '
    file_names: List[str] = list_file_names_in_directory(directory_name=model_weights_directory)
    assert set(file_names) == set(expected_file_names)
    assert isinstance(attention_layer.v_proj, torch.nn.Linear)
    assert isinstance(attention_layer.q_proj, torch.nn.Linear)
    if merge_adapter_into_base_model:
        assert (attention_layer.v_proj.in_features, attention_layer.v_proj.out_features) == (expected_lora_in_features, expected_lora_out_features)
        assert (attention_layer.q_proj.in_features, attention_layer.q_proj.out_features) == (expected_lora_in_features, expected_lora_out_features)
        assert not list(attention_layer.v_proj.children())
        assert not list(attention_layer.q_proj.children())
    else:
        v_proj_named_children: dict[str, torch.nn.Modeule] = dict(attention_layer.v_proj.named_children())
        assert isinstance(v_proj_named_children['lora_A']['default'], torch.nn.Linear)
        assert (v_proj_named_children['lora_A']['default'].in_features, v_proj_named_children['lora_A']['default'].out_features) == (expected_lora_in_features, expected_lora_out_features)
        assert isinstance(v_proj_named_children['lora_B']['default'], torch.nn.Linear)
        assert (v_proj_named_children['lora_B']['default'].in_features, v_proj_named_children['lora_B']['default'].out_features) == (expected_lora_out_features, expected_lora_in_features)
        q_proj_named_children: dict[str, torch.nn.Modeule] = dict(attention_layer.q_proj.named_children())
        assert isinstance(q_proj_named_children['lora_A']['default'], torch.nn.Linear)
        assert (q_proj_named_children['lora_A']['default'].in_features, q_proj_named_children['lora_A']['default'].out_features) == (expected_lora_in_features, expected_lora_out_features)
        assert isinstance(q_proj_named_children['lora_B']['default'], torch.nn.Linear)
        assert (q_proj_named_children['lora_B']['default'].in_features, q_proj_named_children['lora_B']['default'].out_features) == (expected_lora_out_features, expected_lora_in_features)

@pytest.mark.llm
@pytest.mark.parametrize('backend', [pytest.param(LOCAL_BACKEND, id='local')])
@pytest.mark.parametrize('finetune_strategy,adapter_args', [pytest.param(None, {}, id='full'), pytest.param('lora', {}, id='lora-defaults'), pytest.param('lora', {'r': 4, 'dropout': 0.1}, id='lora-modified-defaults'), pytest.param('lora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}}, id='lora_merged'), pytest.param('lora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}}, id='lora_not_merged'), pytest.param('adalora', {}, id='adalora-defaults'), pytest.param('adalora', {'init_r': 8, 'beta1': 0.8}, id='adalora-modified-defaults'), pytest.param('adalora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}}, id='adalora_merged'), pytest.param('adalora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}}, id='adalora_not_merged'), pytest.param('adaption_prompt', {}, id='adaption_prompt-defaults'), pytest.param('adaption_prompt', {'adapter_len': 6, 'adapter_layers': 1}, id='adaption_prompt-modified-defaults')])
def test_llm_finetuning_strategies(tmpdir, csv_filename, backend, finetune_strategy, adapter_args):
    if False:
        print('Hello World!')
    (train_df, prediction_df, config) = _prepare_finetuning_test(csv_filename, finetune_strategy, backend, adapter_args)
    output_directory: str = str(tmpdir)
    model_directory: str = pathlib.Path(output_directory) / 'api_experiment_run' / 'model'
    model = LudwigModel(config)
    model.train(dataset=train_df, output_directory=output_directory, skip_save_processed_input=False)
    model = LudwigModel.load(str(model_directory), backend=backend)
    base_model = LLM(ModelConfig.from_dict(config))
    assert not _compare_models(base_model, model.model)
    (preds, _) = model.predict(dataset=prediction_df, output_directory=output_directory)
    preds = convert_preds(preds)
    assert preds

@pytest.mark.llm
@pytest.mark.parametrize('finetune_strategy,adapter_args,quantization', [pytest.param('lora', {}, {'bits': 4}, id='qlora-4bit'), pytest.param('lora', {}, {'bits': 8}, id='qlora-8bit')])
def test_llm_finetuning_strategies_quantized(tmpdir, csv_filename, finetune_strategy, adapter_args, quantization):
    if False:
        print('Hello World!')
    if _finetune_strategy_requires_cuda(finetune_strategy_name=finetune_strategy, quantization_args=quantization) and (not (torch.cuda.is_available() and torch.cuda.device_count()) > 0):
        pytest.skip('Skip: quantization requires GPU and none are available.')
    backend = LOCAL_BACKEND
    (train_df, prediction_df, config) = _prepare_finetuning_test(csv_filename, finetune_strategy, backend, adapter_args)
    config['backend'] = backend
    config[QUANTIZATION] = quantization
    model = LudwigModel(config)
    model.train(dataset=train_df, output_directory=str(tmpdir), skip_save_processed_input=False)
    model = LudwigModel.load(os.path.join(str(tmpdir), 'api_experiment_run', 'model'))
    base_model = LLM(ModelConfig.from_dict(config))
    assert not _compare_models(base_model, model.model)
    (preds, _) = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)
    assert preds

@pytest.mark.llm
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason='test requires at least 1 gpu')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires gpu support')
@pytest.mark.parametrize('finetune_strategy,adapter_args,quantization,error_raised', [pytest.param('lora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}}, {'bits': 4}, (ImportError, 'Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` '), id='qlora-4bit-not-merged'), pytest.param('lora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}}, {'bits': 8}, (ImportError, 'Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` '), id='qlora-8bit-merged'), pytest.param('lora', {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}}, {'bits': 8}, (ImportError, 'Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` '), id='qlora-8bit-not-merged')])
def test_llm_lora_finetuning_merge_and_unload_quantized_accelerate_required(csv_filename, finetune_strategy, adapter_args, quantization, error_raised):
    if False:
        for i in range(10):
            print('nop')
    input_features: List[dict] = [text_feature(name='input', encoder={'type': 'passthrough'})]
    output_features: List[dict] = [text_feature(name='output')]
    config: dict = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 2}, ADAPTER: {TYPE: finetune_strategy, **adapter_args}, QUANTIZATION: quantization}
    model = LudwigModel(config)
    error_class: type
    error_message: str
    (error_class, error_message) = error_raised
    with pytest.raises(error_class) as excinfo:
        train_df = generate_data(input_features, output_features, filename=csv_filename, num_examples=3)
        model.train(dataset=train_df)
    assert str(excinfo.value) == error_message

@pytest.mark.llm
def test_llm_lora_finetuning_merge_and_unload_4_bit_quantization_not_supported(local_backend: dict):
    if False:
        for i in range(10):
            print('nop')
    input_features: List[dict] = [text_feature(name='input', encoder={'type': 'passthrough'})]
    output_features: List[dict] = [text_feature(name='output')]
    finetune_strategy: str = 'lora'
    config: dict = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 2}, ADAPTER: {TYPE: finetune_strategy, POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}}, QUANTIZATION: {'bits': 4}, BACKEND: local_backend}
    expected_error_class: type = ludwig_error.ConfigValidationError
    expected_error_message: str = 'This operation will entail merging LoRA layers on a 4-bit quantized model.  Calling "save_pretrained()" on that model is currently unsupported.  If you want to merge the LoRA adapter weights into the base model, you need to use 8-bit quantization or do non-quantized based training by removing the quantization section from your Ludwig configuration.'
    with pytest.raises(expected_error_class) as excinfo:
        _ = LudwigModel(config)
    assert str(excinfo.value) == expected_error_message

@pytest.mark.llm
@pytest.mark.parametrize('backend', [pytest.param(LOCAL_BACKEND, id='local')])
@pytest.mark.parametrize('merge_adapter_into_base_model,expected_lora_in_features,expected_lora_out_features,expected_file_names', [pytest.param(False, 32, 8, ['README.md', 'adapter_config.json', 'adapter_model.bin'], id='lora_not_merged'), pytest.param(True, 32, 32, ['README.md', 'adapter_config.json', 'adapter_model.bin', 'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'], id='lora_merged')])
def test_llm_lora_finetuning_merge_and_unload(tmpdir, csv_filename, backend, merge_adapter_into_base_model, expected_lora_in_features, expected_lora_out_features, expected_file_names):
    if False:
        i = 10
        return i + 15
    finetune_strategy: str = 'lora'
    adapter_args: dict = {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: merge_adapter_into_base_model}}
    (train_df, prediction_df, config) = _prepare_finetuning_test(csv_filename=csv_filename, finetune_strategy=finetune_strategy, backend=backend, adapter_args=adapter_args)
    output_directory: str = str(tmpdir)
    model_directory: str = pathlib.Path(output_directory) / 'api_experiment_run' / 'model'
    model_weights_directory: str = pathlib.Path(output_directory) / 'api_experiment_run' / 'model' / 'model_weights'
    model = LudwigModel(config)
    model.train(dataset=train_df, output_directory=output_directory, skip_save_processed_input=False)
    _verify_lm_lora_finetuning_layers(attention_layer=model.model.model.base_model.model.transformer.h[1].attn, merge_adapter_into_base_model=merge_adapter_into_base_model, model_weights_directory=model_weights_directory, expected_lora_in_features=expected_lora_in_features, expected_lora_out_features=expected_lora_out_features, expected_file_names=expected_file_names)
    model = LudwigModel.load(str(model_directory), backend=backend)
    _verify_lm_lora_finetuning_layers(attention_layer=model.model.model.base_model.model.transformer.h[1].attn, merge_adapter_into_base_model=merge_adapter_into_base_model, model_weights_directory=model_weights_directory, expected_lora_in_features=expected_lora_in_features, expected_lora_out_features=expected_lora_out_features, expected_file_names=expected_file_names)

@pytest.mark.llm
@pytest.mark.parametrize('use_adapter', [True, False], ids=['with_adapter', 'without_adapter'])
def test_llm_training_with_gradient_checkpointing(tmpdir, csv_filename, use_adapter):
    if False:
        while True:
            i = 10
    input_features = [text_feature(name='input', encoder={'type': 'passthrough'})]
    output_features = [text_feature(name='output')]
    df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'hf-internal-testing/tiny-random-BartModel', INPUT_FEATURES: input_features, OUTPUT_FEATURES: output_features, TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 1, 'enable_gradient_checkpointing': True}}
    if use_adapter:
        config[ADAPTER] = {TYPE: 'lora'}
    model = LudwigModel(config)
    assert model.config_obj.trainer.enable_gradient_checkpointing
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=False)

@pytest.mark.llm
def test_lora_wrap_on_init():
    if False:
        while True:
            i = 10
    from peft import PeftModel
    from transformers import PreTrainedModel
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')], TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 2}}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    assert isinstance(model.model, PreTrainedModel)
    assert not isinstance(model.model, PeftModel)
    config[ADAPTER] = {TYPE: 'lora'}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    model.prepare_for_training()
    assert not isinstance(model.model, PreTrainedModel)
    assert isinstance(model.model, PeftModel)

def test_llama_rope_scaling():
    if False:
        for i in range(10):
            print('nop')
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')], TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 2}, 'model_parameters': {'rope_scaling': {'type': 'dynamic', 'factor': 2.0}}}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    assert model.model.config.rope_scaling
    assert model.model.config.rope_scaling['type'] == 'dynamic'
    assert model.model.config.rope_scaling['factor'] == 2.0

def test_default_max_sequence_length():
    if False:
        return 10
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: TEST_MODEL_NAME, INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')], TRAINER: {TYPE: 'finetune', BATCH_SIZE: 8, EPOCHS: 2}, ADAPTER: {TYPE: 'lora', PRETRAINED_ADAPTER_WEIGHTS: 'Infernaught/test_adapter_weights'}, BACKEND: {TYPE: 'local'}}
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].preprocessing.max_sequence_length is None
    assert config_obj.output_features[0].preprocessing.max_sequence_length is None

@pytest.mark.llm
@pytest.mark.parametrize('adapter', ['lora', 'adalora', 'adaption_prompt'])
def test_load_pretrained_adapter_weights(adapter):
    if False:
        print('Hello World!')
    from peft import PeftModel
    from transformers import PreTrainedModel
    if adapter == 'lora':
        weights = 'Infernaught/test_adapter_weights'
        base_model = TEST_MODEL_NAME
    elif adapter == 'adalora':
        weights = 'Infernaught/test_adalora_weights'
        base_model = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
    elif adapter == 'adaption_prompt':
        weights = 'Infernaught/test_ap_weights'
        base_model = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
    else:
        raise ()
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: base_model, INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')], TRAINER: {TYPE: 'none', BATCH_SIZE: 8, EPOCHS: 2}, ADAPTER: {TYPE: adapter, PRETRAINED_ADAPTER_WEIGHTS: weights}, BACKEND: {TYPE: 'local'}}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    assert model.config_obj.adapter.pretrained_adapter_weights
    assert model.config_obj.adapter.pretrained_adapter_weights == weights
    model.prepare_for_training()
    assert not isinstance(model.model, PreTrainedModel)
    assert isinstance(model.model, PeftModel)
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].preprocessing.max_sequence_length is None
    assert config_obj.output_features[0].preprocessing.max_sequence_length is None

def _compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module) -> bool:
    if False:
        return 10
    if model_1.__class__.__name__ != model_2.__class__.__name__:
        return False
    if hasattr(model_1, 'model') and hasattr(model_2, 'model') and (not _compare_models(model_1=model_1.model, model_2=model_2.model)):
        return False
    for (key_item_1, key_item_2) in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if not torch.equal(key_item_1[1], key_item_2[1]):
            return False
    return True

def test_global_max_sequence_length_for_llms():
    if False:
        i = 10
        return i + 15
    "Ensures that user specified global_max_sequence_length can never be greater than the model's context\n    length."
    config = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: 'HuggingFaceH4/tiny-random-LlamaForCausalLM', INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')]}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    assert model.global_max_sequence_length == 2048
    config['preprocessing'] = {'global_max_sequence_length': 4096}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    assert model.global_max_sequence_length == 2048

def test_local_path_loading(tmpdir):
    if False:
        return 10
    'Tests that local paths can be used to load models.'
    from huggingface_hub import snapshot_download
    local_path: str = f'{str(tmpdir)}/test_local_path_loading'
    repo_id: str = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
    os.makedirs(local_path, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_path)
    config1 = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: local_path, INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')]}
    config_obj1 = ModelConfig.from_dict(config1)
    model1 = LLM(config_obj1)
    config2 = {MODEL_TYPE: MODEL_LLM, BASE_MODEL: repo_id, INPUT_FEATURES: [text_feature(name='input', encoder={'type': 'passthrough'})], OUTPUT_FEATURES: [text_feature(name='output')]}
    config_obj2 = ModelConfig.from_dict(config2)
    model2 = LLM(config_obj2)
    assert _compare_models(model1.model, model2.model)