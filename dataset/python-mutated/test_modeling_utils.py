import copy
import glob
import json
import os
import os.path
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from huggingface_hub import HfFolder, delete_repo
from huggingface_hub.file_download import http_get
from pytest import mark
from requests.exceptions import HTTPError
from transformers import AutoConfig, AutoModel, PretrainedConfig, is_torch_available, logging
from transformers.testing_utils import TOKEN, USER, CaptureLogger, TestCasePlus, is_staging_test, require_accelerate, require_flax, require_safetensors, require_tf, require_torch, require_torch_accelerator, require_torch_multi_accelerator, require_usr_bin_time, slow, torch_device
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.import_utils import is_flax_available, is_tf_available, is_torchdynamo_available
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig, NoSuperInitConfig
if is_torch_available():
    import torch
    from safetensors.torch import save_file as safe_save_file
    from test_module.custom_modeling import CustomModel, NoSuperInitModel
    from torch import nn
    from transformers import BERT_PRETRAINED_MODEL_ARCHIVE_LIST, AutoModelForCausalLM, AutoTokenizer, BertConfig, BertModel, CLIPTextModel, PreTrainedModel, T5Config, T5ForConditionalGeneration
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.modeling_utils import shard_checkpoint

    class BaseModel(PreTrainedModel):
        base_model_prefix = 'base'
        config_class = PretrainedConfig

        def __init__(self, config):
            if False:
                while True:
                    i = 10
            super().__init__(config)
            self.linear = nn.Linear(5, 5)
            self.linear_2 = nn.Linear(5, 5)

        def forward(self, x):
            if False:
                print('Hello World!')
            return self.linear_2(self.linear(x))

    class BaseModelWithTiedWeights(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(config)
            self.linear = nn.Linear(5, 5)
            self.linear_2 = nn.Linear(5, 5)

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return self.linear_2(self.linear(x))

        def tie_weights(self):
            if False:
                print('Hello World!')
            self.linear_2.weight = self.linear.weight

    class ModelWithHead(PreTrainedModel):
        base_model_prefix = 'base'
        config_class = PretrainedConfig

        def _init_weights(self, module):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def __init__(self, config):
            if False:
                i = 10
                return i + 15
            super().__init__(config)
            self.base = BaseModel(config)
            self.linear = nn.Linear(5, 5)
            self.linear2 = nn.Linear(5, 5)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            return self.linear2(self.linear(self.base(x)))

    class ModelWithHeadAndTiedWeights(PreTrainedModel):
        base_model_prefix = 'base'
        config_class = PretrainedConfig

        def _init_weights(self, module):
            if False:
                return 10
            pass

        def __init__(self, config):
            if False:
                i = 10
                return i + 15
            super().__init__(config)
            self.base = BaseModel(config)
            self.decoder = nn.Linear(5, 5)

        def forward(self, x):
            if False:
                i = 10
                return i + 15
            return self.decoder(self.base(x))

        def tie_weights(self):
            if False:
                while True:
                    i = 10
            self.decoder.weight = self.base.linear.weight
if is_flax_available():
    from transformers import FlaxBertModel
if is_tf_available():
    from transformers import TFBertModel
TINY_T5 = 'patrickvonplaten/t5-tiny-random'
TINY_BERT_FOR_TOKEN_CLASSIFICATION = 'hf-internal-testing/tiny-bert-for-token-classification'

def check_models_equal(model1, model2):
    if False:
        return 10
    models_are_equal = True
    for (model1_p, model2_p) in zip(model1.parameters(), model2.parameters()):
        if model1_p.data.ne(model2_p.data).sum() > 0:
            models_are_equal = False
    return models_are_equal

@require_torch
class ModelUtilsTest(TestCasePlus):

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = BertConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, PretrainedConfig)
            model = BertModel.from_pretrained(model_name)
            (model, loading_info) = BertModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, PreTrainedModel)
            self.assertEqual(len(loading_info['missing_keys']), 0)
            self.assertEqual(len(loading_info['unexpected_keys']), 8)
            self.assertEqual(len(loading_info['mismatched_keys']), 0)
            self.assertEqual(len(loading_info['error_msgs']), 0)
            config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            config.name_or_path = model_name
            model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(model.config, config)

    def test_model_from_pretrained_subfolder(self):
        if False:
            while True:
                i = 10
        config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
        model = BertModel(config)
        subfolder = 'bert'
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder))
            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)
            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)
        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_subfolder_sharded(self):
        if False:
            while True:
                i = 10
        config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
        model = BertModel(config)
        subfolder = 'bert'
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder), max_shard_size='10KB')
            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)
            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)
        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_hub_subfolder(self):
        if False:
            return 10
        subfolder = 'bert'
        model_id = 'hf-internal-testing/tiny-random-bert-subfolder'
        with self.assertRaises(OSError):
            _ = BertModel.from_pretrained(model_id)
        model = BertModel.from_pretrained(model_id, subfolder=subfolder)
        self.assertIsNotNone(model)

    def test_model_from_pretrained_hub_subfolder_sharded(self):
        if False:
            for i in range(10):
                print('nop')
        subfolder = 'bert'
        model_id = 'hf-internal-testing/tiny-random-bert-sharded-subfolder'
        with self.assertRaises(OSError):
            _ = BertModel.from_pretrained(model_id)
        model = BertModel.from_pretrained(model_id, subfolder=subfolder)
        self.assertIsNotNone(model)

    def test_model_from_pretrained_with_different_pretrained_model_name(self):
        if False:
            return 10
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
        self.assertIsNotNone(model)
        logger = logging.get_logger('transformers.configuration_utils')
        with CaptureLogger(logger) as cl:
            BertModel.from_pretrained(TINY_T5)
        self.assertTrue('You are using a model of type t5 to instantiate a model of type bert' in cl.out)

    def test_model_from_config_torch_dtype(self):
        if False:
            while True:
                i = 10
        config = T5Config.from_pretrained(TINY_T5)
        model = AutoModel.from_config(config)
        self.assertEqual(model.dtype, torch.float32)
        model = AutoModel.from_config(config, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        with self.assertRaises(ValueError):
            model = AutoModel.from_config(config, torch_dtype=torch.int64)

    def test_model_from_pretrained_torch_dtype(self):
        if False:
            i = 10
            return i + 15
        model_path = self.get_auto_remove_tmp_dir()
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
        self.assertEqual(model.dtype, torch.float32)

        def remove_torch_dtype(model_path):
            if False:
                while True:
                    i = 10
            file = f'{model_path}/config.json'
            with open(file, 'r', encoding='utf-8') as f:
                s = json.load(f)
            s.pop('torch_dtype')
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(s, f)
        model.save_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.assertEqual(model.dtype, torch.float32)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto')
        self.assertEqual(model.dtype, torch.float32)
        remove_torch_dtype(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto')
        self.assertEqual(model.dtype, torch.float32)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        model = model.half()
        model.save_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto')
        self.assertEqual(model.config.torch_dtype, torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        with open(f'{model_path}/config.json') as f:
            config_dict = json.load(f)
        self.assertEqual(config_dict['torch_dtype'], 'float16')
        remove_torch_dtype(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto')
        self.assertEqual(model.dtype, torch.float16)
        model = AutoModel.from_pretrained(model_path, torch_dtype='auto')
        self.assertEqual(model.dtype, torch.float16)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype='auto')
        self.assertNotEqual(model.config.torch_dtype, 'auto')
        self.assertEqual(model.dtype, torch.float32)
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        model = AutoModel.from_pretrained(TINY_BERT_FOR_TOKEN_CLASSIFICATION, torch_dtype='auto')
        self.assertEqual(model.dtype, torch.float32)

    def test_no_super_init_config_and_model(self):
        if False:
            print('Hello World!')
        config = NoSuperInitConfig(attribute=32)
        model = NoSuperInitModel(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            new_model = NoSuperInitModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_shard_checkpoint(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(100, 200, bias=False), torch.nn.Linear(200, 200, bias=False), torch.nn.Linear(200, 100, bias=False), torch.nn.Linear(100, 50, bias=False))
        state_dict = model.state_dict()
        with self.subTest('No shard when max size is bigger than model size'):
            (shards, index) = shard_checkpoint(state_dict)
            self.assertIsNone(index)
            self.assertDictEqual(shards, {WEIGHTS_NAME: state_dict})
        with self.subTest('Test sharding, no weights bigger than max size'):
            (shards, index) = shard_checkpoint(state_dict, max_shard_size='300kB')
            self.assertDictEqual(index, {'metadata': {'total_size': 340000}, 'weight_map': {'0.weight': 'pytorch_model-00001-of-00002.bin', '1.weight': 'pytorch_model-00001-of-00002.bin', '2.weight': 'pytorch_model-00002-of-00002.bin', '3.weight': 'pytorch_model-00002-of-00002.bin'}})
            shard1 = {'0.weight': state_dict['0.weight'], '1.weight': state_dict['1.weight']}
            shard2 = {'2.weight': state_dict['2.weight'], '3.weight': state_dict['3.weight']}
            self.assertDictEqual(shards, {'pytorch_model-00001-of-00002.bin': shard1, 'pytorch_model-00002-of-00002.bin': shard2})
        with self.subTest('Test sharding with weights bigger than max size'):
            (shards, index) = shard_checkpoint(state_dict, max_shard_size='100kB')
            self.assertDictEqual(index, {'metadata': {'total_size': 340000}, 'weight_map': {'0.weight': 'pytorch_model-00001-of-00003.bin', '1.weight': 'pytorch_model-00002-of-00003.bin', '2.weight': 'pytorch_model-00003-of-00003.bin', '3.weight': 'pytorch_model-00003-of-00003.bin'}})
            shard1 = {'0.weight': state_dict['0.weight']}
            shard2 = {'1.weight': state_dict['1.weight']}
            shard3 = {'2.weight': state_dict['2.weight'], '3.weight': state_dict['3.weight']}
            self.assertDictEqual(shards, {'pytorch_model-00001-of-00003.bin': shard1, 'pytorch_model-00002-of-00003.bin': shard2, 'pytorch_model-00003-of-00003.bin': shard3})

    def test_checkpoint_sharding_local_bin(self):
        if False:
            for i in range(10):
                print('nop')
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            for max_size in ['50kB', '50kiB', '100kB', '100kiB', '200kB', '200kiB']:
                model.save_pretrained(tmp_dir, max_shard_size=max_size, safe_serialization=False)
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith('.bin'):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)
                index_file = os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
                for (shard_file, size) in shard_to_size.items():
                    if max_size.endswith('kiB'):
                        max_size_int = int(max_size[:-3]) * 2 ** 10
                    else:
                        max_size_int = int(max_size[:-2]) * 10 ** 3
                    if size >= max_size_int + 50000:
                        state_dict = torch.load(shard_file)
                        self.assertEqual(len(state_dict), 1)
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.loads(f.read())
                all_shards = set(index['weight_map'].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith('.bin')}
                self.assertSetEqual(all_shards, shards_found)
                new_model = BertModel.from_pretrained(tmp_dir)
                for (p1, p2) in zip(model.parameters(), new_model.parameters()):
                    self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_sharding_from_hub(self):
        if False:
            i = 10
            return i + 15
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-sharded')
        ref_model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        for (p1, p2) in zip(model.parameters(), ref_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local_bin(self):
        if False:
            i = 10
            return i + 15
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant='v2', safe_serialization=False)
            weights_name = '.'.join(WEIGHTS_NAME.split('.')[:-1] + ['v2'] + ['bin'])
            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)
            new_model = BertModel.from_pretrained(tmp_dir, variant='v2')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local_sharded_bin(self):
        if False:
            return 10
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant='v2', max_shard_size='50kB', safe_serialization=False)
            weights_index_name = '.'.join(WEIGHTS_INDEX_NAME.split('.')[:-1] + ['v2'] + ['json'])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))
            for i in range(1, 5):
                weights_name = '.'.join(WEIGHTS_NAME.split('.')[:-1] + [f'v2-0000{i}-of-00005'] + ['bin'])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)
            new_model = BertModel.from_pretrained(tmp_dir, variant='v2')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_checkpoint_variant_local_safe(self):
        if False:
            for i in range(10):
                print('nop')
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant='v2', safe_serialization=True)
            weights_name = '.'.join(SAFE_WEIGHTS_NAME.split('.')[:-1] + ['v2'] + ['safetensors'])
            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)
            new_model = BertModel.from_pretrained(tmp_dir, variant='v2')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_checkpoint_variant_local_sharded_safe(self):
        if False:
            while True:
                i = 10
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant='v2', max_shard_size='50kB', safe_serialization=True)
            weights_index_name = '.'.join(SAFE_WEIGHTS_INDEX_NAME.split('.')[:-1] + ['v2'] + ['json'])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
            for i in range(1, 5):
                weights_name = '.'.join(SAFE_WEIGHTS_NAME.split('.')[:-1] + [f'v2-0000{i}-of-00005'] + ['safetensors'])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)
            new_model = BertModel.from_pretrained(tmp_dir, variant='v2')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_hub(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant', cache_dir=tmp_dir)
            model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant', cache_dir=tmp_dir, variant='v2')
        self.assertIsNotNone(model)

    def test_checkpoint_variant_hub_sharded(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant-sharded', cache_dir=tmp_dir)
            model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant-sharded', cache_dir=tmp_dir, variant='v2')
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_safe(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant-safe', cache_dir=tmp_dir)
            model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant-safe', cache_dir=tmp_dir, variant='v2')
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_sharded_safe(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant-sharded-safe', cache_dir=tmp_dir)
            model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant-sharded-safe', cache_dir=tmp_dir, variant='v2')
        self.assertIsNotNone(model)

    def test_checkpoint_variant_save_load_bin(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-variant', cache_dir=tmp_dir, variant='v2')
            weights_name = '.'.join(WEIGHTS_NAME.split('.')[:-1] + ['v2'] + ['bin'])
            model.save_pretrained(tmp_dir, variant='v2', safe_serialization=False)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))
            model.save_pretrained(tmp_dir, safe_serialization=False)
            weights_name = '.'.join(WEIGHTS_NAME.split('.')[:-1] + ['v2'] + ['bin'])
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
        self.assertIsNotNone(model)

    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_functional(self):
        if False:
            print('Hello World!')
        mnames = ['hf-internal-testing/tiny-random-bert-sharded', 'hf-internal-testing/tiny-random-bert']
        for mname in mnames:
            _ = BertModel.from_pretrained(mname, low_cpu_mem_usage=True)

    @require_usr_bin_time
    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_measured(self):
        if False:
            while True:
                i = 10
        mname = 'bert-base-cased'
        preamble = 'from transformers import AutoModel'
        one_liner_str = f'{preamble}; AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=False)'
        max_rss_normal = self.python_one_liner_max_rss(one_liner_str)
        one_liner_str = f'{preamble};  AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=True)'
        max_rss_low_mem = self.python_one_liner_max_rss(one_liner_str)
        diff_bytes = max_rss_normal - max_rss_low_mem
        diff_percent = diff_bytes / max_rss_low_mem
        self.assertGreater(diff_percent, 0.15, f'should use less CPU memory for low_cpu_mem_usage=True, but got max_rss_normal={max_rss_normal} and max_rss_low_mem={max_rss_low_mem}')

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_multi_accelerator
    @slow
    def test_model_parallelism_gpt2(self):
        if False:
            while True:
                i = 10
        device_map = {'transformer.wte': 0, 'transformer.wpe': 0, 'lm_head': 0, 'transformer.ln_f': 1}
        for i in range(12):
            device_map[f'transformer.h.{i}'] = 0 if i <= 5 else 1
        model = AutoModelForCausalLM.from_pretrained('gpt2', device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        inputs = tokenizer('Hello, my name is', return_tensors='pt')
        output = model.generate(inputs['input_ids'].to(0))
        text_output = tokenizer.decode(output[0].tolist())
        self.assertEqual(text_output, "Hello, my name is John. I'm a writer, and I'm a writer. I'm")

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_from_pretrained_disk_offload_task_model(self):
        if False:
            i = 10
            return i + 15
        model = AutoModel.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        device_map = {'transformer.wte': 0, 'transformer.wpe': 0, 'transformer.h.0': 'cpu', 'transformer.h.1': 'cpu', 'transformer.h.2': 'cpu', 'transformer.h.3': 'disk', 'transformer.h.4': 'disk', 'transformer.ln_f': 0, 'lm_head': 0}
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = torch.tensor([[1, 2, 3]]).to(0)
            model.save_pretrained(tmp_dir)
            new_model = AutoModelForCausalLM.from_pretrained(tmp_dir).to(0)
            outputs1 = new_model.to(0)(inputs)
            offload_folder = os.path.join(tmp_dir, 'offload')
            new_model_with_offload = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map, offload_folder=offload_folder)
            outputs2 = new_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))
            offload_folder = os.path.join(tmp_dir, 'offload')
            new_model_with_offload = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map, offload_folder=offload_folder, offload_state_dict=True)
            outputs2 = new_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_from_pretrained_disk_offload_derived_to_base_model(self):
        if False:
            for i in range(10):
                print('nop')
        derived_model = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        device_map = {'wte': 0, 'wpe': 0, 'h.0': 'cpu', 'h.1': 'cpu', 'h.2': 'cpu', 'h.3': 'disk', 'h.4': 'disk', 'ln_f': 0}
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = torch.tensor([[1, 2, 3]]).to(0)
            derived_model.save_pretrained(tmp_dir, use_safetensors=True)
            base_model = AutoModel.from_pretrained(tmp_dir)
            outputs1 = base_model.to(0)(inputs)
            offload_folder = os.path.join(tmp_dir, 'offload')
            base_model_with_offload = AutoModel.from_pretrained(tmp_dir, device_map=device_map, offload_folder=offload_folder)
            outputs2 = base_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1[0].cpu(), outputs2[0].cpu()))
            new_model_with_offload = AutoModel.from_pretrained(tmp_dir, device_map=device_map, offload_folder=offload_folder, offload_state_dict=True)
            outputs2 = new_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1[0].cpu(), outputs2[0].cpu()))

    def test_cached_files_are_used_when_internet_is_down(self):
        if False:
            return 10
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        _ = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            _ = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
            mock_head.assert_called()

    def test_load_from_one_file(self):
        if False:
            print('Hello World!')
        try:
            tmp_file = tempfile.mktemp()
            with open(tmp_file, 'wb') as f:
                http_get('https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/pytorch_model.bin', f)
            config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
            _ = BertModel.from_pretrained(tmp_file, config=config)
        finally:
            os.remove(tmp_file)

    def test_legacy_load_from_url(self):
        if False:
            for i in range(10):
                print('nop')
        config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
        _ = BertModel.from_pretrained('https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/pytorch_model.bin', config=config)

    @require_safetensors
    def test_use_safetensors(self):
        if False:
            print('Hello World!')
        with self.assertRaises(OSError) as env_error:
            AutoModel.from_pretrained('hf-internal-testing/tiny-random-RobertaModel', use_safetensors=True)
        self.assertTrue('model.safetensors or model.safetensors.index.json and thus cannot be loaded with `safetensors`' in str(env_error.exception))
        with self.assertRaises(OSError) as env_error:
            BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-safetensors', use_safetensors=False)
        self.assertTrue('does not appear to have a file named pytorch_model.bin' in str(env_error.exception))
        with tempfile.TemporaryDirectory() as tmp_dir:
            CLIPTextModel.from_pretrained('hf-internal-testing/diffusers-stable-diffusion-tiny-all', subfolder='text_encoder', use_safetensors=False, cache_dir=tmp_dir)
            all_downloaded_files = glob.glob(os.path.join(tmp_dir, '*', 'snapshots', '*', '*', '*'))
            self.assertTrue(any((f.endswith('bin') for f in all_downloaded_files)))
            self.assertFalse(any((f.endswith('safetensors') for f in all_downloaded_files)))
        with tempfile.TemporaryDirectory() as tmp_dir:
            CLIPTextModel.from_pretrained('hf-internal-testing/diffusers-stable-diffusion-tiny-all', subfolder='text_encoder', use_safetensors=True, cache_dir=tmp_dir)
            all_downloaded_files = glob.glob(os.path.join(tmp_dir, '*', 'snapshots', '*', '*', '*'))
            self.assertTrue(any((f.endswith('safetensors') for f in all_downloaded_files)))
            self.assertFalse(any((f.endswith('bin') for f in all_downloaded_files)))

    @require_safetensors
    def test_safetensors_save_and_load(self):
        if False:
            for i in range(10):
                print('nop')
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
            new_model = BertModel.from_pretrained(tmp_dir)
            for (p1, p2) in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_load_from_hub(self):
        if False:
            return 10
        safetensors_model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-safetensors')
        pytorch_model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        for (p1, p2) in zip(safetensors_model.parameters(), pytorch_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_save_and_load_sharded(self):
        if False:
            print('Hello World!')
        model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size='100kB')
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            new_model = BertModel.from_pretrained(tmp_dir)
            for (p1, p2) in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_load_from_hub_sharded(self):
        if False:
            print('Hello World!')
        safetensors_model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-sharded-safetensors')
        pytorch_model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert-sharded')
        for (p1, p2) in zip(safetensors_model.parameters(), pytorch_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_base_model_to_head_model_load(self):
        if False:
            return 10
        base_model = BaseModel(PretrainedConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_model.save_pretrained(tmp_dir, safe_serialization=False)
            model = ModelWithHead.from_pretrained(tmp_dir)
            for (p1, p2) in zip(model.base.parameters(), base_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
            base_state_dict = base_model.state_dict()
            head_state_dict = model.state_dict()
            base_state_dict['linear2.weight'] = head_state_dict['linear2.weight']
            base_state_dict['linear2.bias'] = head_state_dict['linear2.bias']
            safe_save_file(base_state_dict, os.path.join(tmp_dir, SAFE_WEIGHTS_NAME), metadata={'format': 'pt'})
            with self.assertRaisesRegex(ValueError, 'The state dictionary of the model you are trying to load is corrupted.'):
                _ = ModelWithHead.from_pretrained(tmp_dir)

    def test_tied_weights_reload(self):
        if False:
            return 10
        model = BaseModelWithTiedWeights(PretrainedConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            new_model = BaseModelWithTiedWeights.from_pretrained(tmp_dir)
            self.assertIs(new_model.linear.weight, new_model.linear_2.weight)
            state_dict = model.state_dict()
            del state_dict['linear_2.weight']
            torch.save(state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))
            (new_model, load_info) = BaseModelWithTiedWeights.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertListEqual(load_info['missing_keys'], [])
            self.assertIs(new_model.linear.weight, new_model.linear_2.weight)
            model.save_pretrained(tmp_dir)
            (new_model, load_info) = ModelWithHeadAndTiedWeights.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIs(new_model.base.linear.weight, new_model.decoder.weight)
            self.assertListEqual(load_info['missing_keys'], ['decoder.bias'])

    def test_unexpected_keys_warnings(self):
        if False:
            i = 10
            return i + 15
        model = ModelWithHead(PretrainedConfig())
        logger = logging.get_logger('transformers.modeling_utils')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            with CaptureLogger(logger) as cl:
                (_, loading_info) = BaseModel.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertNotIn('were not used when initializing ModelWithHead', cl.out)
            self.assertEqual(set(loading_info['unexpected_keys']), {'linear.weight', 'linear.bias', 'linear2.weight', 'linear2.bias'})
            state_dict = model.state_dict()
            state_dict['added_key'] = copy.deepcopy(state_dict['linear.weight'])
            safe_save_file(state_dict, os.path.join(tmp_dir, SAFE_WEIGHTS_NAME), metadata={'format': 'pt'})
            with CaptureLogger(logger) as cl:
                (_, loading_info) = ModelWithHead.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIn("were not used when initializing ModelWithHead: ['added_key']", cl.out)
            self.assertEqual(loading_info['unexpected_keys'], ['added_key'])

    def test_warn_if_padding_and_no_attention_mask(self):
        if False:
            while True:
                i = 10
        logger = logging.get_logger('transformers.modeling_utils')
        with self.subTest('Ensure no warnings when pad_token_id is None.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config_no_pad_token = PretrainedConfig()
                config_no_pad_token.pad_token_id = None
                model = ModelWithHead(config_no_pad_token)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn('We strongly recommend passing in an `attention_mask`', cl.out)
        with self.subTest('Ensure no warnings when there is an attention_mask.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            self.assertNotIn('We strongly recommend passing in an `attention_mask`', cl.out)
        with self.subTest('Ensure no warnings when there are no pad_token_ids in the input_ids.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[1, 345, 232, 328, 740, 140, 1695, 69, 6078, 2341, 25]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn('We strongly recommend passing in an `attention_mask`', cl.out)
        with self.subTest('Ensure a warning is shown when the input_ids start with a pad_token_id.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn('We strongly recommend passing in an `attention_mask`', cl.out)
        with self.subTest('Ensure a warning is shown when the input_ids end with a pad_token_id.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[432, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn('We strongly recommend passing in an `attention_mask`', cl.out)
        with self.subTest('Ensure that the warning is shown at most once.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertEqual(cl.out.count('We strongly recommend passing in an `attention_mask`'), 1)
        with self.subTest('Ensure a different warning is shown when the pad_token_id is equal to the bos_token_id.'):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                config.bos_token_id = config.pad_token_id
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn('You may ignore this warning if your `pad_token_id`', cl.out)
        if not is_torchdynamo_available():
            return
        with self.subTest('Ensure that the warning code is skipped when compiling with torchdynamo.'):
            logger.warning_once.cache_clear()
            from torch._dynamo import config, testing
            config = PretrainedConfig()
            config.pad_token_id = 0
            model = ModelWithHead(config)
            input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])

            def f(input_ids):
                if False:
                    i = 10
                    return i + 15
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            compile_counter = testing.CompileCounter()
            opt_fn = torch.compile(f, dynamic=True, backend=compile_counter)
            opt_fn(input_ids)
            self.assertEqual(compile_counter.frame_count, 0)

    @require_torch_accelerator
    @slow
    def test_pretrained_low_mem_new_config(self):
        if False:
            while True:
                i = 10
        model_ids = ['gpt2']
        for model_id in model_ids:
            model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
            model_config.n_layer = 48
            model_config.n_head = 25
            model_config.n_embd = 1600
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, config=model_config, ignore_mismatched_sizes=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            model_ref = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)
            self.assertEqual(model.__class__.__name__, model_ref.__class__.__name__)

    def test_generation_config_is_loaded_with_model(self):
        if False:
            while True:
                i = 10
        model = AutoModelForCausalLM.from_pretrained('joaogante/tiny-random-gpt2-with-generation-config')
        self.assertEqual(model.generation_config.transformers_version, 'foo')
        model = AutoModelForCausalLM.from_pretrained('joaogante/tiny-random-gpt2-with-generation-config', device_map='auto')
        self.assertEqual(model.generation_config.transformers_version, 'foo')

    @require_safetensors
    def test_safetensors_torch_from_torch(self):
        if False:
            print('Hello World!')
        model = BertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-only')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_safetensors
    @require_flax
    def test_safetensors_torch_from_flax(self):
        if False:
            for i in range(10):
                print('nop')
        hub_model = BertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-only')
        model = FlaxBertModel.from_pretrained('hf-internal-testing/tiny-bert-flax-only')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(hub_model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_tf
    @require_safetensors
    def test_safetensors_torch_from_tf(self):
        if False:
            return 10
        hub_model = BertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-only')
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-tf-only')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(hub_model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_safetensors
    def test_safetensors_torch_from_torch_sharded(self):
        if False:
            return 10
        model = BertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-only')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size='100kB')
            new_model = BertModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

@require_torch
@is_staging_test
class ModelPushToHubTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        try:
            delete_repo(token=cls._token, repo_id='test-model')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='valid_org/test-model-org')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='test-dynamic-model')
        except HTTPError:
            pass

    @unittest.skip('This test is flaky')
    def test_push_to_hub(self):
        if False:
            i = 10
            return i + 15
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        model = BertModel(config)
        model.push_to_hub('test-model', token=self._token)
        new_model = BertModel.from_pretrained(f'{USER}/test-model')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        delete_repo(token=self._token, repo_id='test-model')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id='test-model', push_to_hub=True, token=self._token)
        new_model = BertModel.from_pretrained(f'{USER}/test-model')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_with_description(self):
        if False:
            return 10
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        model = BertModel(config)
        COMMIT_DESCRIPTION = '\nThe commit description supports markdown synthax see:\n```python\n>>> form transformers import AutoConfig\n>>> config = AutoConfig.from_pretrained("bert-base-uncased")\n```\n'
        commit_details = model.push_to_hub('test-model', use_auth_token=self._token, create_pr=True, commit_description=COMMIT_DESCRIPTION)
        self.assertEqual(commit_details.commit_description, COMMIT_DESCRIPTION)

    @unittest.skip('This test is flaky')
    def test_push_to_hub_in_organization(self):
        if False:
            print('Hello World!')
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        model = BertModel(config)
        model.push_to_hub('valid_org/test-model-org', token=self._token)
        new_model = BertModel.from_pretrained('valid_org/test-model-org')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        delete_repo(token=self._token, repo_id='valid_org/test-model-org')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=self._token, repo_id='valid_org/test-model-org')
        new_model = BertModel.from_pretrained('valid_org/test-model-org')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_dynamic_model(self):
        if False:
            for i in range(10):
                print('nop')
        CustomConfig.register_for_auto_class()
        CustomModel.register_for_auto_class()
        config = CustomConfig(hidden_size=32)
        model = CustomModel(config)
        model.push_to_hub('test-dynamic-model', token=self._token)
        self.assertDictEqual(config.auto_map, {'AutoConfig': 'custom_configuration.CustomConfig', 'AutoModel': 'custom_modeling.CustomModel'})
        new_model = AutoModel.from_pretrained(f'{USER}/test-dynamic-model', trust_remote_code=True)
        self.assertEqual(new_model.__class__.__name__, 'CustomModel')
        for (p1, p2) in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        config = AutoConfig.from_pretrained(f'{USER}/test-dynamic-model', trust_remote_code=True)
        new_model = AutoModel.from_config(config, trust_remote_code=True)
        self.assertEqual(new_model.__class__.__name__, 'CustomModel')

@require_torch
class AttentionMaskTester(unittest.TestCase):

    def check_non_causal(self, bsz, q_len, kv_len, mask_2d, mask_4d):
        if False:
            for i in range(10):
                print('nop')
        mask_indices = (mask_2d != 1)[:, None].broadcast_to((bsz, q_len, kv_len))
        mask_4d_values = mask_4d[:, 0][mask_indices]
        is_inf = mask_4d_values == -float('inf')
        is_min = mask_4d_values == torch.finfo(mask_4d.dtype).min
        assert torch.logical_or(is_inf, is_min).all()

    def check_to_4d(self, mask_converter, q_len, kv_len, additional_mask=None, bsz=3):
        if False:
            i = 10
            return i + 15
        mask_2d = torch.ones((bsz, kv_len), device=torch_device, dtype=torch.long)
        if additional_mask is not None:
            for (bsz_idx, seq_idx) in additional_mask:
                mask_2d[bsz_idx, seq_idx] = 0
        mask_4d = mask_converter.to_4d(mask_2d, query_length=q_len, key_value_length=kv_len)
        assert mask_4d.shape == (bsz, 1, q_len, kv_len)
        assert mask_4d.min() != float('-inf')
        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)
            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
            if 0 in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif not mask_converter.is_causal and context is None:
            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == 0
            if 0 in mask_2d:
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif mask_converter.is_causal and context is not None:
            num_tokens_masked = q_len * (q_len - 1) // 2 + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked
            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
            if 0 in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)

    def check_to_causal(self, mask_converter, q_len, kv_len, bsz=3):
        if False:
            for i in range(10):
                print('nop')
        mask_4d = mask_converter.to_causal_4d(bsz, query_length=q_len, key_value_length=kv_len, device=torch_device)
        if q_len == 1 and mask_converter.sliding_window is None:
            assert mask_4d is None
            return
        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)
            assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
        elif not mask_converter.is_causal and context is None:
            assert (mask_4d != 0).sum().cpu().item() == 0
        elif mask_converter.is_causal and context is not None:
            num_tokens_masked = q_len * (q_len - 1) // 2 + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked
            assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked

    def compute_num_context_mask(self, kv_len, context, q_len):
        if False:
            i = 10
            return i + 15
        c_mask_len = kv_len - context
        num_mask_triangle = c_mask_len * (c_mask_len + 1) // 2
        cut_mask_len = max(c_mask_len - q_len, 0)
        num_cut_mask = cut_mask_len * (cut_mask_len + 1) // 2
        return num_mask_triangle - num_cut_mask

    def test_2d_to_4d_causal(self):
        if False:
            return 10
        mask_converter = AttentionMaskConverter(is_causal=True)
        self.check_to_4d(mask_converter, q_len=1, kv_len=7)
        self.check_to_4d(mask_converter, q_len=3, kv_len=7)
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)
        self.check_to_4d(mask_converter, q_len=1, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=3, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 0), (1, 0), (1, 1)])

    def test_2d_to_4d(self):
        if False:
            i = 10
            return i + 15
        mask_converter = AttentionMaskConverter(is_causal=False)
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_2d_to_4d_causal_sliding(self):
        if False:
            return 10
        mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=5)
        self.check_to_4d(mask_converter, q_len=1, kv_len=7)
        self.check_to_4d(mask_converter, q_len=3, kv_len=7)
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)
        self.check_to_4d(mask_converter, q_len=1, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=3, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_causal_mask(self):
        if False:
            for i in range(10):
                print('nop')
        mask_converter = AttentionMaskConverter(is_causal=True)
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)

    def test_causal_mask_sliding(self):
        if False:
            i = 10
            return i + 15
        mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=3)
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)