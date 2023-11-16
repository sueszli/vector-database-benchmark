from __future__ import annotations
import inspect
import json
import os
import random
import tempfile
import unittest
import unittest.mock as mock
from huggingface_hub import HfFolder, Repository, delete_repo, snapshot_download
from huggingface_hub.file_download import http_get
from requests.exceptions import HTTPError
from transformers import is_tf_available, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import TOKEN, USER, CaptureLogger, _tf_gpu_memory_limit, is_pt_tf_cross_test, is_staging_test, require_safetensors, require_tf, require_torch, slow
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, logging
logger = logging.get_logger(__name__)
if is_tf_available():
    import h5py
    import numpy as np
    import tensorflow as tf
    from transformers import BertConfig, PreTrainedModel, PushToHubCallback, RagRetriever, TFBertForMaskedLM, TFBertForSequenceClassification, TFBertModel, TFPreTrainedModel, TFRagModel
    from transformers.modeling_tf_utils import tf_shard_checkpoint, unpack_inputs
    from transformers.tf_utils import stable_softmax
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if _tf_gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=_tf_gpu_memory_limit)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print('Logical GPUs', logical_gpus)
            except RuntimeError as e:
                print(e)
if is_torch_available():
    from transformers import BertModel

@require_tf
class TFModelUtilsTest(unittest.TestCase):

    def test_cached_files_are_used_when_internet_is_down(self):
        if False:
            return 10
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        _ = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            _ = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
            mock_head.assert_called()

    def test_load_from_one_file(self):
        if False:
            print('Hello World!')
        try:
            tmp_file = tempfile.mktemp()
            with open(tmp_file, 'wb') as f:
                http_get('https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/tf_model.h5', f)
            config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
            _ = TFBertModel.from_pretrained(tmp_file, config=config)
        finally:
            os.remove(tmp_file)

    def test_legacy_load_from_url(self):
        if False:
            i = 10
            return i + 15
        config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
        _ = TFBertModel.from_pretrained('https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/tf_model.h5', config=config)

    def test_unpack_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        class DummyModel:

            def __init__(self):
                if False:
                    print('Hello World!')
                config_kwargs = {'output_attentions': False, 'output_hidden_states': False, 'return_dict': False}
                self.config = PretrainedConfig(**config_kwargs)
                self.main_input_name = 'input_ids'

            @unpack_inputs
            def call(self, input_ids=None, past_key_values=None, output_attentions=None, output_hidden_states=None, return_dict=None):
                if False:
                    print('Hello World!')
                return (input_ids, past_key_values, output_attentions, output_hidden_states, return_dict)

            @unpack_inputs
            def foo(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
                if False:
                    print('Hello World!')
                return (pixel_values, output_attentions, output_hidden_states, return_dict)
        dummy_model = DummyModel()
        input_ids = tf.constant([0, 1, 2, 3], dtype=tf.int32)
        past_key_values = tf.constant([4, 5, 6, 7], dtype=tf.int32)
        pixel_values = tf.constant([8, 9, 10, 11], dtype=tf.int32)
        output = dummy_model.call(input_ids=input_ids, past_key_values=past_key_values)
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertFalse(output[4])
        output = dummy_model.call(input_ids, past_key_values)
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertFalse(output[4])
        output = dummy_model.call(input_ids={'input_ids': input_ids, 'past_key_values': past_key_values})
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertFalse(output[4])
        output = dummy_model.call(input_ids=input_ids, past_key_values=past_key_values, output_attentions=False, return_dict=True)
        tf.debugging.assert_equal(output[0], input_ids)
        tf.debugging.assert_equal(output[1], past_key_values)
        self.assertFalse(output[2])
        self.assertFalse(output[3])
        self.assertTrue(output[4])
        with self.assertRaises(ValueError):
            output = dummy_model.call(input_ids=input_ids, past_key_values=past_key_values, foo='bar')
        output = dummy_model.foo(pixel_values=pixel_values)
        tf.debugging.assert_equal(output[0], pixel_values)
        self.assertFalse(output[1])
        self.assertFalse(output[2])
        self.assertFalse(output[3])

    def test_xla_stable_softmax(self):
        if False:
            while True:
                i = 10
        large_penalty = -1000000000.0
        n_tokens = 10
        batch_size = 8

        def masked_softmax(x, boolean_mask):
            if False:
                return 10
            numerical_mask = (1.0 - tf.cast(boolean_mask, dtype=tf.float32)) * large_penalty
            masked_x = x + numerical_mask
            return stable_softmax(masked_x)
        xla_masked_softmax = tf.function(masked_softmax, jit_compile=True)
        xla_stable_softmax = tf.function(stable_softmax, jit_compile=True)
        x = tf.random.normal((batch_size, n_tokens))
        masked_tokens = random.randint(0, n_tokens)
        boolean_mask = tf.convert_to_tensor([[1] * (n_tokens - masked_tokens) + [0] * masked_tokens], dtype=tf.int32)
        numerical_mask = (1.0 - tf.cast(boolean_mask, dtype=tf.float32)) * large_penalty
        masked_x = x + numerical_mask
        xla_out = xla_stable_softmax(masked_x)
        out = stable_softmax(masked_x)
        assert tf.experimental.numpy.allclose(xla_out, out)
        unstable_out = tf.nn.softmax(masked_x)
        assert tf.experimental.numpy.allclose(unstable_out, out)
        xla_out = xla_masked_softmax(x, boolean_mask)
        out = masked_softmax(x, boolean_mask)
        assert tf.experimental.numpy.allclose(xla_out, out)

    def test_checkpoint_sharding_from_hub(self):
        if False:
            while True:
                i = 10
        model = TFBertModel.from_pretrained('ArthurZ/tiny-random-bert-sharded')
        ref_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        for (p1, p2) in zip(model.weights, ref_model.weights):
            assert np.allclose(p1.numpy(), p2.numpy())

    def test_sharded_checkpoint_with_prefix(self):
        if False:
            print('Hello World!')
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert', load_weight_prefix='a/b')
        sharded_model = TFBertModel.from_pretrained('ArthurZ/tiny-random-bert-sharded', load_weight_prefix='a/b')
        for (p1, p2) in zip(model.weights, sharded_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))
            self.assertTrue(p1.name.startswith('a/b/'))
            self.assertTrue(p2.name.startswith('a/b/'))

    def test_sharded_checkpoint_transfer(self):
        if False:
            print('Hello World!')
        TFBertForSequenceClassification.from_pretrained('ArthurZ/tiny-random-bert-sharded')

    @is_pt_tf_cross_test
    def test_checkpoint_sharding_local_from_pt(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_dir:
            _ = Repository(local_dir=tmp_dir, clone_from='hf-internal-testing/tiny-random-bert-sharded')
            model = TFBertModel.from_pretrained(tmp_dir, from_pt=True)
            ref_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
            for (p1, p2) in zip(model.weights, ref_model.weights):
                assert np.allclose(p1.numpy(), p2.numpy())

    @is_pt_tf_cross_test
    def test_checkpoint_loading_with_prefix_from_pt(self):
        if False:
            i = 10
            return i + 15
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert', from_pt=True, load_weight_prefix='a/b')
        ref_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert', from_pt=True)
        for (p1, p2) in zip(model.weights, ref_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))
            self.assertTrue(p1.name.startswith('a/b/'))

    @is_pt_tf_cross_test
    def test_checkpoint_sharding_hub_from_pt(self):
        if False:
            while True:
                i = 10
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert-sharded', from_pt=True)
        ref_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        for (p1, p2) in zip(model.weights, ref_model.weights):
            assert np.allclose(p1.numpy(), p2.numpy())

    def test_shard_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        model = tf.keras.Sequential([tf.keras.layers.Dense(200, use_bias=False), tf.keras.layers.Dense(200, use_bias=False), tf.keras.layers.Dense(100, use_bias=False), tf.keras.layers.Dense(50, use_bias=False)])
        inputs = tf.zeros((1, 100), dtype=tf.float32)
        model(inputs)
        weights = model.weights
        weights_dict = {w.name: w for w in weights}
        with self.subTest('No shard when max size is bigger than model size'):
            (shards, index) = tf_shard_checkpoint(weights)
            self.assertIsNone(index)
            self.assertDictEqual(shards, {TF2_WEIGHTS_NAME: weights})
        with self.subTest('Test sharding, no weights bigger than max size'):
            (shards, index) = tf_shard_checkpoint(weights, max_shard_size='300kB')
            self.assertDictEqual(index, {'metadata': {'total_size': 340000}, 'weight_map': {'dense/kernel:0': 'tf_model-00001-of-00002.h5', 'dense_1/kernel:0': 'tf_model-00001-of-00002.h5', 'dense_2/kernel:0': 'tf_model-00002-of-00002.h5', 'dense_3/kernel:0': 'tf_model-00002-of-00002.h5'}})
            shard1 = [weights_dict['dense/kernel:0'], weights_dict['dense_1/kernel:0']]
            shard2 = [weights_dict['dense_2/kernel:0'], weights_dict['dense_3/kernel:0']]
            self.assertDictEqual(shards, {'tf_model-00001-of-00002.h5': shard1, 'tf_model-00002-of-00002.h5': shard2})
        with self.subTest('Test sharding with weights bigger than max size'):
            (shards, index) = tf_shard_checkpoint(weights, max_shard_size='100kB')
            self.assertDictEqual(index, {'metadata': {'total_size': 340000}, 'weight_map': {'dense/kernel:0': 'tf_model-00001-of-00003.h5', 'dense_1/kernel:0': 'tf_model-00002-of-00003.h5', 'dense_2/kernel:0': 'tf_model-00003-of-00003.h5', 'dense_3/kernel:0': 'tf_model-00003-of-00003.h5'}})
            shard1 = [weights_dict['dense/kernel:0']]
            shard2 = [weights_dict['dense_1/kernel:0']]
            shard3 = [weights_dict['dense_2/kernel:0'], weights_dict['dense_3/kernel:0']]
            self.assertDictEqual(shards, {'tf_model-00001-of-00003.h5': shard1, 'tf_model-00002-of-00003.h5': shard2, 'tf_model-00003-of-00003.h5': shard3})

    @slow
    def test_special_layer_name_sharding(self):
        if False:
            print('Hello World!')
        retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', index_name='exact', use_dummy_dataset=True)
        model = TFRagModel.from_pretrained('facebook/rag-token-nq', retriever=retriever)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for max_size in ['150kB', '150kiB', '200kB', '200kiB']:
                model.save_pretrained(tmp_dir, max_shard_size=max_size)
                ref_model = TFRagModel.from_pretrained(tmp_dir, retriever=retriever)
                for (p1, p2) in zip(model.weights, ref_model.weights):
                    assert np.allclose(p1.numpy(), p2.numpy())

    def test_checkpoint_sharding_local(self):
        if False:
            while True:
                i = 10
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            for max_size in ['150kB', '150kiB', '200kB', '200kiB']:
                model.save_pretrained(tmp_dir, max_shard_size=max_size)
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith('.h5'):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)
                index_file = os.path.join(tmp_dir, TF2_WEIGHTS_INDEX_NAME)
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_NAME)))
                for (shard_file, size) in shard_to_size.items():
                    if max_size.endswith('kiB'):
                        max_size_int = int(max_size[:-3]) * 2 ** 10
                    else:
                        max_size_int = int(max_size[:-2]) * 10 ** 3
                    if size >= max_size_int + 50000:
                        with h5py.File(shard_file, 'r') as state_file:
                            self.assertEqual(len(state_file), 1)
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.loads(f.read())
                all_shards = set(index['weight_map'].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith('.h5')}
                self.assertSetEqual(all_shards, shards_found)
                new_model = TFBertModel.from_pretrained(tmp_dir)
                model.build()
                new_model.build()
                for (p1, p2) in zip(model.weights, new_model.weights):
                    self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @slow
    def test_save_pretrained_signatures(self):
        if False:
            return 10
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')

        @tf.function(input_signature=[[tf.TensorSpec([None, None], tf.int32, name='input_ids'), tf.TensorSpec([None, None], tf.int32, name='token_type_ids'), tf.TensorSpec([None, None], tf.int32, name='attention_mask')]])
        def serving_fn(input):
            if False:
                while True:
                    i = 10
            return model(input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, saved_model=True, signatures=None)
            model_loaded = tf.keras.models.load_model(f'{tmp_dir}/saved_model/1')
            self.assertTrue('serving_default' in list(model_loaded.signatures.keys()))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, saved_model=True, signatures={'custom_signature': serving_fn})
            model_loaded = tf.keras.models.load_model(f'{tmp_dir}/saved_model/1')
            self.assertTrue('custom_signature' in list(model_loaded.signatures.keys()))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, saved_model=True, signatures={'custom_signature_1': serving_fn, 'custom_signature_2': serving_fn})
            model_loaded = tf.keras.models.load_model(f'{tmp_dir}/saved_model/1')
            self.assertTrue('custom_signature_1' in list(model_loaded.signatures.keys()))
            self.assertTrue('custom_signature_2' in list(model_loaded.signatures.keys()))

    @require_safetensors
    def test_safetensors_save_and_load(self):
        if False:
            while True:
                i = 10
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, TF2_WEIGHTS_NAME)))
            new_model = TFBertModel.from_pretrained(tmp_dir)
            for (p1, p2) in zip(model.weights, new_model.weights):
                self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @is_pt_tf_cross_test
    def test_safetensors_save_and_load_pt_to_tf(self):
        if False:
            return 10
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        pt_model = BertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_model.save_pretrained(tmp_dir, safe_serialization=True)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            new_model = TFBertModel.from_pretrained(tmp_dir)
            for (p1, p2) in zip(model.weights, new_model.weights):
                self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_hub(self):
        if False:
            print('Hello World!')
        tf_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert')
        safetensors_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert-safetensors-tf')
        for (p1, p2) in zip(safetensors_model.weights, tf_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))
        safetensors_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-random-bert-safetensors')
        for (p1, p2) in zip(safetensors_model.weights, tf_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_tf_from_tf(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-tf-only')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = TFBertModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(model.weights, new_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    @is_pt_tf_cross_test
    def test_safetensors_tf_from_torch(self):
        if False:
            for i in range(10):
                print('nop')
        hub_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-tf-only')
        model = BertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-only')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = TFBertModel.from_pretrained(tmp_dir)
        for (p1, p2) in zip(hub_model.weights, new_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_tf_from_sharded_h5_with_sharded_safetensors_local(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = snapshot_download('hf-internal-testing/tiny-bert-tf-safetensors-h5-sharded', cache_dir=tmp_dir)
            TFBertModel.from_pretrained(path)

    @require_safetensors
    def test_safetensors_tf_from_sharded_h5_with_sharded_safetensors_hub(self):
        if False:
            for i in range(10):
                print('nop')
        TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-tf-safetensors-h5-sharded')

    @require_safetensors
    def test_safetensors_load_from_local(self):
        if False:
            i = 10
            return i + 15
        '\n        This test checks that we can load safetensors from a checkpoint that only has those on the Hub\n        '
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download('hf-internal-testing/tiny-bert-tf-only', cache_dir=tmp)
            tf_model = TFBertModel.from_pretrained(location)
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download('hf-internal-testing/tiny-bert-tf-safetensors-only', cache_dir=tmp)
            safetensors_model = TFBertModel.from_pretrained(location)
        for (p1, p2) in zip(tf_model.weights, safetensors_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_hub_from_safetensors_pt(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This test checks that we can load safetensors from a checkpoint that only has those on the Hub.\n        saved in the "pt" format.\n        '
        tf_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-h5')
        safetensors_model = TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-safetensors')
        for (p1, p2) in zip(tf_model.weights, safetensors_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_local_from_safetensors_pt(self):
        if False:
            while True:
                i = 10
        '\n        This test checks that we can load safetensors from a local checkpoint that only has those\n        saved in the "pt" format.\n        '
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download('hf-internal-testing/tiny-bert-h5', cache_dir=tmp)
            tf_model = TFBertModel.from_pretrained(location)
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download('hf-internal-testing/tiny-bert-pt-safetensors', cache_dir=tmp)
            safetensors_model = TFBertModel.from_pretrained(location)
        for (p1, p2) in zip(tf_model.weights, safetensors_model.weights):
            self.assertTrue(np.allclose(p1.numpy(), p2.numpy()))

    @require_safetensors
    def test_safetensors_load_from_hub_h5_before_safetensors(self):
        if False:
            while True:
                i = 10
        "\n        This test checks that we'll first download h5 weights before safetensors\n        The safetensors file on that repo is a pt safetensors and therefore cannot be loaded without PyTorch\n        "
        TFBertModel.from_pretrained('hf-internal-testing/tiny-bert-pt-safetensors-msgpack')

    @require_safetensors
    def test_safetensors_load_from_local_h5_before_safetensors(self):
        if False:
            while True:
                i = 10
        "\n        This test checks that we'll first download h5 weights before safetensors\n        The safetensors file on that repo is a pt safetensors and therefore cannot be loaded without PyTorch\n        "
        with tempfile.TemporaryDirectory() as tmp:
            location = snapshot_download('hf-internal-testing/tiny-bert-pt-safetensors-msgpack', cache_dir=tmp)
            TFBertModel.from_pretrained(location)

@require_tf
@is_staging_test
class TFModelPushToHubTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        try:
            delete_repo(token=cls._token, repo_id='test-model-tf')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='test-model-tf-callback')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='valid_org/test-model-tf-org')
        except HTTPError:
            pass

    def test_push_to_hub(self):
        if False:
            while True:
                i = 10
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        model = TFBertModel(config)
        model.build()
        logging.set_verbosity_info()
        logger = logging.get_logger('transformers.utils.hub')
        with CaptureLogger(logger) as cl:
            model.push_to_hub('test-model-tf', token=self._token)
        logging.set_verbosity_warning()
        self.assertIn('Uploading the following files to __DUMMY_TRANSFORMERS_USER__/test-model-tf', cl.out)
        new_model = TFBertModel.from_pretrained(f'{USER}/test-model-tf')
        models_equal = True
        for (p1, p2) in zip(model.weights, new_model.weights):
            if not tf.math.reduce_all(p1 == p2):
                models_equal = False
                break
        self.assertTrue(models_equal)
        delete_repo(token=self._token, repo_id='test-model-tf')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id='test-model-tf', push_to_hub=True, token=self._token)
        new_model = TFBertModel.from_pretrained(f'{USER}/test-model-tf')
        models_equal = True
        for (p1, p2) in zip(model.weights, new_model.weights):
            if not tf.math.reduce_all(p1 == p2):
                models_equal = False
                break
        self.assertTrue(models_equal)

    @is_pt_tf_cross_test
    def test_push_to_hub_callback(self):
        if False:
            for i in range(10):
                print('nop')
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        model = TFBertForMaskedLM(config)
        model.compile()
        with tempfile.TemporaryDirectory() as tmp_dir:
            push_to_hub_callback = PushToHubCallback(output_dir=tmp_dir, hub_model_id='test-model-tf-callback', hub_token=self._token)
            model.fit(model.dummy_inputs, model.dummy_inputs, epochs=1, callbacks=[push_to_hub_callback])
        new_model = TFBertForMaskedLM.from_pretrained(f'{USER}/test-model-tf-callback')
        models_equal = True
        for (p1, p2) in zip(model.weights, new_model.weights):
            if not tf.math.reduce_all(p1 == p2):
                models_equal = False
                break
        self.assertTrue(models_equal)
        tf_push_to_hub_params = dict(inspect.signature(TFPreTrainedModel.push_to_hub).parameters)
        tf_push_to_hub_params.pop('base_model_card_args')
        pt_push_to_hub_params = dict(inspect.signature(PreTrainedModel.push_to_hub).parameters)
        pt_push_to_hub_params.pop('deprecated_kwargs')
        self.assertDictEaual(tf_push_to_hub_params, pt_push_to_hub_params)

    def test_push_to_hub_in_organization(self):
        if False:
            print('Hello World!')
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        model = TFBertModel(config)
        model.build()
        model.push_to_hub('valid_org/test-model-tf-org', token=self._token)
        new_model = TFBertModel.from_pretrained('valid_org/test-model-tf-org')
        models_equal = True
        for (p1, p2) in zip(model.weights, new_model.weights):
            if not tf.math.reduce_all(p1 == p2):
                models_equal = False
                break
        self.assertTrue(models_equal)
        delete_repo(token=self._token, repo_id='valid_org/test-model-tf-org')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=self._token, repo_id='valid_org/test-model-tf-org')
        new_model = TFBertModel.from_pretrained('valid_org/test-model-tf-org')
        models_equal = True
        for (p1, p2) in zip(model.weights, new_model.weights):
            if not tf.math.reduce_all(p1 == p2):
                models_equal = False
                break
        self.assertTrue(models_equal)