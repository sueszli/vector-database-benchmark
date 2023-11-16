import json
import os
import pytest
import torch
from allennlp.common import cached_transformers
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from transformers import AutoConfig, AutoModel

class TestCachedTransformers(AllenNlpTestCase):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        cached_transformers._clear_caches()

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().teardown_method()
        cached_transformers._clear_caches()

    def test_get_missing_from_cache_local_files_only(self):
        if False:
            print('Hello World!')
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get('bert-base-uncased', True, cache_dir=self.TEST_DIR, local_files_only=True)

    def clear_test_dir(self):
        if False:
            i = 10
            return i + 15
        for f in os.listdir(str(self.TEST_DIR)):
            os.remove(str(self.TEST_DIR) + '/' + f)
        assert len(os.listdir(str(self.TEST_DIR))) == 0

    def test_from_pretrained_avoids_weights_download_if_override_weights(self):
        if False:
            print('Hello World!')
        config = AutoConfig.from_pretrained('epwalsh/bert-xsmall-dummy', cache_dir=self.TEST_DIR)
        transformer = AutoModel.from_config(AutoConfig.from_pretrained('epwalsh/bert-xsmall-dummy', cache_dir=self.TEST_DIR))
        transformer = AutoModel.from_config(config)
        self.clear_test_dir()
        save_weights_path = str(self.TEST_DIR / 'bert_weights.pth')
        torch.save(transformer.state_dict(), save_weights_path)
        override_transformer = cached_transformers.get('epwalsh/bert-xsmall-dummy', False, override_weights_file=save_weights_path, cache_dir=self.TEST_DIR)
        json_fnames = [fname for fname in os.listdir(str(self.TEST_DIR)) if fname.endswith('.json')]
        assert len(json_fnames) == 1
        json_data = json.load(open(str(self.TEST_DIR / json_fnames[0])))
        assert json_data['url'] == 'https://huggingface.co/epwalsh/bert-xsmall-dummy/resolve/main/config.json'
        resource_id = os.path.splitext(json_fnames[0])[0]
        assert set(os.listdir(str(self.TEST_DIR))) == set([json_fnames[0], resource_id, resource_id + '.lock', 'bert_weights.pth'])
        for (p1, p2) in zip(transformer.parameters(), override_transformer.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

    def test_reinit_modules_no_op(self):
        if False:
            for i in range(10):
                print('nop')
        preinit_weights = torch.cat([layer.attention.output.dense.weight for layer in cached_transformers.get('bert-base-cased', True).encoder.layer])
        postinit_weights = torch.cat([layer.attention.output.dense.weight for layer in cached_transformers.get('bert-base-cased', True).encoder.layer])
        assert torch.equal(postinit_weights, preinit_weights)

    def test_reinit_modules_with_layer_indices(self):
        if False:
            print('Hello World!')
        preinit_weights = torch.cat([layer.attention.output.dense.weight for layer in cached_transformers.get('bert-base-cased', True).encoder.layer])
        postinit_weights = torch.cat([layer.attention.output.dense.weight for layer in cached_transformers.get('bert-base-cased', True, reinit_modules=2).encoder.layer])
        assert torch.equal(postinit_weights[:10], preinit_weights[:10])
        assert not torch.equal(postinit_weights[10:], preinit_weights[10:])
        postinit_weights = torch.cat([layer.attention.output.dense.weight for layer in cached_transformers.get('bert-base-cased', True, reinit_modules=(10, 11)).encoder.layer])
        assert torch.equal(postinit_weights[:10], preinit_weights[:10])
        assert not torch.equal(postinit_weights[10:], preinit_weights[10:])
        with pytest.raises(ValueError):
            _ = cached_transformers.get('bert-base-cased', True, reinit_modules=1000)
        with pytest.raises(ValueError):
            _ = cached_transformers.get('bert-base-cased', True, reinit_modules=(1, 1000))
        with pytest.raises(ValueError):
            _ = cached_transformers.get('bert-base-cased', True, reinit_modules=(1, 'attentions'))
        with pytest.raises(ConfigurationError):
            _ = cached_transformers.get('sshleifer/tiny-gpt2', True, reinit_modules=1)
        with pytest.raises(ConfigurationError):
            _ = cached_transformers.get('sshleifer/tiny-gpt2', True, reinit_modules=(1, 2))

    def test_reinit_modules_with_regex_strings(self):
        if False:
            while True:
                i = 10
        reinit_module = 'wpe'
        preinit_weights = list(cached_transformers.get('sshleifer/tiny-gpt2', True).get_submodule(reinit_module).parameters())
        postinit_weights = list(cached_transformers.get('sshleifer/tiny-gpt2', True, reinit_modules=(reinit_module,)).get_submodule(reinit_module).parameters())
        assert all((not torch.equal(pre, post) for (pre, post) in zip(preinit_weights, postinit_weights)))

    def test_from_pretrained_no_load_weights(self):
        if False:
            for i in range(10):
                print('nop')
        _ = cached_transformers.get('epwalsh/bert-xsmall-dummy', False, load_weights=False, cache_dir=self.TEST_DIR)
        json_fnames = [fname for fname in os.listdir(str(self.TEST_DIR)) if fname.endswith('.json')]
        assert len(json_fnames) == 1
        json_data = json.load(open(str(self.TEST_DIR / json_fnames[0])))
        assert json_data['url'] == 'https://huggingface.co/epwalsh/bert-xsmall-dummy/resolve/main/config.json'
        resource_id = os.path.splitext(json_fnames[0])[0]
        assert set(os.listdir(str(self.TEST_DIR))) == set([json_fnames[0], resource_id, resource_id + '.lock'])

    def test_from_pretrained_no_load_weights_local_config(self):
        if False:
            i = 10
            return i + 15
        config = AutoConfig.from_pretrained('epwalsh/bert-xsmall-dummy', cache_dir=self.TEST_DIR)
        self.clear_test_dir()
        local_config_path = str(self.TEST_DIR / 'local_config.json')
        config.to_json_file(local_config_path, use_diff=False)
        _ = cached_transformers.get(local_config_path, False, load_weights=False, cache_dir=self.TEST_DIR)
        assert os.listdir(str(self.TEST_DIR)) == ['local_config.json']

    def test_get_tokenizer_missing_from_cache_local_files_only(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get_tokenizer('bert-base-uncased', cache_dir=self.TEST_DIR, local_files_only=True)