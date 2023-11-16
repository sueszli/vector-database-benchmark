import os
import re
import tempfile
import unittest
from pathlib import Path
import transformers
from transformers.commands.add_new_model_like import ModelPatterns, _re_class_func, add_content_to_file, add_content_to_text, clean_frameworks_in_init, duplicate_doc_file, duplicate_module, filter_framework_files, find_base_model_checkpoint, get_model_files, get_module_from_file, parse_module_content, replace_model_patterns, retrieve_info_for_model, retrieve_model_classes, simplify_replacements
from transformers.testing_utils import require_flax, require_tf, require_torch
BERT_MODEL_FILES = {'src/transformers/models/bert/__init__.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/tokenization_bert.py', 'src/transformers/models/bert/tokenization_bert_fast.py', 'src/transformers/models/bert/tokenization_bert_tf.py', 'src/transformers/models/bert/modeling_bert.py', 'src/transformers/models/bert/modeling_flax_bert.py', 'src/transformers/models/bert/modeling_tf_bert.py', 'src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py', 'src/transformers/models/bert/convert_bert_original_tf2_checkpoint_to_pytorch.py', 'src/transformers/models/bert/convert_bert_pytorch_checkpoint_to_original_tf.py', 'src/transformers/models/bert/convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py'}
VIT_MODEL_FILES = {'src/transformers/models/vit/__init__.py', 'src/transformers/models/vit/configuration_vit.py', 'src/transformers/models/vit/convert_dino_to_pytorch.py', 'src/transformers/models/vit/convert_vit_timm_to_pytorch.py', 'src/transformers/models/vit/feature_extraction_vit.py', 'src/transformers/models/vit/image_processing_vit.py', 'src/transformers/models/vit/modeling_vit.py', 'src/transformers/models/vit/modeling_tf_vit.py', 'src/transformers/models/vit/modeling_flax_vit.py'}
WAV2VEC2_MODEL_FILES = {'src/transformers/models/wav2vec2/__init__.py', 'src/transformers/models/wav2vec2/configuration_wav2vec2.py', 'src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py', 'src/transformers/models/wav2vec2/convert_wav2vec2_original_s3prl_checkpoint_to_pytorch.py', 'src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py', 'src/transformers/models/wav2vec2/modeling_wav2vec2.py', 'src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py', 'src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py', 'src/transformers/models/wav2vec2/processing_wav2vec2.py', 'src/transformers/models/wav2vec2/tokenization_wav2vec2.py'}
REPO_PATH = Path(transformers.__path__[0]).parent.parent

@require_torch
@require_tf
@require_flax
class TestAddNewModelLike(unittest.TestCase):

    def init_file(self, file_name, content):
        if False:
            print('Hello World!')
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)

    def check_result(self, file_name, expected_result):
        if False:
            i = 10
            return i + 15
        with open(file_name, 'r', encoding='utf-8') as f:
            result = f.read()
            self.assertEqual(result, expected_result)

    def test_re_class_func(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(_re_class_func.search('def my_function(x, y):').groups()[0], 'my_function')
        self.assertEqual(_re_class_func.search('class MyClass:').groups()[0], 'MyClass')
        self.assertEqual(_re_class_func.search('class MyClass(SuperClass):').groups()[0], 'MyClass')

    def test_model_patterns_defaults(self):
        if False:
            return 10
        model_patterns = ModelPatterns('GPT-New new', 'huggingface/gpt-new-base')
        self.assertEqual(model_patterns.model_type, 'gpt-new-new')
        self.assertEqual(model_patterns.model_lower_cased, 'gpt_new_new')
        self.assertEqual(model_patterns.model_camel_cased, 'GPTNewNew')
        self.assertEqual(model_patterns.model_upper_cased, 'GPT_NEW_NEW')
        self.assertEqual(model_patterns.config_class, 'GPTNewNewConfig')
        self.assertIsNone(model_patterns.tokenizer_class)
        self.assertIsNone(model_patterns.feature_extractor_class)
        self.assertIsNone(model_patterns.processor_class)

    def test_parse_module_content(self):
        if False:
            print('Hello World!')
        test_code = 'SOME_CONSTANT = a constant\n\nCONSTANT_DEFINED_ON_SEVERAL_LINES = [\n    first_item,\n    second_item\n]\n\ndef function(args):\n    some code\n\n# Copied from transformers.some_module\nclass SomeClass:\n    some code\n'
        expected_parts = ['SOME_CONSTANT = a constant\n', 'CONSTANT_DEFINED_ON_SEVERAL_LINES = [\n    first_item,\n    second_item\n]', '', 'def function(args):\n    some code\n', '# Copied from transformers.some_module\nclass SomeClass:\n    some code\n']
        self.assertEqual(parse_module_content(test_code), expected_parts)

    def test_add_content_to_text(self):
        if False:
            return 10
        test_text = 'all_configs = {\n    "gpt": "GPTConfig",\n    "bert": "BertConfig",\n    "t5": "T5Config",\n}'
        expected = 'all_configs = {\n    "gpt": "GPTConfig",\n    "gpt2": "GPT2Config",\n    "bert": "BertConfig",\n    "t5": "T5Config",\n}'
        line = '    "gpt2": "GPT2Config",'
        self.assertEqual(add_content_to_text(test_text, line, add_before='bert'), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_before='bert', exact_match=True), test_text)
        self.assertEqual(add_content_to_text(test_text, line, add_before='    "bert": "BertConfig",', exact_match=True), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_before=re.compile('^\\s*"bert":')), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_after='gpt'), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_after='gpt', exact_match=True), test_text)
        self.assertEqual(add_content_to_text(test_text, line, add_after='    "gpt": "GPTConfig",', exact_match=True), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_after=re.compile('^\\s*"gpt":')), expected)

    def test_add_content_to_file(self):
        if False:
            while True:
                i = 10
        test_text = 'all_configs = {\n    "gpt": "GPTConfig",\n    "bert": "BertConfig",\n    "t5": "T5Config",\n}'
        expected = 'all_configs = {\n    "gpt": "GPTConfig",\n    "gpt2": "GPT2Config",\n    "bert": "BertConfig",\n    "t5": "T5Config",\n}'
        line = '    "gpt2": "GPT2Config",'
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, 'code.py')
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before='bert')
            self.check_result(file_name, expected)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before='bert', exact_match=True)
            self.check_result(file_name, test_text)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before='    "bert": "BertConfig",', exact_match=True)
            self.check_result(file_name, expected)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before=re.compile('^\\s*"bert":'))
            self.check_result(file_name, expected)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after='gpt')
            self.check_result(file_name, expected)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after='gpt', exact_match=True)
            self.check_result(file_name, test_text)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after='    "gpt": "GPTConfig",', exact_match=True)
            self.check_result(file_name, expected)
            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after=re.compile('^\\s*"gpt":'))
            self.check_result(file_name, expected)

    def test_simplify_replacements(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(simplify_replacements([('Bert', 'NewBert')]), [('Bert', 'NewBert')])
        self.assertEqual(simplify_replacements([('Bert', 'NewBert'), ('bert', 'new-bert')]), [('Bert', 'NewBert'), ('bert', 'new-bert')])
        self.assertEqual(simplify_replacements([('BertConfig', 'NewBertConfig'), ('Bert', 'NewBert'), ('bert', 'new-bert')]), [('Bert', 'NewBert'), ('bert', 'new-bert')])

    def test_replace_model_patterns(self):
        if False:
            return 10
        bert_model_patterns = ModelPatterns('Bert', 'bert-base-cased')
        new_bert_model_patterns = ModelPatterns('New Bert', 'huggingface/bert-new-base')
        bert_test = 'class TFBertPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = BertConfig\n    load_tf_weights = load_tf_weights_in_bert\n    base_model_prefix = "bert"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n    model_type = "bert"\n\nBERT_CONSTANT = "value"\n'
        bert_expected = 'class TFNewBertPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = NewBertConfig\n    load_tf_weights = load_tf_weights_in_new_bert\n    base_model_prefix = "new_bert"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n    model_type = "new-bert"\n\nNEW_BERT_CONSTANT = "value"\n'
        (bert_converted, replacements) = replace_model_patterns(bert_test, bert_model_patterns, new_bert_model_patterns)
        self.assertEqual(bert_converted, bert_expected)
        self.assertEqual(replacements, '')
        bert_test = bert_test.replace('    model_type = "bert"\n', '')
        bert_expected = bert_expected.replace('    model_type = "new-bert"\n', '')
        (bert_converted, replacements) = replace_model_patterns(bert_test, bert_model_patterns, new_bert_model_patterns)
        self.assertEqual(bert_converted, bert_expected)
        self.assertEqual(replacements, 'BERT->NEW_BERT,Bert->NewBert,bert->new_bert')
        gpt_model_patterns = ModelPatterns('GPT2', 'gpt2')
        new_gpt_model_patterns = ModelPatterns('GPT-New new', 'huggingface/gpt-new-base')
        gpt_test = 'class GPT2PreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = GPT2Config\n    load_tf_weights = load_tf_weights_in_gpt2\n    base_model_prefix = "transformer"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n\nGPT2_CONSTANT = "value"\n'
        gpt_expected = 'class GPTNewNewPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = GPTNewNewConfig\n    load_tf_weights = load_tf_weights_in_gpt_new_new\n    base_model_prefix = "transformer"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n\nGPT_NEW_NEW_CONSTANT = "value"\n'
        (gpt_converted, replacements) = replace_model_patterns(gpt_test, gpt_model_patterns, new_gpt_model_patterns)
        self.assertEqual(gpt_converted, gpt_expected)
        self.assertEqual(replacements, '')
        roberta_model_patterns = ModelPatterns('RoBERTa', 'roberta-base', model_camel_cased='Roberta')
        new_roberta_model_patterns = ModelPatterns('RoBERTa-New', 'huggingface/roberta-new-base', model_camel_cased='RobertaNew')
        roberta_test = '# Copied from transformers.models.bert.BertModel with Bert->Roberta\nclass RobertaModel(RobertaPreTrainedModel):\n    """ The base RoBERTa model. """\n    checkpoint = roberta-base\n    base_model_prefix = "roberta"\n        '
        roberta_expected = '# Copied from transformers.models.bert.BertModel with Bert->RobertaNew\nclass RobertaNewModel(RobertaNewPreTrainedModel):\n    """ The base RoBERTa-New model. """\n    checkpoint = huggingface/roberta-new-base\n    base_model_prefix = "roberta_new"\n        '
        (roberta_converted, replacements) = replace_model_patterns(roberta_test, roberta_model_patterns, new_roberta_model_patterns)
        self.assertEqual(roberta_converted, roberta_expected)

    def test_get_module_from_file(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(get_module_from_file('/git/transformers/src/transformers/models/bert/modeling_tf_bert.py'), 'transformers.models.bert.modeling_tf_bert')
        self.assertEqual(get_module_from_file('/transformers/models/gpt2/modeling_gpt2.py'), 'transformers.models.gpt2.modeling_gpt2')
        with self.assertRaises(ValueError):
            get_module_from_file('/models/gpt2/modeling_gpt2.py')

    def test_duplicate_module(self):
        if False:
            i = 10
            return i + 15
        bert_model_patterns = ModelPatterns('Bert', 'bert-base-cased')
        new_bert_model_patterns = ModelPatterns('New Bert', 'huggingface/bert-new-base')
        bert_test = 'class TFBertPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = BertConfig\n    load_tf_weights = load_tf_weights_in_bert\n    base_model_prefix = "bert"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n\nBERT_CONSTANT = "value"\n'
        bert_expected = 'class TFNewBertPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = NewBertConfig\n    load_tf_weights = load_tf_weights_in_new_bert\n    base_model_prefix = "new_bert"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n\nNEW_BERT_CONSTANT = "value"\n'
        bert_expected_with_copied_from = '# Copied from transformers.bert_module.TFBertPreTrainedModel with Bert->NewBert,bert->new_bert\n' + bert_expected
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = os.path.join(tmp_dir, 'transformers')
            os.makedirs(work_dir)
            file_name = os.path.join(work_dir, 'bert_module.py')
            dest_file_name = os.path.join(work_dir, 'new_bert_module.py')
            self.init_file(file_name, bert_test)
            duplicate_module(file_name, bert_model_patterns, new_bert_model_patterns)
            self.check_result(dest_file_name, bert_expected_with_copied_from)
            self.init_file(file_name, bert_test)
            duplicate_module(file_name, bert_model_patterns, new_bert_model_patterns, add_copied_from=False)
            self.check_result(dest_file_name, bert_expected)

    def test_duplicate_module_with_copied_from(self):
        if False:
            i = 10
            return i + 15
        bert_model_patterns = ModelPatterns('Bert', 'bert-base-cased')
        new_bert_model_patterns = ModelPatterns('New Bert', 'huggingface/bert-new-base')
        bert_test = '# Copied from transformers.models.xxx.XxxModel with Xxx->Bert\nclass TFBertPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = BertConfig\n    load_tf_weights = load_tf_weights_in_bert\n    base_model_prefix = "bert"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n\nBERT_CONSTANT = "value"\n'
        bert_expected = '# Copied from transformers.models.xxx.XxxModel with Xxx->NewBert\nclass TFNewBertPreTrainedModel(PreTrainedModel):\n    """\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    """\n\n    config_class = NewBertConfig\n    load_tf_weights = load_tf_weights_in_new_bert\n    base_model_prefix = "new_bert"\n    is_parallelizable = True\n    supports_gradient_checkpointing = True\n\nNEW_BERT_CONSTANT = "value"\n'
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = os.path.join(tmp_dir, 'transformers')
            os.makedirs(work_dir)
            file_name = os.path.join(work_dir, 'bert_module.py')
            dest_file_name = os.path.join(work_dir, 'new_bert_module.py')
            self.init_file(file_name, bert_test)
            duplicate_module(file_name, bert_model_patterns, new_bert_model_patterns)
            self.check_result(dest_file_name, bert_expected)
            self.init_file(file_name, bert_test)
            duplicate_module(file_name, bert_model_patterns, new_bert_model_patterns, add_copied_from=False)
            self.check_result(dest_file_name, bert_expected)

    def test_filter_framework_files(self):
        if False:
            return 10
        files = ['modeling_bert.py', 'modeling_tf_bert.py', 'modeling_flax_bert.py', 'configuration_bert.py']
        self.assertEqual(filter_framework_files(files), files)
        self.assertEqual(set(filter_framework_files(files, ['pt', 'tf', 'flax'])), set(files))
        self.assertEqual(set(filter_framework_files(files, ['pt'])), {'modeling_bert.py', 'configuration_bert.py'})
        self.assertEqual(set(filter_framework_files(files, ['tf'])), {'modeling_tf_bert.py', 'configuration_bert.py'})
        self.assertEqual(set(filter_framework_files(files, ['flax'])), {'modeling_flax_bert.py', 'configuration_bert.py'})
        self.assertEqual(set(filter_framework_files(files, ['pt', 'tf'])), {'modeling_tf_bert.py', 'modeling_bert.py', 'configuration_bert.py'})
        self.assertEqual(set(filter_framework_files(files, ['tf', 'flax'])), {'modeling_tf_bert.py', 'modeling_flax_bert.py', 'configuration_bert.py'})
        self.assertEqual(set(filter_framework_files(files, ['pt', 'flax'])), {'modeling_bert.py', 'modeling_flax_bert.py', 'configuration_bert.py'})

    def test_get_model_files(self):
        if False:
            for i in range(10):
                print('nop')
        bert_files = get_model_files('bert')
        doc_file = str(Path(bert_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/bert.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files['model_files']}
        self.assertEqual(model_files, BERT_MODEL_FILES)
        self.assertEqual(bert_files['module_name'], 'bert')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files['test_files']}
        bert_test_files = {'tests/models/bert/test_tokenization_bert.py', 'tests/models/bert/test_modeling_bert.py', 'tests/models/bert/test_modeling_tf_bert.py', 'tests/models/bert/test_modeling_flax_bert.py'}
        self.assertEqual(test_files, bert_test_files)
        vit_files = get_model_files('vit')
        doc_file = str(Path(vit_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/vit.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in vit_files['model_files']}
        self.assertEqual(model_files, VIT_MODEL_FILES)
        self.assertEqual(vit_files['module_name'], 'vit')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in vit_files['test_files']}
        vit_test_files = {'tests/models/vit/test_image_processing_vit.py', 'tests/models/vit/test_modeling_vit.py', 'tests/models/vit/test_modeling_tf_vit.py', 'tests/models/vit/test_modeling_flax_vit.py'}
        self.assertEqual(test_files, vit_test_files)
        wav2vec2_files = get_model_files('wav2vec2')
        doc_file = str(Path(wav2vec2_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/wav2vec2.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in wav2vec2_files['model_files']}
        self.assertEqual(model_files, WAV2VEC2_MODEL_FILES)
        self.assertEqual(wav2vec2_files['module_name'], 'wav2vec2')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in wav2vec2_files['test_files']}
        wav2vec2_test_files = {'tests/models/wav2vec2/test_feature_extraction_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_tf_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_flax_wav2vec2.py', 'tests/models/wav2vec2/test_processor_wav2vec2.py', 'tests/models/wav2vec2/test_tokenization_wav2vec2.py'}
        self.assertEqual(test_files, wav2vec2_test_files)

    def test_get_model_files_only_pt(self):
        if False:
            return 10
        bert_files = get_model_files('bert', frameworks=['pt'])
        doc_file = str(Path(bert_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/bert.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files['model_files']}
        bert_model_files = BERT_MODEL_FILES - {'src/transformers/models/bert/modeling_tf_bert.py', 'src/transformers/models/bert/modeling_flax_bert.py'}
        self.assertEqual(model_files, bert_model_files)
        self.assertEqual(bert_files['module_name'], 'bert')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files['test_files']}
        bert_test_files = {'tests/models/bert/test_tokenization_bert.py', 'tests/models/bert/test_modeling_bert.py'}
        self.assertEqual(test_files, bert_test_files)
        vit_files = get_model_files('vit', frameworks=['pt'])
        doc_file = str(Path(vit_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/vit.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in vit_files['model_files']}
        vit_model_files = VIT_MODEL_FILES - {'src/transformers/models/vit/modeling_tf_vit.py', 'src/transformers/models/vit/modeling_flax_vit.py'}
        self.assertEqual(model_files, vit_model_files)
        self.assertEqual(vit_files['module_name'], 'vit')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in vit_files['test_files']}
        vit_test_files = {'tests/models/vit/test_image_processing_vit.py', 'tests/models/vit/test_modeling_vit.py'}
        self.assertEqual(test_files, vit_test_files)
        wav2vec2_files = get_model_files('wav2vec2', frameworks=['pt'])
        doc_file = str(Path(wav2vec2_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/wav2vec2.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in wav2vec2_files['model_files']}
        wav2vec2_model_files = WAV2VEC2_MODEL_FILES - {'src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py', 'src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py'}
        self.assertEqual(model_files, wav2vec2_model_files)
        self.assertEqual(wav2vec2_files['module_name'], 'wav2vec2')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in wav2vec2_files['test_files']}
        wav2vec2_test_files = {'tests/models/wav2vec2/test_feature_extraction_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_wav2vec2.py', 'tests/models/wav2vec2/test_processor_wav2vec2.py', 'tests/models/wav2vec2/test_tokenization_wav2vec2.py'}
        self.assertEqual(test_files, wav2vec2_test_files)

    def test_get_model_files_tf_and_flax(self):
        if False:
            for i in range(10):
                print('nop')
        bert_files = get_model_files('bert', frameworks=['tf', 'flax'])
        doc_file = str(Path(bert_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/bert.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files['model_files']}
        bert_model_files = BERT_MODEL_FILES - {'src/transformers/models/bert/modeling_bert.py'}
        self.assertEqual(model_files, bert_model_files)
        self.assertEqual(bert_files['module_name'], 'bert')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files['test_files']}
        bert_test_files = {'tests/models/bert/test_tokenization_bert.py', 'tests/models/bert/test_modeling_tf_bert.py', 'tests/models/bert/test_modeling_flax_bert.py'}
        self.assertEqual(test_files, bert_test_files)
        vit_files = get_model_files('vit', frameworks=['tf', 'flax'])
        doc_file = str(Path(vit_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/vit.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in vit_files['model_files']}
        vit_model_files = VIT_MODEL_FILES - {'src/transformers/models/vit/modeling_vit.py'}
        self.assertEqual(model_files, vit_model_files)
        self.assertEqual(vit_files['module_name'], 'vit')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in vit_files['test_files']}
        vit_test_files = {'tests/models/vit/test_image_processing_vit.py', 'tests/models/vit/test_modeling_tf_vit.py', 'tests/models/vit/test_modeling_flax_vit.py'}
        self.assertEqual(test_files, vit_test_files)
        wav2vec2_files = get_model_files('wav2vec2', frameworks=['tf', 'flax'])
        doc_file = str(Path(wav2vec2_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/wav2vec2.md')
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in wav2vec2_files['model_files']}
        wav2vec2_model_files = WAV2VEC2_MODEL_FILES - {'src/transformers/models/wav2vec2/modeling_wav2vec2.py'}
        self.assertEqual(model_files, wav2vec2_model_files)
        self.assertEqual(wav2vec2_files['module_name'], 'wav2vec2')
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in wav2vec2_files['test_files']}
        wav2vec2_test_files = {'tests/models/wav2vec2/test_feature_extraction_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_tf_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_flax_wav2vec2.py', 'tests/models/wav2vec2/test_processor_wav2vec2.py', 'tests/models/wav2vec2/test_tokenization_wav2vec2.py'}
        self.assertEqual(test_files, wav2vec2_test_files)

    def test_find_base_model_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(find_base_model_checkpoint('bert'), 'bert-base-uncased')
        self.assertEqual(find_base_model_checkpoint('gpt2'), 'gpt2')

    def test_retrieve_model_classes(self):
        if False:
            print('Hello World!')
        gpt_classes = {k: set(v) for (k, v) in retrieve_model_classes('gpt2').items()}
        expected_gpt_classes = {'pt': {'GPT2ForTokenClassification', 'GPT2Model', 'GPT2LMHeadModel', 'GPT2ForSequenceClassification'}, 'tf': {'TFGPT2Model', 'TFGPT2ForSequenceClassification', 'TFGPT2LMHeadModel'}, 'flax': {'FlaxGPT2Model', 'FlaxGPT2LMHeadModel'}}
        self.assertEqual(gpt_classes, expected_gpt_classes)
        del expected_gpt_classes['flax']
        gpt_classes = {k: set(v) for (k, v) in retrieve_model_classes('gpt2', frameworks=['pt', 'tf']).items()}
        self.assertEqual(gpt_classes, expected_gpt_classes)
        del expected_gpt_classes['pt']
        gpt_classes = {k: set(v) for (k, v) in retrieve_model_classes('gpt2', frameworks=['tf']).items()}
        self.assertEqual(gpt_classes, expected_gpt_classes)

    def test_retrieve_info_for_model_with_bert(self):
        if False:
            while True:
                i = 10
        bert_info = retrieve_info_for_model('bert')
        bert_classes = ['BertForTokenClassification', 'BertForQuestionAnswering', 'BertForNextSentencePrediction', 'BertForSequenceClassification', 'BertForMaskedLM', 'BertForMultipleChoice', 'BertModel', 'BertForPreTraining', 'BertLMHeadModel']
        expected_model_classes = {'pt': set(bert_classes), 'tf': {f'TF{m}' for m in bert_classes}, 'flax': {f'Flax{m}' for m in bert_classes[:-1] + ['BertForCausalLM']}}
        self.assertEqual(set(bert_info['frameworks']), {'pt', 'tf', 'flax'})
        model_classes = {k: set(v) for (k, v) in bert_info['model_classes'].items()}
        self.assertEqual(model_classes, expected_model_classes)
        all_bert_files = bert_info['model_files']
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files['model_files']}
        self.assertEqual(model_files, BERT_MODEL_FILES)
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files['test_files']}
        bert_test_files = {'tests/models/bert/test_tokenization_bert.py', 'tests/models/bert/test_modeling_bert.py', 'tests/models/bert/test_modeling_tf_bert.py', 'tests/models/bert/test_modeling_flax_bert.py'}
        self.assertEqual(test_files, bert_test_files)
        doc_file = str(Path(all_bert_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/bert.md')
        self.assertEqual(all_bert_files['module_name'], 'bert')
        bert_model_patterns = bert_info['model_patterns']
        self.assertEqual(bert_model_patterns.model_name, 'BERT')
        self.assertEqual(bert_model_patterns.checkpoint, 'bert-base-uncased')
        self.assertEqual(bert_model_patterns.model_type, 'bert')
        self.assertEqual(bert_model_patterns.model_lower_cased, 'bert')
        self.assertEqual(bert_model_patterns.model_camel_cased, 'Bert')
        self.assertEqual(bert_model_patterns.model_upper_cased, 'BERT')
        self.assertEqual(bert_model_patterns.config_class, 'BertConfig')
        self.assertEqual(bert_model_patterns.tokenizer_class, 'BertTokenizer')
        self.assertIsNone(bert_model_patterns.feature_extractor_class)
        self.assertIsNone(bert_model_patterns.processor_class)

    def test_retrieve_info_for_model_pt_tf_with_bert(self):
        if False:
            print('Hello World!')
        bert_info = retrieve_info_for_model('bert', frameworks=['pt', 'tf'])
        bert_classes = ['BertForTokenClassification', 'BertForQuestionAnswering', 'BertForNextSentencePrediction', 'BertForSequenceClassification', 'BertForMaskedLM', 'BertForMultipleChoice', 'BertModel', 'BertForPreTraining', 'BertLMHeadModel']
        expected_model_classes = {'pt': set(bert_classes), 'tf': {f'TF{m}' for m in bert_classes}}
        self.assertEqual(set(bert_info['frameworks']), {'pt', 'tf'})
        model_classes = {k: set(v) for (k, v) in bert_info['model_classes'].items()}
        self.assertEqual(model_classes, expected_model_classes)
        all_bert_files = bert_info['model_files']
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files['model_files']}
        bert_model_files = BERT_MODEL_FILES - {'src/transformers/models/bert/modeling_flax_bert.py'}
        self.assertEqual(model_files, bert_model_files)
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files['test_files']}
        bert_test_files = {'tests/models/bert/test_tokenization_bert.py', 'tests/models/bert/test_modeling_bert.py', 'tests/models/bert/test_modeling_tf_bert.py'}
        self.assertEqual(test_files, bert_test_files)
        doc_file = str(Path(all_bert_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/bert.md')
        self.assertEqual(all_bert_files['module_name'], 'bert')
        bert_model_patterns = bert_info['model_patterns']
        self.assertEqual(bert_model_patterns.model_name, 'BERT')
        self.assertEqual(bert_model_patterns.checkpoint, 'bert-base-uncased')
        self.assertEqual(bert_model_patterns.model_type, 'bert')
        self.assertEqual(bert_model_patterns.model_lower_cased, 'bert')
        self.assertEqual(bert_model_patterns.model_camel_cased, 'Bert')
        self.assertEqual(bert_model_patterns.model_upper_cased, 'BERT')
        self.assertEqual(bert_model_patterns.config_class, 'BertConfig')
        self.assertEqual(bert_model_patterns.tokenizer_class, 'BertTokenizer')
        self.assertIsNone(bert_model_patterns.feature_extractor_class)
        self.assertIsNone(bert_model_patterns.processor_class)

    def test_retrieve_info_for_model_with_vit(self):
        if False:
            print('Hello World!')
        vit_info = retrieve_info_for_model('vit')
        vit_classes = ['ViTForImageClassification', 'ViTModel']
        pt_only_classes = ['ViTForMaskedImageModeling']
        expected_model_classes = {'pt': set(vit_classes + pt_only_classes), 'tf': {f'TF{m}' for m in vit_classes}, 'flax': {f'Flax{m}' for m in vit_classes}}
        self.assertEqual(set(vit_info['frameworks']), {'pt', 'tf', 'flax'})
        model_classes = {k: set(v) for (k, v) in vit_info['model_classes'].items()}
        self.assertEqual(model_classes, expected_model_classes)
        all_vit_files = vit_info['model_files']
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_vit_files['model_files']}
        self.assertEqual(model_files, VIT_MODEL_FILES)
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_vit_files['test_files']}
        vit_test_files = {'tests/models/vit/test_image_processing_vit.py', 'tests/models/vit/test_modeling_vit.py', 'tests/models/vit/test_modeling_tf_vit.py', 'tests/models/vit/test_modeling_flax_vit.py'}
        self.assertEqual(test_files, vit_test_files)
        doc_file = str(Path(all_vit_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/vit.md')
        self.assertEqual(all_vit_files['module_name'], 'vit')
        vit_model_patterns = vit_info['model_patterns']
        self.assertEqual(vit_model_patterns.model_name, 'ViT')
        self.assertEqual(vit_model_patterns.checkpoint, 'google/vit-base-patch16-224-in21k')
        self.assertEqual(vit_model_patterns.model_type, 'vit')
        self.assertEqual(vit_model_patterns.model_lower_cased, 'vit')
        self.assertEqual(vit_model_patterns.model_camel_cased, 'ViT')
        self.assertEqual(vit_model_patterns.model_upper_cased, 'VIT')
        self.assertEqual(vit_model_patterns.config_class, 'ViTConfig')
        self.assertEqual(vit_model_patterns.feature_extractor_class, 'ViTFeatureExtractor')
        self.assertEqual(vit_model_patterns.image_processor_class, 'ViTImageProcessor')
        self.assertIsNone(vit_model_patterns.tokenizer_class)
        self.assertIsNone(vit_model_patterns.processor_class)

    def test_retrieve_info_for_model_with_wav2vec2(self):
        if False:
            return 10
        wav2vec2_info = retrieve_info_for_model('wav2vec2')
        wav2vec2_classes = ['Wav2Vec2Model', 'Wav2Vec2ForPreTraining', 'Wav2Vec2ForAudioFrameClassification', 'Wav2Vec2ForCTC', 'Wav2Vec2ForMaskedLM', 'Wav2Vec2ForSequenceClassification', 'Wav2Vec2ForXVector']
        expected_model_classes = {'pt': set(wav2vec2_classes), 'tf': {f'TF{m}' for m in wav2vec2_classes[:1]}, 'flax': {f'Flax{m}' for m in wav2vec2_classes[:2]}}
        self.assertEqual(set(wav2vec2_info['frameworks']), {'pt', 'tf', 'flax'})
        model_classes = {k: set(v) for (k, v) in wav2vec2_info['model_classes'].items()}
        self.assertEqual(model_classes, expected_model_classes)
        all_wav2vec2_files = wav2vec2_info['model_files']
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_wav2vec2_files['model_files']}
        self.assertEqual(model_files, WAV2VEC2_MODEL_FILES)
        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_wav2vec2_files['test_files']}
        wav2vec2_test_files = {'tests/models/wav2vec2/test_feature_extraction_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_tf_wav2vec2.py', 'tests/models/wav2vec2/test_modeling_flax_wav2vec2.py', 'tests/models/wav2vec2/test_processor_wav2vec2.py', 'tests/models/wav2vec2/test_tokenization_wav2vec2.py'}
        self.assertEqual(test_files, wav2vec2_test_files)
        doc_file = str(Path(all_wav2vec2_files['doc_file']).relative_to(REPO_PATH))
        self.assertEqual(doc_file, 'docs/source/en/model_doc/wav2vec2.md')
        self.assertEqual(all_wav2vec2_files['module_name'], 'wav2vec2')
        wav2vec2_model_patterns = wav2vec2_info['model_patterns']
        self.assertEqual(wav2vec2_model_patterns.model_name, 'Wav2Vec2')
        self.assertEqual(wav2vec2_model_patterns.checkpoint, 'facebook/wav2vec2-base-960h')
        self.assertEqual(wav2vec2_model_patterns.model_type, 'wav2vec2')
        self.assertEqual(wav2vec2_model_patterns.model_lower_cased, 'wav2vec2')
        self.assertEqual(wav2vec2_model_patterns.model_camel_cased, 'Wav2Vec2')
        self.assertEqual(wav2vec2_model_patterns.model_upper_cased, 'WAV_2_VEC_2')
        self.assertEqual(wav2vec2_model_patterns.config_class, 'Wav2Vec2Config')
        self.assertEqual(wav2vec2_model_patterns.feature_extractor_class, 'Wav2Vec2FeatureExtractor')
        self.assertEqual(wav2vec2_model_patterns.processor_class, 'Wav2Vec2Processor')
        self.assertEqual(wav2vec2_model_patterns.tokenizer_class, 'Wav2Vec2CTCTokenizer')

    def test_clean_frameworks_in_init_with_gpt(self):
        if False:
            return 10
        test_init = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available\n\n_import_structure = {\n    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],\n    "tokenization_gpt2": ["GPT2Tokenizer"],\n}\n\ntry:\n    if not is_tokenizers_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_gpt2"] = ["GPT2Model"]\n\ntry:\n    if not is_tf_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_tf_gpt2"] = ["TFGPT2Model"]\n\ntry:\n    if not is_flax_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2Model"]\n\nif TYPE_CHECKING:\n    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig\n    from .tokenization_gpt2 import GPT2Tokenizer\n\n    try:\n        if not is_tokenizers_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .tokenization_gpt2_fast import GPT2TokenizerFast\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_gpt2 import GPT2Model\n\n    try:\n        if not is_tf_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_tf_gpt2 import TFGPT2Model\n\n    try:\n        if not is_flax_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_flax_gpt2 import FlaxGPT2Model\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        init_no_tokenizer = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available\n\n_import_structure = {\n    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],\n}\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_gpt2"] = ["GPT2Model"]\n\ntry:\n    if not is_tf_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_tf_gpt2"] = ["TFGPT2Model"]\n\ntry:\n    if not is_flax_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2Model"]\n\nif TYPE_CHECKING:\n    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_gpt2 import GPT2Model\n\n    try:\n        if not is_tf_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_tf_gpt2 import TFGPT2Model\n\n    try:\n        if not is_flax_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_flax_gpt2 import FlaxGPT2Model\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        init_pt_only = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_tokenizers_available, is_torch_available\n\n_import_structure = {\n    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],\n    "tokenization_gpt2": ["GPT2Tokenizer"],\n}\n\ntry:\n    if not is_tokenizers_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_gpt2"] = ["GPT2Model"]\n\nif TYPE_CHECKING:\n    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig\n    from .tokenization_gpt2 import GPT2Tokenizer\n\n    try:\n        if not is_tokenizers_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .tokenization_gpt2_fast import GPT2TokenizerFast\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_gpt2 import GPT2Model\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        init_pt_only_no_tokenizer = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_torch_available\n\n_import_structure = {\n    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],\n}\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_gpt2"] = ["GPT2Model"]\n\nif TYPE_CHECKING:\n    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_gpt2 import GPT2Model\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, '../__init__.py')
            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, keep_processing=False)
            self.check_result(file_name, init_no_tokenizer)
            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, frameworks=['pt'])
            self.check_result(file_name, init_pt_only)
            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, frameworks=['pt'], keep_processing=False)
            self.check_result(file_name, init_pt_only_no_tokenizer)

    def test_clean_frameworks_in_init_with_vit(self):
        if False:
            while True:
                i = 10
        test_init = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available, is_vision_available\n\n_import_structure = {\n    "configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],\n}\n\ntry:\n    if not is_vision_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["image_processing_vit"] = ["ViTImageProcessor"]\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_vit"] = ["ViTModel"]\n\ntry:\n    if not is_tf_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_tf_vit"] = ["TFViTModel"]\n\ntry:\n    if not is_flax_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_flax_vit"] = ["FlaxViTModel"]\n\nif TYPE_CHECKING:\n    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig\n\n    try:\n        if not is_vision_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .image_processing_vit import ViTImageProcessor\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_vit import ViTModel\n\n    try:\n        if not is_tf_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_tf_vit import TFViTModel\n\n    try:\n        if not is_flax_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_flax_vit import FlaxViTModel\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        init_no_feature_extractor = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available\n\n_import_structure = {\n    "configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],\n}\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_vit"] = ["ViTModel"]\n\ntry:\n    if not is_tf_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_tf_vit"] = ["TFViTModel"]\n\ntry:\n    if not is_flax_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_flax_vit"] = ["FlaxViTModel"]\n\nif TYPE_CHECKING:\n    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_vit import ViTModel\n\n    try:\n        if not is_tf_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_tf_vit import TFViTModel\n\n    try:\n        if not is_flax_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_flax_vit import FlaxViTModel\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        init_pt_only = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_torch_available, is_vision_available\n\n_import_structure = {\n    "configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],\n}\n\ntry:\n    if not is_vision_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["image_processing_vit"] = ["ViTImageProcessor"]\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_vit"] = ["ViTModel"]\n\nif TYPE_CHECKING:\n    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig\n\n    try:\n        if not is_vision_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .image_processing_vit import ViTImageProcessor\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_vit import ViTModel\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        init_pt_only_no_feature_extractor = '\nfrom typing import TYPE_CHECKING\n\nfrom ...utils import _LazyModule, is_torch_available\n\n_import_structure = {\n    "configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],\n}\n\ntry:\n    if not is_torch_available():\n        raise OptionalDependencyNotAvailable()\nexcept OptionalDependencyNotAvailable:\n    pass\nelse:\n    _import_structure["modeling_vit"] = ["ViTModel"]\n\nif TYPE_CHECKING:\n    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig\n\n    try:\n        if not is_torch_available():\n            raise OptionalDependencyNotAvailable()\n    except OptionalDependencyNotAvailable:\n        pass\n    else:\n        from .modeling_vit import ViTModel\n\nelse:\n    import sys\n\n    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)\n'
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, '../__init__.py')
            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, keep_processing=False)
            self.check_result(file_name, init_no_feature_extractor)
            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, frameworks=['pt'])
            self.check_result(file_name, init_pt_only)
            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, frameworks=['pt'], keep_processing=False)
            self.check_result(file_name, init_pt_only_no_feature_extractor)

    def test_duplicate_doc_file(self):
        if False:
            while True:
                i = 10
        test_doc = '\n# GPT2\n\n## Overview\n\nOverview of the model.\n\n## GPT2Config\n\n[[autodoc]] GPT2Config\n\n## GPT2Tokenizer\n\n[[autodoc]] GPT2Tokenizer\n    - save_vocabulary\n\n## GPT2TokenizerFast\n\n[[autodoc]] GPT2TokenizerFast\n\n## GPT2 specific outputs\n\n[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput\n\n[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput\n\n## GPT2Model\n\n[[autodoc]] GPT2Model\n    - forward\n\n## TFGPT2Model\n\n[[autodoc]] TFGPT2Model\n    - call\n\n## FlaxGPT2Model\n\n[[autodoc]] FlaxGPT2Model\n    - __call__\n\n'
        test_new_doc = '\n# GPT-New New\n\n## Overview\n\nThe GPT-New New model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.\n<INSERT SHORT SUMMARY HERE>\n\nThe abstract from the paper is the following:\n\n*<INSERT PAPER ABSTRACT HERE>*\n\nTips:\n\n<INSERT TIPS ABOUT MODEL HERE>\n\nThis model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).\nThe original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).\n\n\n## GPTNewNewConfig\n\n[[autodoc]] GPTNewNewConfig\n\n## GPTNewNewTokenizer\n\n[[autodoc]] GPTNewNewTokenizer\n    - save_vocabulary\n\n## GPTNewNewTokenizerFast\n\n[[autodoc]] GPTNewNewTokenizerFast\n\n## GPTNewNew specific outputs\n\n[[autodoc]] models.gpt_new_new.modeling_gpt_new_new.GPTNewNewDoubleHeadsModelOutput\n\n[[autodoc]] models.gpt_new_new.modeling_tf_gpt_new_new.TFGPTNewNewDoubleHeadsModelOutput\n\n## GPTNewNewModel\n\n[[autodoc]] GPTNewNewModel\n    - forward\n\n## TFGPTNewNewModel\n\n[[autodoc]] TFGPTNewNewModel\n    - call\n\n## FlaxGPTNewNewModel\n\n[[autodoc]] FlaxGPTNewNewModel\n    - __call__\n\n'
        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_file = os.path.join(tmp_dir, 'gpt2.md')
            new_doc_file = os.path.join(tmp_dir, 'gpt-new-new.md')
            gpt2_model_patterns = ModelPatterns('GPT2', 'gpt2', tokenizer_class='GPT2Tokenizer')
            new_model_patterns = ModelPatterns('GPT-New New', 'huggingface/gpt-new-new', tokenizer_class='GPTNewNewTokenizer')
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns)
            self.check_result(new_doc_file, test_new_doc)
            test_new_doc_pt_only = test_new_doc.replace('\n## TFGPTNewNewModel\n\n[[autodoc]] TFGPTNewNewModel\n    - call\n\n## FlaxGPTNewNewModel\n\n[[autodoc]] FlaxGPTNewNewModel\n    - __call__\n\n', '')
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns, frameworks=['pt'])
            self.check_result(new_doc_file, test_new_doc_pt_only)
            test_new_doc_no_tok = test_new_doc.replace('\n## GPTNewNewTokenizer\n\n[[autodoc]] GPTNewNewTokenizer\n    - save_vocabulary\n\n## GPTNewNewTokenizerFast\n\n[[autodoc]] GPTNewNewTokenizerFast\n', '')
            new_model_patterns = ModelPatterns('GPT-New New', 'huggingface/gpt-new-new', tokenizer_class='GPT2Tokenizer')
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns)
            print(test_new_doc_no_tok)
            self.check_result(new_doc_file, test_new_doc_no_tok)
            test_new_doc_pt_only_no_tok = test_new_doc_no_tok.replace('\n## TFGPTNewNewModel\n\n[[autodoc]] TFGPTNewNewModel\n    - call\n\n## FlaxGPTNewNewModel\n\n[[autodoc]] FlaxGPTNewNewModel\n    - __call__\n\n', '')
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns, frameworks=['pt'])
            self.check_result(new_doc_file, test_new_doc_pt_only_no_tok)