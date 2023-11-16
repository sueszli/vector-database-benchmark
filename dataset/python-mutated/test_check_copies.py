import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, 'utils'))
import check_copies
from check_copies import convert_to_localized_md, find_code_in_transformers, is_copy_consistent
REFERENCE_CODE = '    def __init__(self, config):\n        super().__init__()\n        self.transform = BertPredictionHeadTransform(config)\n\n        # The output weights are the same as the input embeddings, but there is\n        # an output-only bias for each token.\n        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n\n        self.bias = nn.Parameter(torch.zeros(config.vocab_size))\n\n        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`\n        self.decoder.bias = self.bias\n\n    def forward(self, hidden_states):\n        hidden_states = self.transform(hidden_states)\n        hidden_states = self.decoder(hidden_states)\n        return hidden_states\n'
MOCK_BERT_CODE = 'from ...modeling_utils import PreTrainedModel\n\ndef bert_function(x):\n    return x\n\n\nclass BertAttention(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n\n\nclass BertModel(BertPreTrainedModel):\n    def __init__(self, config):\n        super().__init__()\n        self.bert = BertEncoder(config)\n\n    @add_docstring(BERT_DOCSTRING)\n    def forward(self, x):\n        return self.bert(x)\n'
MOCK_BERT_COPY_CODE = 'from ...modeling_utils import PreTrainedModel\n\n# Copied from transformers.models.bert.modeling_bert.bert_function\ndef bert_copy_function(x):\n    return x\n\n\n# Copied from transformers.models.bert.modeling_bert.BertAttention\nclass BertCopyAttention(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n\n\n# Copied from transformers.models.bert.modeling_bert.BertModel with Bert->BertCopy all-casing\nclass BertCopyModel(BertCopyPreTrainedModel):\n    def __init__(self, config):\n        super().__init__()\n        self.bertcopy = BertCopyEncoder(config)\n\n    @add_docstring(BERTCOPY_DOCSTRING)\n    def forward(self, x):\n        return self.bertcopy(x)\n'

def replace_in_file(filename, old, new):
    if False:
        return 10
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def create_tmp_repo(tmp_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a mock repository in a temporary folder for testing.\n    '
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    model_dir = tmp_dir / 'src' / 'transformers' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    models = {'bert': MOCK_BERT_CODE, 'bertcopy': MOCK_BERT_COPY_CODE}
    for (model, code) in models.items():
        model_subdir = model_dir / model
        model_subdir.mkdir(exist_ok=True)
        with open(model_subdir / f'modeling_{model}.py', 'w', encoding='utf-8') as f:
            f.write(code)

@contextmanager
def patch_transformer_repo_path(new_folder):
    if False:
        i = 10
        return i + 15
    '\n    Temporarily patches the variables defines in `check_copies` to use a different location for the repo.\n    '
    old_repo_path = check_copies.REPO_PATH
    old_doc_path = check_copies.PATH_TO_DOCS
    old_transformer_path = check_copies.TRANSFORMERS_PATH
    repo_path = Path(new_folder).resolve()
    check_copies.REPO_PATH = str(repo_path)
    check_copies.PATH_TO_DOCS = str(repo_path / 'docs' / 'source' / 'en')
    check_copies.TRANSFORMERS_PATH = str(repo_path / 'src' / 'transformers')
    try:
        yield
    finally:
        check_copies.REPO_PATH = old_repo_path
        check_copies.PATH_TO_DOCS = old_doc_path
        check_copies.TRANSFORMERS_PATH = old_transformer_path

class CopyCheckTester(unittest.TestCase):

    def test_find_code_in_transformers(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                code = find_code_in_transformers('models.bert.modeling_bert.BertAttention')
        reference_code = 'class BertAttention(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n'
        self.assertEqual(code, reference_code)

    def test_is_copy_consistent(self):
        if False:
            return 10
        path_to_check = ['src', 'transformers', 'models', 'bertcopy', 'modeling_bertcopy.py']
        with tempfile.TemporaryDirectory() as tmp_folder:
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                file_to_check = os.path.join(tmp_folder, *path_to_check)
                diffs = is_copy_consistent(file_to_check)
                self.assertEqual(diffs, [])
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                file_to_check = os.path.join(tmp_folder, *path_to_check)
                replace_in_file(file_to_check, 'self.bertcopy(x)', 'self.bert(x)')
                diffs = is_copy_consistent(file_to_check)
                self.assertEqual(diffs, [['models.bert.modeling_bert.BertModel', 22]])
                diffs = is_copy_consistent(file_to_check, overwrite=True)
                with open(file_to_check, 'r', encoding='utf-8') as f:
                    self.assertEqual(f.read(), MOCK_BERT_COPY_CODE)

    def test_convert_to_localized_md(self):
        if False:
            return 10
        localized_readme = check_copies.LOCALIZED_READMES['README_zh-hans.md']
        md_list = '1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.\n1. **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (from HuggingFace), released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/main/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/distillation) and a German version of DistilBERT.\n1. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.'
        localized_md_list = '1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (来自 Google Research and the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n'
        converted_md_list_sample = '1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (来自 Google Research and the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n1. **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (来自 HuggingFace) 伴随论文 [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) 由 Victor Sanh, Lysandre Debut and Thomas Wolf 发布。 The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/main/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/distillation) and a German version of DistilBERT.\n1. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (来自 Google Research/Stanford University) 伴随论文 [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) 由 Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning 发布。\n'
        (num_models_equal, converted_md_list) = convert_to_localized_md(md_list, localized_md_list, localized_readme['format_model_list'])
        self.assertFalse(num_models_equal)
        self.assertEqual(converted_md_list, converted_md_list_sample)
        (num_models_equal, converted_md_list) = convert_to_localized_md(md_list, converted_md_list, localized_readme['format_model_list'])
        self.assertTrue(num_models_equal)
        link_changed_md_list = '1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.'
        link_unchanged_md_list = '1. **[ALBERT](https://huggingface.co/transformers/main/model_doc/albert.html)** (来自 Google Research and the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n'
        converted_md_list_sample = '1. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (来自 Google Research and the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。\n'
        (num_models_equal, converted_md_list) = convert_to_localized_md(link_changed_md_list, link_unchanged_md_list, localized_readme['format_model_list'])
        self.assertEqual(converted_md_list, converted_md_list_sample)