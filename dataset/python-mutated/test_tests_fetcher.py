import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from git import Repo
from transformers.testing_utils import CaptureStdout
REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(REPO_PATH, 'utils'))
import tests_fetcher
from tests_fetcher import checkout_commit, clean_code, create_module_to_test_map, create_reverse_dependency_map, create_reverse_dependency_tree, diff_is_docstring_only, extract_imports, get_all_tests, get_diff, get_module_dependencies, get_tree_starting_at, infer_tests_to_run, init_test_examples_dependencies, parse_commit_message, print_tree_deps_of
BERT_MODELING_FILE = 'src/transformers/models/bert/modeling_bert.py'
BERT_MODEL_FILE = "from ...modeling_utils import PreTrainedModel\nfrom ...utils import is_torch_available\nfrom .configuration_bert import BertConfig\n\nclass BertModel:\n    '''\n    This is the docstring.\n    '''\n    This is the code\n"
BERT_MODEL_FILE_NEW_DOCSTRING = "from ...modeling_utils import PreTrainedModel\nfrom ...utils import is_torch_available\nfrom .configuration_bert import BertConfig\n\nclass BertModel:\n    '''\n    This is the docstring. It has been updated.\n    '''\n    This is the code\n"
BERT_MODEL_FILE_NEW_CODE = "from ...modeling_utils import PreTrainedModel\nfrom ...utils import is_torch_available\nfrom .configuration_bert import BertConfig\n\nclass BertModel:\n    '''\n    This is the docstring.\n    '''\n    This is the code. It has been updated\n"

def create_tmp_repo(tmp_dir, models=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a repository in a temporary directory mimicking the structure of Transformers. Uses the list of models\n    provided (which defaults to just `["bert"]`).\n    '
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    repo = Repo.init(tmp_dir)
    if models is None:
        models = ['bert']
    class_names = [model[0].upper() + model[1:] for model in models]
    transformers_dir = tmp_dir / 'src' / 'transformers'
    transformers_dir.mkdir(parents=True, exist_ok=True)
    with open(transformers_dir / '__init__.py', 'w') as f:
        init_lines = ['from .utils import cached_file, is_torch_available']
        init_lines.extend([f'from .models.{model} import {cls}Config, {cls}Model' for (model, cls) in zip(models, class_names)])
        f.write('\n'.join(init_lines) + '\n')
    with open(transformers_dir / 'configuration_utils.py', 'w') as f:
        f.write('from .utils import cached_file\n\ncode')
    with open(transformers_dir / 'modeling_utils.py', 'w') as f:
        f.write('from .utils import cached_file\n\ncode')
    utils_dir = tmp_dir / 'src' / 'transformers' / 'utils'
    utils_dir.mkdir(exist_ok=True)
    with open(utils_dir / '__init__.py', 'w') as f:
        f.write('from .hub import cached_file\nfrom .imports import is_torch_available\n')
    with open(utils_dir / 'hub.py', 'w') as f:
        f.write('import huggingface_hub\n\ncode')
    with open(utils_dir / 'imports.py', 'w') as f:
        f.write('code')
    model_dir = tmp_dir / 'src' / 'transformers' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / '__init__.py', 'w') as f:
        f.write('\n'.join([f'import {model}' for model in models]))
    for (model, cls) in zip(models, class_names):
        model_dir = tmp_dir / 'src' / 'transformers' / 'models' / model
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / '__init__.py', 'w') as f:
            f.write(f'from .configuration_{model} import {cls}Config\nfrom .modeling_{model} import {cls}Model\n')
        with open(model_dir / f'configuration_{model}.py', 'w') as f:
            f.write('from ...configuration_utils import PretrainedConfig\ncode')
        with open(model_dir / f'modeling_{model}.py', 'w') as f:
            modeling_code = BERT_MODEL_FILE.replace('bert', model).replace('Bert', cls)
            f.write(modeling_code)
    test_dir = tmp_dir / 'tests'
    test_dir.mkdir(exist_ok=True)
    with open(test_dir / 'test_modeling_common.py', 'w') as f:
        f.write('from transformers.modeling_utils import PreTrainedModel\ncode')
    for (model, cls) in zip(models, class_names):
        test_model_dir = test_dir / 'models' / model
        test_model_dir.mkdir(parents=True, exist_ok=True)
        (test_model_dir / '__init__.py').touch()
        with open(test_model_dir / f'test_modeling_{model}.py', 'w') as f:
            f.write(f'from transformers import {cls}Config, {cls}Model\nfrom ...test_modeling_common import ModelTesterMixin\n\ncode')
    example_dir = tmp_dir / 'examples'
    example_dir.mkdir(exist_ok=True)
    for framework in ['flax', 'pytorch', 'tensorflow']:
        framework_dir = example_dir / framework
        framework_dir.mkdir(exist_ok=True)
        with open(framework_dir / f'test_{framework}_examples.py', 'w') as f:
            f.write('test_args = "run_glue.py"\n')
        glue_dir = framework_dir / 'text-classification'
        glue_dir.mkdir(exist_ok=True)
        with open(glue_dir / 'run_glue.py', 'w') as f:
            f.write('from transformers import BertModel\n\ncode')
    repo.index.add(['examples', 'src', 'tests'])
    repo.index.commit('Initial commit')
    repo.create_head('main')
    repo.head.reference = repo.refs.main
    repo.delete_head('master')
    return repo

@contextmanager
def patch_transformer_repo_path(new_folder):
    if False:
        for i in range(10):
            print('nop')
    '\n    Temporarily patches the variables defines in `tests_fetcher` to use a different location for the repo.\n    '
    old_repo_path = tests_fetcher.PATH_TO_REPO
    tests_fetcher.PATH_TO_REPO = Path(new_folder).resolve()
    tests_fetcher.PATH_TO_EXAMPLES = tests_fetcher.PATH_TO_REPO / 'examples'
    tests_fetcher.PATH_TO_TRANFORMERS = tests_fetcher.PATH_TO_REPO / 'src/transformers'
    tests_fetcher.PATH_TO_TESTS = tests_fetcher.PATH_TO_REPO / 'tests'
    try:
        yield
    finally:
        tests_fetcher.PATH_TO_REPO = old_repo_path
        tests_fetcher.PATH_TO_EXAMPLES = tests_fetcher.PATH_TO_REPO / 'examples'
        tests_fetcher.PATH_TO_TRANFORMERS = tests_fetcher.PATH_TO_REPO / 'src/transformers'
        tests_fetcher.PATH_TO_TESTS = tests_fetcher.PATH_TO_REPO / 'tests'

def commit_changes(filenames, contents, repo, commit_message='Commit'):
    if False:
        return 10
    '\n    Commit new `contents` to `filenames` inside a given `repo`.\n    '
    if not isinstance(filenames, list):
        filenames = [filenames]
    if not isinstance(contents, list):
        contents = [contents]
    folder = Path(repo.working_dir)
    for (filename, content) in zip(filenames, contents):
        with open(folder / filename, 'w') as f:
            f.write(content)
    repo.index.add(filenames)
    commit = repo.index.commit(commit_message)
    return commit.hexsha

class TestFetcherTester(unittest.TestCase):

    def test_checkout_commit(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            repo = create_tmp_repo(tmp_folder)
            initial_sha = repo.head.commit.hexsha
            new_sha = commit_changes(BERT_MODELING_FILE, BERT_MODEL_FILE_NEW_DOCSTRING, repo)
            assert repo.head.commit.hexsha == new_sha
            with checkout_commit(repo, initial_sha):
                assert repo.head.commit.hexsha == initial_sha
                with open(tmp_folder / BERT_MODELING_FILE) as f:
                    assert f.read() == BERT_MODEL_FILE
            assert repo.head.commit.hexsha == new_sha
            with open(tmp_folder / BERT_MODELING_FILE) as f:
                assert f.read() == BERT_MODEL_FILE_NEW_DOCSTRING

    def test_clean_code(self):
        if False:
            i = 10
            return i + 15
        assert clean_code('"""\nDocstring\n"""\ncode\n"""Long string"""\ncode\n') == 'code\ncode'
        assert clean_code("'''\nDocstring\n'''\ncode\n'''Long string'''\ncode\n'''") == 'code\ncode'
        assert clean_code('code\n# Comment\ncode') == 'code\ncode'
        assert clean_code('code  # inline comment\ncode') == 'code  \ncode'

    def test_get_all_tests(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                assert get_all_tests() == ['tests/models/bert', 'tests/test_modeling_common.py']

    def test_get_all_tests_on_full_repo(self):
        if False:
            return 10
        all_tests = get_all_tests()
        assert 'tests/models/albert' in all_tests
        assert 'tests/models/bert' in all_tests
        assert 'tests/repo_utils' in all_tests
        assert 'tests/test_pipeline_mixin.py' in all_tests
        assert 'tests/models' not in all_tests
        assert 'tests/__pycache__' not in all_tests
        assert 'tests/models/albert/test_modeling_albert.py' not in all_tests
        assert 'tests/repo_utils/test_tests_fetcher.py' not in all_tests

    def test_diff_is_docstring_only(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            repo = create_tmp_repo(tmp_folder)
            branching_point = repo.refs.main.commit
            bert_file = BERT_MODELING_FILE
            commit_changes(bert_file, BERT_MODEL_FILE_NEW_DOCSTRING, repo)
            assert diff_is_docstring_only(repo, branching_point, bert_file)
            commit_changes(bert_file, BERT_MODEL_FILE_NEW_CODE, repo)
            assert not diff_is_docstring_only(repo, branching_point, bert_file)

    def test_get_diff(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            repo = create_tmp_repo(tmp_folder)
            initial_commit = repo.refs.main.commit
            bert_file = BERT_MODELING_FILE
            commit_changes(bert_file, BERT_MODEL_FILE_NEW_DOCSTRING, repo)
            assert get_diff(repo, repo.head.commit, repo.head.commit.parents) == []
            commit_changes(bert_file, BERT_MODEL_FILE_NEW_DOCSTRING + '\n# Adding a comment\n', repo)
            assert get_diff(repo, repo.head.commit, repo.head.commit.parents) == []
            commit_changes(bert_file, BERT_MODEL_FILE_NEW_CODE, repo)
            assert get_diff(repo, repo.head.commit, repo.head.commit.parents) == ['src/transformers/models/bert/modeling_bert.py']
            commit_changes('src/transformers/utils/hub.py', 'import huggingface_hub\n\nnew code', repo)
            assert get_diff(repo, repo.head.commit, repo.head.commit.parents) == ['src/transformers/utils/hub.py']
            assert get_diff(repo, repo.head.commit, [initial_commit]) == ['src/transformers/models/bert/modeling_bert.py', 'src/transformers/utils/hub.py']

    def test_extract_imports_relative(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            expected_bert_imports = [('src/transformers/modeling_utils.py', ['PreTrainedModel']), ('src/transformers/utils/__init__.py', ['is_torch_available']), ('src/transformers/models/bert/configuration_bert.py', ['BertConfig'])]
            expected_utils_imports = [('src/transformers/utils/hub.py', ['cached_file']), ('src/transformers/utils/imports.py', ['is_torch_available'])]
            with patch_transformer_repo_path(tmp_folder):
                assert extract_imports(BERT_MODELING_FILE) == expected_bert_imports
                assert extract_imports('src/transformers/utils/__init__.py') == expected_utils_imports
            with open(tmp_folder / BERT_MODELING_FILE, 'w') as f:
                f.write('from ...utils import cached_file, is_torch_available\nfrom .configuration_bert import BertConfig\n')
            expected_bert_imports = [('src/transformers/utils/__init__.py', ['cached_file', 'is_torch_available']), ('src/transformers/models/bert/configuration_bert.py', ['BertConfig'])]
            with patch_transformer_repo_path(tmp_folder):
                assert extract_imports(BERT_MODELING_FILE) == expected_bert_imports
            with open(tmp_folder / BERT_MODELING_FILE, 'w') as f:
                f.write('from ...utils import (\n    cached_file,\n    is_torch_available\n)\nfrom .configuration_bert import BertConfig\n')
            expected_bert_imports = [('src/transformers/models/bert/configuration_bert.py', ['BertConfig']), ('src/transformers/utils/__init__.py', ['cached_file', 'is_torch_available'])]
            with patch_transformer_repo_path(tmp_folder):
                assert extract_imports(BERT_MODELING_FILE) == expected_bert_imports

    def test_extract_imports_absolute(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            with open(tmp_folder / BERT_MODELING_FILE, 'w') as f:
                f.write('from transformers.utils import cached_file, is_torch_available\nfrom transformers.models.bert.configuration_bert import BertConfig\n')
            expected_bert_imports = [('src/transformers/utils/__init__.py', ['cached_file', 'is_torch_available']), ('src/transformers/models/bert/configuration_bert.py', ['BertConfig'])]
            with patch_transformer_repo_path(tmp_folder):
                assert extract_imports(BERT_MODELING_FILE) == expected_bert_imports
            with open(tmp_folder / BERT_MODELING_FILE, 'w') as f:
                f.write('from transformers.utils import (\n    cached_file,\n    is_torch_available\n)\nfrom transformers.models.bert.configuration_bert import BertConfig\n')
            expected_bert_imports = [('src/transformers/models/bert/configuration_bert.py', ['BertConfig']), ('src/transformers/utils/__init__.py', ['cached_file', 'is_torch_available'])]
            with patch_transformer_repo_path(tmp_folder):
                assert extract_imports(BERT_MODELING_FILE) == expected_bert_imports
            with open(tmp_folder / BERT_MODELING_FILE, 'w') as f:
                f.write('from transformers.utils import (\n    cached_file,\n    is_torch_available\n)\nfrom transformers import BertConfig\n')
            expected_bert_imports = [('src/transformers/__init__.py', ['BertConfig']), ('src/transformers/utils/__init__.py', ['cached_file', 'is_torch_available'])]
            with patch_transformer_repo_path(tmp_folder):
                assert extract_imports(BERT_MODELING_FILE) == expected_bert_imports

    def test_get_module_dependencies(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            expected_bert_dependencies = ['src/transformers/modeling_utils.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/utils/imports.py']
            with patch_transformer_repo_path(tmp_folder):
                assert get_module_dependencies(BERT_MODELING_FILE) == expected_bert_dependencies
            expected_test_bert_dependencies = ['tests/test_modeling_common.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py']
            with patch_transformer_repo_path(tmp_folder):
                assert get_module_dependencies('tests/models/bert/test_modeling_bert.py') == expected_test_bert_dependencies
            (tmp_folder / 'src/transformers/utils/logging.py').touch()
            with open(tmp_folder / BERT_MODELING_FILE, 'a') as f:
                f.write('from ...utils import logging\n')
            expected_bert_dependencies = ['src/transformers/modeling_utils.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/utils/logging.py', 'src/transformers/utils/imports.py']
            with patch_transformer_repo_path(tmp_folder):
                assert get_module_dependencies(BERT_MODELING_FILE) == expected_bert_dependencies
            create_tmp_repo(tmp_folder)
            with open(tmp_folder / BERT_MODELING_FILE, 'a') as f:
                f.write('from ...utils import CONSTANT\n')
            expected_bert_dependencies = ['src/transformers/modeling_utils.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/utils/__init__.py', 'src/transformers/utils/imports.py']
            with patch_transformer_repo_path(tmp_folder):
                assert get_module_dependencies(BERT_MODELING_FILE) == expected_bert_dependencies
            create_tmp_repo(tmp_folder)
            expected_example_dependencies = ['src/transformers/models/bert/modeling_bert.py']
            with patch_transformer_repo_path(tmp_folder):
                assert get_module_dependencies('examples/pytorch/text-classification/run_glue.py') == expected_example_dependencies

    def test_create_reverse_dependency_tree(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                tree = create_reverse_dependency_tree()
            init_edges = ['src/transformers/utils/hub.py', 'src/transformers/utils/imports.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py']
            assert {f for (f, g) in tree if g == 'src/transformers/__init__.py'} == set(init_edges)
            bert_edges = ['src/transformers/modeling_utils.py', 'src/transformers/utils/imports.py', 'src/transformers/models/bert/configuration_bert.py']
            assert {f for (f, g) in tree if g == 'src/transformers/models/bert/modeling_bert.py'} == set(bert_edges)
            test_bert_edges = ['tests/test_modeling_common.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py']
            assert {f for (f, g) in tree if g == 'tests/models/bert/test_modeling_bert.py'} == set(test_bert_edges)

    def test_get_tree_starting_at(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                edges = create_reverse_dependency_tree()
                bert_tree = get_tree_starting_at('src/transformers/models/bert/modeling_bert.py', edges)
                config_utils_tree = get_tree_starting_at('src/transformers/configuration_utils.py', edges)
            expected_bert_tree = ['src/transformers/models/bert/modeling_bert.py', [('src/transformers/models/bert/modeling_bert.py', 'tests/models/bert/test_modeling_bert.py')]]
            assert bert_tree == expected_bert_tree
            expected_config_tree = ['src/transformers/configuration_utils.py', [('src/transformers/configuration_utils.py', 'src/transformers/models/bert/configuration_bert.py')], [('src/transformers/models/bert/configuration_bert.py', 'tests/models/bert/test_modeling_bert.py'), ('src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py')]]
            assert [set(v) for v in config_utils_tree] == [set(v) for v in expected_config_tree]

    def test_print_tree_deps_of(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            expected_std_out = 'src/transformers/models/bert/modeling_bert.py\n  tests/models/bert/test_modeling_bert.py\nsrc/transformers/configuration_utils.py\n  src/transformers/models/bert/configuration_bert.py\n    src/transformers/models/bert/modeling_bert.py\n    tests/models/bert/test_modeling_bert.py'
            expected_std_out_2 = 'src/transformers/models/bert/modeling_bert.py\n  tests/models/bert/test_modeling_bert.py\nsrc/transformers/configuration_utils.py\n  src/transformers/models/bert/configuration_bert.py\n    tests/models/bert/test_modeling_bert.py\n    src/transformers/models/bert/modeling_bert.py'
            with patch_transformer_repo_path(tmp_folder), CaptureStdout() as cs:
                print_tree_deps_of('src/transformers/models/bert/modeling_bert.py')
                print_tree_deps_of('src/transformers/configuration_utils.py')
            assert cs.out.strip() in [expected_std_out, expected_std_out_2]

    def test_init_test_examples_dependencies(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            expected_example_deps = {'examples/flax/test_flax_examples.py': ['examples/flax/text-classification/run_glue.py', 'examples/flax/test_flax_examples.py'], 'examples/pytorch/test_pytorch_examples.py': ['examples/pytorch/text-classification/run_glue.py', 'examples/pytorch/test_pytorch_examples.py'], 'examples/tensorflow/test_tensorflow_examples.py': ['examples/tensorflow/text-classification/run_glue.py', 'examples/tensorflow/test_tensorflow_examples.py']}
            expected_examples = {'examples/flax/test_flax_examples.py', 'examples/flax/text-classification/run_glue.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/pytorch/text-classification/run_glue.py', 'examples/tensorflow/test_tensorflow_examples.py', 'examples/tensorflow/text-classification/run_glue.py'}
            with patch_transformer_repo_path(tmp_folder):
                (example_deps, all_examples) = init_test_examples_dependencies()
                assert example_deps == expected_example_deps
                assert {str(f.relative_to(tmp_folder)) for f in all_examples} == expected_examples

    def test_create_reverse_dependency_map(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            create_tmp_repo(tmp_folder)
            with patch_transformer_repo_path(tmp_folder):
                reverse_map = create_reverse_dependency_map()
            expected_bert_deps = {'src/transformers/__init__.py', 'src/transformers/models/bert/__init__.py', 'tests/models/bert/test_modeling_bert.py', 'examples/flax/test_flax_examples.py', 'examples/flax/text-classification/run_glue.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/pytorch/text-classification/run_glue.py', 'examples/tensorflow/test_tensorflow_examples.py', 'examples/tensorflow/text-classification/run_glue.py'}
            assert set(reverse_map['src/transformers/models/bert/modeling_bert.py']) == expected_bert_deps
            expected_init_deps = {'src/transformers/utils/__init__.py', 'src/transformers/utils/hub.py', 'src/transformers/utils/imports.py', 'src/transformers/models/bert/__init__.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py', 'src/transformers/configuration_utils.py', 'src/transformers/modeling_utils.py', 'tests/test_modeling_common.py', 'tests/models/bert/test_modeling_bert.py', 'examples/flax/test_flax_examples.py', 'examples/flax/text-classification/run_glue.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/pytorch/text-classification/run_glue.py', 'examples/tensorflow/test_tensorflow_examples.py', 'examples/tensorflow/text-classification/run_glue.py'}
            assert set(reverse_map['src/transformers/__init__.py']) == expected_init_deps
            expected_init_deps = {'src/transformers/__init__.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py', 'tests/models/bert/test_modeling_bert.py', 'examples/flax/test_flax_examples.py', 'examples/flax/text-classification/run_glue.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/pytorch/text-classification/run_glue.py', 'examples/tensorflow/test_tensorflow_examples.py', 'examples/tensorflow/text-classification/run_glue.py'}
            assert set(reverse_map['src/transformers/models/bert/__init__.py']) == expected_init_deps
            create_tmp_repo(tmp_folder, models=['bert', 'gpt2'])
            with patch_transformer_repo_path(tmp_folder):
                reverse_map = create_reverse_dependency_map()
            expected_init_deps = {'src/transformers/__init__.py', 'src/transformers/models/bert/configuration_bert.py', 'src/transformers/models/bert/modeling_bert.py', 'tests/models/bert/test_modeling_bert.py', 'examples/flax/test_flax_examples.py', 'examples/flax/text-classification/run_glue.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/pytorch/text-classification/run_glue.py', 'examples/tensorflow/test_tensorflow_examples.py', 'examples/tensorflow/text-classification/run_glue.py'}
            assert set(reverse_map['src/transformers/models/bert/__init__.py']) == expected_init_deps

    def test_create_module_to_test_map(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            models = models = ['bert', 'gpt2'] + [f'bert{i}' for i in range(10)]
            create_tmp_repo(tmp_folder, models=models)
            with patch_transformer_repo_path(tmp_folder):
                test_map = create_module_to_test_map(filter_models=True)
            expected_bert_tests = {'examples/flax/test_flax_examples.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/tensorflow/test_tensorflow_examples.py', 'tests/models/bert/test_modeling_bert.py'}
            for model in models:
                if model != 'bert':
                    assert test_map[f'src/transformers/models/{model}/modeling_{model}.py'] == [f'tests/models/{model}/test_modeling_{model}.py']
                else:
                    assert set(test_map[f'src/transformers/models/{model}/modeling_{model}.py']) == expected_bert_tests
            expected_init_tests = {'examples/flax/test_flax_examples.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/tensorflow/test_tensorflow_examples.py', 'tests/test_modeling_common.py', 'tests/models/bert/test_modeling_bert.py', 'tests/models/gpt2/test_modeling_gpt2.py'}
            assert set(test_map['src/transformers/__init__.py']) == expected_init_tests

    def test_infer_tests_to_run(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            models = ['bert', 'gpt2'] + [f'bert{i}' for i in range(10)]
            repo = create_tmp_repo(tmp_folder, models=models)
            commit_changes('src/transformers/models/bert/modeling_bert.py', BERT_MODEL_FILE_NEW_CODE, repo)
            example_tests = {'examples/flax/test_flax_examples.py', 'examples/pytorch/test_pytorch_examples.py', 'examples/tensorflow/test_tensorflow_examples.py'}
            with patch_transformer_repo_path(tmp_folder):
                infer_tests_to_run(tmp_folder / 'test-output.txt', diff_with_last_commit=True)
                with open(tmp_folder / 'test-output.txt', 'r') as f:
                    tests_to_run = f.read()
                with open(tmp_folder / 'examples_test_list.txt', 'r') as f:
                    example_tests_to_run = f.read()
            assert tests_to_run == 'tests/models/bert/test_modeling_bert.py'
            assert set(example_tests_to_run.split(' ')) == example_tests
            repo = create_tmp_repo(tmp_folder, models=models)
            branch = repo.create_head('new_model')
            branch.checkout()
            with open(tmp_folder / 'src/transformers/__init__.py', 'a') as f:
                f.write('from .models.t5 import T5Config, T5Model\n')
            model_dir = tmp_folder / 'src/transformers/models/t5'
            model_dir.mkdir(exist_ok=True)
            with open(model_dir / '__init__.py', 'w') as f:
                f.write('from .configuration_t5 import T5Config\nfrom .modeling_t5 import T5Model\n')
            with open(model_dir / 'configuration_t5.py', 'w') as f:
                f.write('from ...configuration_utils import PretrainedConfig\ncode')
            with open(model_dir / 'modeling_t5.py', 'w') as f:
                modeling_code = BERT_MODEL_FILE.replace('bert', 't5').replace('Bert', 'T5')
                f.write(modeling_code)
            test_dir = tmp_folder / 'tests/models/t5'
            test_dir.mkdir(exist_ok=True)
            (test_dir / '__init__.py').touch()
            with open(test_dir / 'test_modeling_t5.py', 'w') as f:
                f.write('from transformers import T5Config, T5Model\nfrom ...test_modeling_common import ModelTesterMixin\n\ncode')
            repo.index.add(['src', 'tests'])
            repo.index.commit('Add T5 model')
            with patch_transformer_repo_path(tmp_folder):
                infer_tests_to_run(tmp_folder / 'test-output.txt')
                with open(tmp_folder / 'test-output.txt', 'r') as f:
                    tests_to_run = f.read()
                with open(tmp_folder / 'examples_test_list.txt', 'r') as f:
                    example_tests_to_run = f.read()
            expected_tests = {'tests/models/bert/test_modeling_bert.py', 'tests/models/gpt2/test_modeling_gpt2.py', 'tests/models/t5/test_modeling_t5.py', 'tests/test_modeling_common.py'}
            assert set(tests_to_run.split(' ')) == expected_tests
            assert set(example_tests_to_run.split(' ')) == example_tests
            with patch_transformer_repo_path(tmp_folder):
                infer_tests_to_run(tmp_folder / 'test-output.txt', filter_models=False)
                with open(tmp_folder / 'test-output.txt', 'r') as f:
                    tests_to_run = f.read()
                with open(tmp_folder / 'examples_test_list.txt', 'r') as f:
                    example_tests_to_run = f.read()
            expected_tests = [f'tests/models/{name}/test_modeling_{name}.py' for name in models + ['t5']]
            expected_tests = set(expected_tests + ['tests/test_modeling_common.py'])
            assert set(tests_to_run.split(' ')) == expected_tests
            assert set(example_tests_to_run.split(' ')) == example_tests

    def test_infer_tests_to_run_with_test_modifs(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            models = ['bert', 'gpt2'] + [f'bert{i}' for i in range(10)]
            repo = create_tmp_repo(tmp_folder, models=models)
            commit_changes('tests/models/bert/test_modeling_bert.py', 'from transformers import BertConfig, BertModel\nfrom ...test_modeling_common import ModelTesterMixin\n\ncode1', repo)
            with patch_transformer_repo_path(tmp_folder):
                infer_tests_to_run(tmp_folder / 'test-output.txt', diff_with_last_commit=True)
                with open(tmp_folder / 'test-output.txt', 'r') as f:
                    tests_to_run = f.read()
            assert tests_to_run == 'tests/models/bert/test_modeling_bert.py'

    def test_infer_tests_to_run_with_examples_modifs(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = Path(tmp_folder)
            models = ['bert', 'gpt2']
            repo = create_tmp_repo(tmp_folder, models=models)
            commit_changes('examples/pytorch/text-classification/run_glue.py', 'from transformers import BertModeln\n\ncode1', repo)
            with patch_transformer_repo_path(tmp_folder):
                infer_tests_to_run(tmp_folder / 'test-output.txt', diff_with_last_commit=True)
                with open(tmp_folder / 'examples_test_list.txt', 'r') as f:
                    example_tests_to_run = f.read()
            assert example_tests_to_run == 'examples/pytorch/test_pytorch_examples.py'
            repo = create_tmp_repo(tmp_folder, models=models)
            commit_changes('examples/pytorch/test_pytorch_examples.py', 'test_args = "run_glue.py"\nmore_code', repo)
            with patch_transformer_repo_path(tmp_folder):
                infer_tests_to_run(tmp_folder / 'test-output.txt', diff_with_last_commit=True)
                with open(tmp_folder / 'examples_test_list.txt', 'r') as f:
                    example_tests_to_run = f.read()
            assert example_tests_to_run == 'examples/pytorch/test_pytorch_examples.py'

    def test_parse_commit_message(self):
        if False:
            i = 10
            return i + 15
        assert parse_commit_message('Normal commit') == {'skip': False, 'no_filter': False, 'test_all': False}
        assert parse_commit_message('[skip ci] commit') == {'skip': True, 'no_filter': False, 'test_all': False}
        assert parse_commit_message('[ci skip] commit') == {'skip': True, 'no_filter': False, 'test_all': False}
        assert parse_commit_message('[skip-ci] commit') == {'skip': True, 'no_filter': False, 'test_all': False}
        assert parse_commit_message('[skip_ci] commit') == {'skip': True, 'no_filter': False, 'test_all': False}
        assert parse_commit_message('[no filter] commit') == {'skip': False, 'no_filter': True, 'test_all': False}
        assert parse_commit_message('[no-filter] commit') == {'skip': False, 'no_filter': True, 'test_all': False}
        assert parse_commit_message('[no_filter] commit') == {'skip': False, 'no_filter': True, 'test_all': False}
        assert parse_commit_message('[filter-no] commit') == {'skip': False, 'no_filter': True, 'test_all': False}
        assert parse_commit_message('[test all] commit') == {'skip': False, 'no_filter': False, 'test_all': True}
        assert parse_commit_message('[all test] commit') == {'skip': False, 'no_filter': False, 'test_all': True}
        assert parse_commit_message('[test-all] commit') == {'skip': False, 'no_filter': False, 'test_all': True}
        assert parse_commit_message('[all_test] commit') == {'skip': False, 'no_filter': False, 'test_all': True}