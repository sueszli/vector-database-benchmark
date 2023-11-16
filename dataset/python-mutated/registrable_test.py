import inspect
import os
from typing import List
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import push_python_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.regularizers.regularizer import Regularizer

@pytest.fixture()
def empty_registrable():
    if False:
        i = 10
        return i + 15

    class EmptyRegistrable(Registrable):
        pass
    yield EmptyRegistrable

class TestRegistrable(AllenNlpTestCase):

    def test_registrable_functionality_works(self):
        if False:
            while True:
                i = 10
        base_class = Tokenizer
        assert 'fake' not in base_class.list_available()

        @base_class.register('fake')
        class Fake(base_class):
            pass
        assert base_class.by_name('fake') == Fake
        default = base_class.default_implementation
        if default is not None:
            assert base_class.list_available()[0] == default
            base_class.default_implementation = 'fake'
            assert base_class.list_available()[0] == 'fake'
            with pytest.raises(ConfigurationError):
                base_class.default_implementation = 'not present'
                base_class.list_available()
            base_class.default_implementation = default
        with pytest.raises(ConfigurationError):

            @base_class.register('fake')
            class FakeAlternate(base_class):
                pass

        @base_class.register('fake', exist_ok=True)
        class FakeAlternate2(base_class):
            pass
        assert base_class.by_name('fake') == FakeAlternate2
        del Registrable._registry[base_class]['fake']

    def test_registry_has_builtin_tokenizers(self):
        if False:
            for i in range(10):
                print('nop')
        assert Tokenizer.by_name('spacy').__name__ == 'SpacyTokenizer'
        assert Tokenizer.by_name('character').__name__ == 'CharacterTokenizer'

    def test_registry_has_builtin_token_indexers(self):
        if False:
            i = 10
            return i + 15
        assert TokenIndexer.by_name('single_id').__name__ == 'SingleIdTokenIndexer'
        assert TokenIndexer.by_name('characters').__name__ == 'TokenCharactersIndexer'

    def test_registry_has_builtin_regularizers(self):
        if False:
            print('Hello World!')
        assert Regularizer.by_name('l1').__name__ == 'L1Regularizer'
        assert Regularizer.by_name('l2').__name__ == 'L2Regularizer'

    def test_registry_has_builtin_token_embedders(self):
        if False:
            return 10
        assert TokenEmbedder.by_name('embedding').__name__ == 'Embedding'
        assert TokenEmbedder.by_name('character_encoding').__name__ == 'TokenCharactersEncoder'

    def test_registry_has_builtin_text_field_embedders(self):
        if False:
            print('Hello World!')
        assert TextFieldEmbedder.by_name('basic').__name__ == 'BasicTextFieldEmbedder'

    def test_implicit_include_package(self):
        if False:
            return 10
        packagedir = self.TEST_DIR / 'testpackage'
        packagedir.mkdir()
        (packagedir / '__init__.py').touch()
        with push_python_path(self.TEST_DIR):
            reader = DatasetReader.by_name('text_classification_json')
            with open(inspect.getabsfile(reader)) as f:
                code = f.read().replace('@DatasetReader.register("text_classification_json")', '@DatasetReader.register("text_classification_json-fake")')
            with open(os.path.join(packagedir, 'reader.py'), 'w') as f:
                f.write(code)
            with pytest.raises(ConfigurationError) as exc:
                DatasetReader.by_name('text_classification_json-fake')
                assert 'is not a registered name' in str(exc.value)
            with pytest.raises(ConfigurationError) as exc:
                DatasetReader.by_name('testpackage.text_classification_json.TextClassificationJsonReader')
                assert 'unable to import module' in str(exc.value)
            with pytest.raises(ConfigurationError):
                DatasetReader.by_name('testpackage.reader.FakeReader')
                assert 'unable to find class' in str(exc.value)
            duplicate_reader = DatasetReader.by_name('testpackage.reader.TextClassificationJsonReader')
            assert duplicate_reader.__name__ == 'TextClassificationJsonReader'

    def test_to_params_no_arguments(self, empty_registrable):
        if False:
            print('Hello World!')

        @empty_registrable.register('no-args')
        class NoArguments(empty_registrable):
            pass
        obj = NoArguments()
        assert obj.to_params().params == {'type': 'no-args'}

    def test_to_params_no_pos_arguments(self, empty_registrable):
        if False:
            while True:
                i = 10

        @empty_registrable.register('no-pos-args')
        class NoPosArguments(empty_registrable):

            def __init__(self, A: bool=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.A = A
        obj = NoPosArguments()
        assert obj.to_params().params == {'type': 'no-pos-args'}

    def test_to_params_pos_arguments(self, empty_registrable):
        if False:
            print('Hello World!')

        @empty_registrable.register('pos-args')
        class PosArguments(empty_registrable):

            def __init__(self, A: bool, B: int, C: List):
                if False:
                    return 10
                self.A = A
                self._B = B
                self._msg = C
        obj = PosArguments(False, 5, [])
        assert obj.to_params().params == {'type': 'pos-args', 'A': False, 'B': 5}

    def test_to_params_not_registered(self, empty_registrable):
        if False:
            while True:
                i = 10

        class NotRegistered(empty_registrable):
            pass
        obj = NotRegistered()
        with pytest.raises(KeyError):
            obj.to_params()

    def test_to_params_nested(self, empty_registrable):
        if False:
            print('Hello World!')

        class NestedBase(empty_registrable):
            pass

        @NestedBase.register('nested')
        class NestedClass(NestedBase):
            pass
        obj = NestedClass()
        assert obj.to_params().params == {'type': 'nested'}

@pytest.mark.parametrize('name', ['sequence-tagging', 'sequence-taggign'])
def test_suggestions_when_name_not_found(name):
    if False:
        return 10
    with pytest.raises(ConfigurationError) as exc:
        DatasetReader.by_name(name)
        assert "did you mean 'sequence_tagging'?" in str(exc.value)