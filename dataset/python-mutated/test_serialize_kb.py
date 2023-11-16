from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import srsly
from numpy import zeros
from thinc.api import Config
from spacy import Errors, util
from spacy.kb.kb_in_memory import InMemoryLookupKB
from spacy.util import SimpleFrozenList, ensure_path, load_model_from_config, registry
from spacy.vocab import Vocab
from ..util import make_tempdir

def test_serialize_kb_disk(en_vocab):
    if False:
        while True:
            i = 10
    kb1 = _get_dummy_kb(en_vocab)
    _check_kb(kb1)
    with make_tempdir() as d:
        dir_path = ensure_path(d)
        if not dir_path.exists():
            dir_path.mkdir()
        file_path = dir_path / 'kb'
        kb1.to_disk(str(file_path))
        kb2 = InMemoryLookupKB(vocab=en_vocab, entity_vector_length=3)
        kb2.from_disk(str(file_path))
    _check_kb(kb2)

def _get_dummy_kb(vocab):
    if False:
        while True:
            i = 10
    kb = InMemoryLookupKB(vocab, entity_vector_length=3)
    kb.add_entity(entity='Q53', freq=33, entity_vector=[0, 5, 3])
    kb.add_entity(entity='Q17', freq=2, entity_vector=[7, 1, 0])
    kb.add_entity(entity='Q007', freq=7, entity_vector=[0, 0, 7])
    kb.add_entity(entity='Q44', freq=342, entity_vector=[4, 4, 4])
    kb.add_alias(alias='double07', entities=['Q17', 'Q007'], probabilities=[0.1, 0.9])
    kb.add_alias(alias='guy', entities=['Q53', 'Q007', 'Q17', 'Q44'], probabilities=[0.3, 0.3, 0.2, 0.1])
    kb.add_alias(alias='random', entities=['Q007'], probabilities=[1.0])
    return kb

def _check_kb(kb):
    if False:
        print('Hello World!')
    assert kb.get_size_entities() == 4
    for entity_string in ['Q53', 'Q17', 'Q007', 'Q44']:
        assert entity_string in kb.get_entity_strings()
    for entity_string in ['', 'Q0']:
        assert entity_string not in kb.get_entity_strings()
    assert kb.get_size_aliases() == 3
    for alias_string in ['double07', 'guy', 'random']:
        assert alias_string in kb.get_alias_strings()
    for alias_string in ['nothingness', '', 'randomnoise']:
        assert alias_string not in kb.get_alias_strings()
    candidates = sorted(kb.get_alias_candidates('double07'), key=lambda x: x.entity_)
    assert len(candidates) == 2
    assert candidates[0].entity_ == 'Q007'
    assert 6.999 < candidates[0].entity_freq < 7.01
    assert candidates[0].entity_vector == [0, 0, 7]
    assert candidates[0].alias_ == 'double07'
    assert 0.899 < candidates[0].prior_prob < 0.901
    assert candidates[1].entity_ == 'Q17'
    assert 1.99 < candidates[1].entity_freq < 2.01
    assert candidates[1].entity_vector == [7, 1, 0]
    assert candidates[1].alias_ == 'double07'
    assert 0.099 < candidates[1].prior_prob < 0.101

def test_serialize_subclassed_kb():
    if False:
        i = 10
        return i + 15
    'Check that IO of a custom KB works fine as part of an EL pipe.'
    config_string = '\n    [nlp]\n    lang = "en"\n    pipeline = ["entity_linker"]\n\n    [components]\n\n    [components.entity_linker]\n    factory = "entity_linker"\n    \n    [components.entity_linker.generate_empty_kb]\n    @misc = "kb_test.CustomEmptyKB.v1"\n    \n    [initialize]\n\n    [initialize.components]\n\n    [initialize.components.entity_linker]\n\n    [initialize.components.entity_linker.kb_loader]\n    @misc = "kb_test.CustomKB.v1"\n    entity_vector_length = 342\n    custom_field = 666\n    '

    class SubInMemoryLookupKB(InMemoryLookupKB):

        def __init__(self, vocab, entity_vector_length, custom_field):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(vocab, entity_vector_length)
            self.custom_field = custom_field

        def to_disk(self, path, exclude: Iterable[str]=SimpleFrozenList()):
            if False:
                print('Hello World!')
            'We overwrite InMemoryLookupKB.to_disk() to ensure that self.custom_field is stored as well.'
            path = ensure_path(path)
            if not path.exists():
                path.mkdir(parents=True)
            if not path.is_dir():
                raise ValueError(Errors.E928.format(loc=path))

            def serialize_custom_fields(file_path: Path) -> None:
                if False:
                    return 10
                srsly.write_json(file_path, {'custom_field': self.custom_field})
            serialize = {'contents': lambda p: self.write_contents(p), 'strings.json': lambda p: self.vocab.strings.to_disk(p), 'custom_fields': lambda p: serialize_custom_fields(p)}
            util.to_disk(path, serialize, exclude)

        def from_disk(self, path, exclude: Iterable[str]=SimpleFrozenList()):
            if False:
                for i in range(10):
                    print('nop')
            'We overwrite InMemoryLookupKB.from_disk() to ensure that self.custom_field is loaded as well.'
            path = ensure_path(path)
            if not path.exists():
                raise ValueError(Errors.E929.format(loc=path))
            if not path.is_dir():
                raise ValueError(Errors.E928.format(loc=path))

            def deserialize_custom_fields(file_path: Path) -> None:
                if False:
                    i = 10
                    return i + 15
                self.custom_field = srsly.read_json(file_path)['custom_field']
            deserialize: Dict[str, Callable[[Any], Any]] = {'contents': lambda p: self.read_contents(p), 'strings.json': lambda p: self.vocab.strings.from_disk(p), 'custom_fields': lambda p: deserialize_custom_fields(p)}
            util.from_disk(path, deserialize, exclude)

    @registry.misc('kb_test.CustomEmptyKB.v1')
    def empty_custom_kb() -> Callable[[Vocab, int], SubInMemoryLookupKB]:
        if False:
            while True:
                i = 10

        def empty_kb_factory(vocab: Vocab, entity_vector_length: int):
            if False:
                for i in range(10):
                    print('nop')
            return SubInMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length, custom_field=0)
        return empty_kb_factory

    @registry.misc('kb_test.CustomKB.v1')
    def custom_kb(entity_vector_length: int, custom_field: int) -> Callable[[Vocab], SubInMemoryLookupKB]:
        if False:
            print('Hello World!')

        def custom_kb_factory(vocab):
            if False:
                for i in range(10):
                    print('nop')
            kb = SubInMemoryLookupKB(vocab=vocab, entity_vector_length=entity_vector_length, custom_field=custom_field)
            kb.add_entity('random_entity', 0.0, zeros(entity_vector_length))
            return kb
        return custom_kb_factory
    config = Config().from_str(config_string)
    nlp = load_model_from_config(config, auto_fill=True)
    nlp.initialize()
    entity_linker = nlp.get_pipe('entity_linker')
    assert type(entity_linker.kb) == SubInMemoryLookupKB
    assert entity_linker.kb.entity_vector_length == 342
    assert entity_linker.kb.custom_field == 666
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        entity_linker2 = nlp2.get_pipe('entity_linker')
        assert type(entity_linker2.kb) == SubInMemoryLookupKB
        assert entity_linker2.kb.entity_vector_length == 342
        assert entity_linker2.kb.custom_field == 666