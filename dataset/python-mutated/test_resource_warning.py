import warnings
from unittest import TestCase
import pytest
import srsly
from numpy import zeros
from spacy.kb.kb_in_memory import InMemoryLookupKB, Writer
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir

def nlp():
    if False:
        return 10
    return Language()

def vectors():
    if False:
        while True:
            i = 10
    data = zeros((3, 1), dtype='f')
    keys = ['cat', 'dog', 'rat']
    return Vectors(data=data, keys=keys)

def custom_pipe():
    if False:
        i = 10
        return i + 15

    class SerializableDummy:

        def __init__(self, **cfg):
            if False:
                return 10
            if cfg:
                self.cfg = cfg
            else:
                self.cfg = None
            super(SerializableDummy, self).__init__()

        def to_bytes(self, exclude=tuple(), disable=None, **kwargs):
            if False:
                print('Hello World!')
            return srsly.msgpack_dumps({'dummy': srsly.json_dumps(None)})

        def from_bytes(self, bytes_data, exclude):
            if False:
                i = 10
                return i + 15
            return self

        def to_disk(self, path, exclude=tuple(), **kwargs):
            if False:
                print('Hello World!')
            pass

        def from_disk(self, path, exclude=tuple(), **kwargs):
            if False:
                print('Hello World!')
            return self

    class MyPipe(TrainablePipe):

        def __init__(self, vocab, model=True, **cfg):
            if False:
                return 10
            if cfg:
                self.cfg = cfg
            else:
                self.cfg = None
            self.model = SerializableDummy()
            self.vocab = vocab
    return MyPipe(Vocab())

def tagger():
    if False:
        i = 10
        return i + 15
    nlp = Language()
    tagger = nlp.add_pipe('tagger')
    tagger.add_label('A')
    nlp.initialize()
    return tagger

def entity_linker():
    if False:
        print('Hello World!')
    nlp = Language()

    def create_kb(vocab):
        if False:
            while True:
                i = 10
        kb = InMemoryLookupKB(vocab, entity_vector_length=1)
        kb.add_entity('test', 0.0, zeros((1,), dtype='f'))
        return kb
    entity_linker = nlp.add_pipe('entity_linker')
    entity_linker.set_kb(create_kb)
    nlp.initialize()
    return entity_linker
objects_to_test = ([nlp(), vectors(), custom_pipe(), tagger(), entity_linker()], ['nlp', 'vectors', 'custom_pipe', 'tagger', 'entity_linker'])

def write_obj_and_catch_warnings(obj):
    if False:
        return 10
    with make_tempdir() as d:
        with warnings.catch_warnings(record=True) as warnings_list:
            warnings.filterwarnings('always', category=ResourceWarning)
            obj.to_disk(d)
            return list(filter(lambda x: isinstance(x, ResourceWarning), warnings_list))

@pytest.mark.parametrize('obj', objects_to_test[0], ids=objects_to_test[1])
def test_to_disk_resource_warning(obj):
    if False:
        i = 10
        return i + 15
    warnings_list = write_obj_and_catch_warnings(obj)
    assert len(warnings_list) == 0

def test_writer_with_path_py35():
    if False:
        for i in range(10):
            print('nop')
    writer = None
    with make_tempdir() as d:
        path = d / 'test'
        try:
            writer = Writer(path)
        except Exception as e:
            pytest.fail(str(e))
        finally:
            if writer:
                writer.close()

def test_save_and_load_knowledge_base():
    if False:
        return 10
    nlp = Language()
    kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    with make_tempdir() as d:
        path = d / 'kb'
        try:
            kb.to_disk(path)
        except Exception as e:
            pytest.fail(str(e))
        try:
            kb_loaded = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
            kb_loaded.from_disk(path)
        except Exception as e:
            pytest.fail(str(e))

class TestToDiskResourceWarningUnittest(TestCase):

    def test_resource_warning(self):
        if False:
            for i in range(10):
                print('nop')
        scenarios = zip(*objects_to_test)
        for scenario in scenarios:
            with self.subTest(msg=scenario[1]):
                warnings_list = write_obj_and_catch_warnings(scenario[0])
                self.assertEqual(len(warnings_list), 0)